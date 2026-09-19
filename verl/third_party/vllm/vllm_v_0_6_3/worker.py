# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 改编自 https://github.com/vllm-project/vllm/blob/main/vllm/worker/worker.py
"""一个 GPU worker 类。"""
import gc
import os
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed
import torch.nn as nn
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoRAConfig,
    ParallelConfig,
    PromptAdapterConfig,
    SchedulerConfig,
    SpeculativeConfig,
)

# TODO(sgm): 检查为什么 vllm 在 vllm.model_executor.parallel_utils.parallel_state 中有类似的文件
from vllm.distributed import get_tensor_model_parallel_group, init_distributed_environment, set_custom_all_reduce
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.worker.model_runner_base import ModelRunnerInputBase
from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype
from vllm.worker.worker_base import WorkerInput

from .config import LoadConfig, LoadFormat, ModelConfig
from .dtensor_weight_loaders import load_dtensor_weights
from .hf_weight_loader import load_hf_weights
from .megatron_weight_loaders import load_megatron_weights
from .model_runner import ModelRunner
from .parallel_state import ensure_model_parallel_initialized


class Worker(Worker):
    """在 GPU 上执行模型（的一部分）的 worker 类。

    每个 worker 绑定一块 GPU。worker 负责维护 KV cache 并在 GPU 上
    执行模型。在分布式推理的情况下，每个 worker 会被分配模型的
    一个分片。
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict],  # 模型本身或其参数字典
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        # self.model = model  # 将在 init_model 中被替换
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.parallel_config.rank = rank
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.prompt_adapter_config = prompt_adapter_config
        self.is_driver_worker = is_driver_worker  # TODO: 我们不需要 driver
        # if parallel_config and is_driver_worker:
        #     assert rank % parallel_config.tensor_parallel_size == 0, \
        #            "Driver worker should be rank 0 of tensor parallel group."
        if self.model_config.trust_remote_code:
            # 注意：延迟导入，避免在初始化之前导入 torch
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        # 如果 draft 模型是 mlp_speculator，则返回目标模型的 hidden states
        speculative_args = (
            {} if speculative_config is None or (speculative_config.draft_model_config.model == model_config.model) or
            (speculative_config.draft_model_config.hf_config.model_type not in ["medusa", "mlp_speculator"]) else {
                "return_hidden_states": True
            })

        # TODO(sgm): 设置正确的 model runner 类
        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_runner_cls is not None:
            ModelRunnerClass = model_runner_cls
        elif self.model_config.embedding_mode:
            ModelRunnerClass = EmbeddingModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            model,  # [VERL]: 为 verl 添加
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            prompt_adapter_config=prompt_adapter_config,
            **speculative_args,
        )

        # 未初始化的 cache engine。将由 initialize_cache 初始化。
        self.cache_engine: List[CacheEngine] = None
        # 初始化 gpu_cache，因为 embedding 模型不会初始化 kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None

        # NOTE(sgm): [VERL] 用于卸载（offload）推理引擎参数
        self.cpu_model = None

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce 在到达同步点之前不会释放输入
            # 张量。这会导致随着 all_reduce 调用次数增加，内存占用不断
            # 增长。该环境变量可以禁用这种行为。
            # 相关 issue：
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # NOTE(sgm): 为 verl 修改，环境变量将由 TORCHRUN 设置。
            self.rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{local_rank}")
            if self.rank < 0:
                raise ValueError("Invalid or unspecified rank.")
            torch.cuda.set_device(self.device)

            # 使用 TORCHRUN 设置的 world_size
            world_size = int(os.getenv("WORLD_SIZE", "-1"))
            assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
            self.parallel_config.world_size = world_size

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # 初始化分布式环境。
        init_worker_distributed_environment(self.parallel_config, self.rank, self.distributed_init_method,
                                            self.local_rank)
        # 设置随机种子。
        set_random_seed(self.model_config.seed)
        # self.model = get_model(actor_model=self.model, model_config=self.model_config)

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """分析模型的峰值内存占用，以确定在不发生 OOM 的情况下
        可以分配多少个 KV block。

        引擎会首先对当前的内存占用进行分析，然后计算在剩余空闲
        内存中最多可以分配的 GPU 和 CPU block 数量。

        .. tip::
            你可以通过调整 `gpu_memory_utilization` 参数来限制
            GPU 内存的使用。
        """
        # 分析模型的内存占用，并得到在剩余空闲内存中可分配的
        # cache block 最大数量。
        torch.cuda.empty_cache()
        # torch.cuda.reset_peak_memory_stats()

        # 使用 dummy 输入执行一次前向传播，以分析模型的内存占用。
        self.model_runner.profile_run()

        # 根据分析得到的峰值内存，计算可分配的 block 数量。
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = total_gpu_memory - free_gpu_memory

        assert peak_memory > 0, ("Error in memory profiling. This happens when the GPU memory was "
                                 "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes()

        # NOTE(sgm) [VERL] 使用剩余内存
        num_gpu_blocks = int((free_gpu_memory * self.cache_config.gpu_memory_utilization) // cache_block_size)
        # num_gpu_blocks = int((total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_memory) // cache_block_size)

        num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()

        # NOTE(sgm): 为 [VERL] 添加，在所有 rank 之间同步 block 数量
        num_gpu_blocks = torch.tensor([num_gpu_blocks], device="cuda")
        num_cpu_blocks = torch.tensor([num_cpu_blocks], device="cuda")

        torch.distributed.all_reduce(num_gpu_blocks,
                                     op=torch.distributed.ReduceOp.MIN,
                                     group=get_tensor_model_parallel_group().device_group)
        torch.distributed.all_reduce(num_cpu_blocks,
                                     op=torch.distributed.ReduceOp.MIN,
                                     group=get_tensor_model_parallel_group().device_group)
        num_gpu_blocks = num_gpu_blocks.item()
        num_cpu_blocks = num_cpu_blocks.item()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def _init_cache_engine(self):
        if self.cache_engine is None and self.gpu_cache is None:
            super()._init_cache_engine()

    def free_cache_engine(self):
        # 确保 `enforce_eager=True`
        self.cache_engine = None
        self.gpu_cache = None

    # NOTE(sgm): [VERL]: 改编自 _execute_model_spmd()
    def execute_model(self,
                      execute_model_req: ExecuteModelRequest,
                      intermediate_tensors: Optional[IntermediateTensors] = None) -> Optional[List[SamplerOutput]]:
        """
        以单程序多数据（SPMD）方式执行模型。
        所有 worker 接收相同的请求，准备输入并执行模型。
        """
        assert execute_model_req is not None, ("_execute_model_spmd() requires each worker to take in an "
                                               "ExecuteModelRequest")
        worker_input: WorkerInput = self.prepare_worker_input(execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = self.model_runner.prepare_model_input(
            execute_model_req.seq_group_metadata_list)

        # verl.worker.workerbase.WorkerBase
        # 交换 cache
        super().execute_worker(worker_input)

        # 如果没有输入，则无需执行模型。
        if worker_input.num_seq_groups == 0:
            return []

        return self.model_runner.execute_model(
            model_input,
            self.kv_cache[worker_input.virtual_engine] if self.kv_cache is not None else None,
            intermediate_tensors,
        )

    # 假设输入是 .state_dict()
    def sync_model_weights(self, actor_weights: Dict, load_format: str):
        if load_format in [LoadFormat.MEGATRON, LoadFormat.AUTO]:
            load_megatron_weights(actor_weights, self.model_runner.model)
        elif load_format == LoadFormat.HF:
            # 完整的模型 state dict，无分片
            load_hf_weights(actor_weights, self.model_runner.model)
        elif load_format == LoadFormat.DTENSOR:
            load_dtensor_weights(actor_weights, self.model_runner.model)

    def offload_model_weights(self) -> None:
        if self.cpu_model == None:
            self.cpu_model = {}
            for name, params in self.model_runner.model.named_parameters():
                self.cpu_model[name] = torch.empty_like(params, device="cpu")
                params.data = self.cpu_model[name]
        else:
            for name, params in self.model_runner.model.named_parameters():
                params.data = self.cpu_model[name]


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = "env://",
    local_rank: int = -1,
) -> None:
    """初始化分布式环境。"""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    # NOTE(sgm) 使用 tcp://localhost:xxxx 在没有 megatron 的 HF 设置下会挂起
    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank)

    ensure_model_parallel_initialized(
        tensor_model_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_parallel_size,
    )

    # TODO(sgm): 检查是否需要这段代码
    # if pynccl_utils.is_initialized():
    #     pynccl_world_size = pynccl_utils.get_world_size()
    #     if pynccl_world_size != parallel_config.world_size:
    #         raise RuntimeError(
    #             "pynccl is already initialized but the pynccl world "
    #             "size does not match parallel_config.world_size "
    #             f"({pynccl_world_size} vs. {parallel_config.world_size}).")
    # elif parallel_config.world_size > 1:
    #     # NOTE(woosuk): 当 world size 为 1 时，不初始化 pynccl 进程组。
    #     # NOTE(kaichao): 默认情况下，pynccl 为 tp 组初始化。
    #     pynccl_utils.init_process_group(
    #         group=get_tensor_model_parallel_cpu_group())

    # # 初始化自定义的快速 all-reduce 实现。
    # if not parallel_config.disable_custom_all_reduce:
    #     init_custom_ar()

    # 用于预热的少量 all_reduce。
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    # if pynccl_utils.is_initialized():
    #     pynccl_utils.all_reduce(torch.zeros(1).cuda())
