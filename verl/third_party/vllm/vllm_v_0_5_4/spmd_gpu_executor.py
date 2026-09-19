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
# 改编自 https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py

import os
import socket
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import vllm.envs as envs
from vllm.executor.executor_base import ExecutorBase, ExecutorAsyncBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput, ExecuteModelRequest

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, MultiModalConfig, ParallelConfig, PromptAdapterConfig,
                         SchedulerConfig, SpeculativeConfig)
from .config import ModelConfig, LoadConfig

logger = init_logger(__name__)


class SPMDGPUExecutor(ExecutorBase):
    """基于 SPMD 的多 GPU executor 实现。"""

    def __init__(
        self,
        model, # pytorch 模型本身或其参数字典
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        multimodal_config: Optional[MultiModalConfig],
        speculative_config: Optional[SpeculativeConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.multimodal_config = multimodal_config
        self.speculative_config = speculative_config
        self.prompt_adapter_config = prompt_adapter_config

        distributed_init_method = initialize_cluster(parallel_config)
        self._init_executor(model, distributed_init_method)

    # TODO(sgm): verl 目前不支持投机解码
    def _init_executor(self, model, distributed_init_method) -> None:
        assert (not self.speculative_config), "Speculative decoding not yet supported for multi-GPU backend."

        # 为每个 GPU 创建并行 worker。
        self._init_workers_sp(model, distributed_init_method)

    def _init_workers_sp(self, model, distributed_init_method: str):
        # 延迟导入 Worker，避免在 Worker 中设置 CUDA_VISIBLE_DEVICES
        # 之前就导入 torch.cuda/xformers
        from .worker import Worker  # pylint: disable=import-outside-toplevel

        rank = int(os.getenv("RANK"))
        local_rank = int(os.getenv("LOCAL_RANK"))
        print(f'local rank {local_rank}')

        # 参见 https://github.com/NVIDIA/nccl/issues/1234
        os.environ['NCCL_CUMEM_ENABLE'] = '0'

        self.worker = Worker(
            model,
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            self.cache_config,
            self.load_config,
            local_rank,
            rank,
            distributed_init_method,
            lora_config=self.lora_config,
            multimodal_config=self.multimodal_config,
            speculative_config=None,
            prompt_adapter_config=self.speculative_config,
            is_driver_worker=True,
            model_runner_cls=None,  # 使用默认值
        )

        # NOTE(shengguangming): torch.distributed.init_process_group 会在 init_model() 内部被调用
        self.worker.init_device()
        self.worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """确定可用的 KV block 数量。

        该方法会在每个 worker 上调用 `determine_num_available_blocks`，
        并取各结果的最小值，从而保证所选的缓存大小对所有 worker 都
        兼容。

        Returns:
            - tuple[num_gpu_blocks, num_cpu_blocks]
        """
        # 获取 GPU 和 CPU 上可分配的最大 block 数量。
        num_blocks = self.worker.determine_num_available_blocks()

        # NOTE(shengguangming): 现在我们不使用共享的集中式控制器，
        # 而是每个进程拥有自己的调度器
        num_gpu_blocks = num_blocks[0]
        num_cpu_blocks = num_blocks[1]

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """在所有 worker 中初始化 KV cache。
        """

        # NOTE: 我们在这里记录日志，以避免 worker 数量大于一时出现重复日志。
        # 我们可以在引擎中记录，但并非所有 executor 都有 GPU。
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks, num_cpu_blocks)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        if torch.distributed.get_rank() == 0:
            print(
                f'before init cache memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB'
            )
        self.worker.initialize_cache(num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=num_cpu_blocks)
        if torch.distributed.get_rank() == 0:
            print(
                f'after init cache memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB'
            )

    # NOTE(sgm): 重建 KVCache 时不会对模型进行性能分析或捕获（CUDAGraph）
    def init_cache_engine(self) -> None:
        self.worker._init_cache_engine()

    def free_cache_engine(self) -> None:
        self.worker.free_cache_engine()

    def execute_model(self, execute_model_req) -> List[SamplerOutput]:
        all_outputs = self.worker.execute_model(execute_model_req=execute_model_req)

        # NOTE(sgm):
        # verl 下的 vllm 中每个 GPU 都有自己的 spmd_gpu_executor，因此所有 GPU 都应返回输出
        # 而在使用 Ray 的 vllm 中，只有 driver worker 返回采样结果。
        return all_outputs

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.worker.add_lora(lora_request=lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.worker.remove_lora(lora_id=lora_id)

    def list_loras(self) -> Set[int]:
        return self.worker.list_loras()

    def check_health(self) -> None:
        # 只要 SPMDExecutor 还在运行，它就始终是健康的。
        return

    # NOTE(sgm) 为 verl 添加以通过抽象类测试，并未使用
    from vllm.prompt_adapter.request import PromptAdapterRequest

    def add_prompt_adapter(self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        assert prompt_adapter_request.prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return self.worker.add_prompt_adapter(prompt_adapter_request)

    def list_prompt_adapters(self) -> Set[int]:
        return self.worker.list_prompt_adapters()

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.worker.pin_lora(lora_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
                "prompt_adapter_id must be greater than 0."
        return self.worker.pin_prompt_adapter(prompt_adapter_id)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return self.worker.remove_prompt_adapter(prompt_adapter_id)

    # NOTE(sgm): 为 verl 添加
    def offload_model_weights(self) -> None:
        self.worker.offload_model_weights()

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        self.worker.sync_model_weights(actor_weights=actor_weights, load_format=load_format)


def initialize_cluster(
    parallel_config: ParallelConfig,
    engine_use_ray: bool = False,
    ray_address: Optional[str] = None,
) -> Tuple[str, Optional[None]]:
    """初始化分布式集群（可能会使用 Ray）。

    Args:
        parallel_config: 并行执行的配置。

    Returns:
        `distributed_init_method` 是用于初始化分布式后端的地址。
    """

    # 在本地初始化集群。
    port = get_open_port()
    # 我们需要设置分布式初始化方法，以确保
    # 分布式 megatron 代码（例如获取 world size）正常工作。
    # distributed_init_method = f"tcp://localhost:{port}"
    distributed_init_method = 'env://'
    return distributed_init_method


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# TODO(sgm): 尚未实现 async executor
class SPMDGPUExecutorAsync(SPMDGPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(self, execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        """在给定的序列上执行一步模型推理。"""
        raise NotImplementedError

    async def check_health_async(self) -> None:
        """检查 executor 是否健康。如果不健康，应当抛出
        异常。"""
        self.check_health()
