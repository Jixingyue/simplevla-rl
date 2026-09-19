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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py

import torch
from typing import Dict, Optional, Union, Type

import vllm
from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, LoRAConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig, VisionLanguageConfig)
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import (SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.engine.metrics import StatLogger
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled, usage_message)
from vllm.utils import Counter
from vllm.engine.llm_engine import _load_generation_config_dict
from vllm.engine.llm_engine import LLMEngine

import torch.nn as nn
from .arg_utils import EngineArgs
from .tokenizer import TokenizerGroup
from .config import ModelConfig, LoadConfig

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


class LLMEngine(LLMEngine):
    """接收请求并生成文本的 LLM 引擎。

    这是 vLLM 引擎的主类。它接收来自客户端的请求，并由 LLM 生成文本。
    它包含一个 tokenizer、一个语言模型（可能分布在多个 GPU 上），
    以及为中间状态（即 KV cache）分配的 GPU 显存空间。该类利用
    迭代级调度和高效的内存管理来最大化服务吞吐量。

    `LLM` 类封装了此类用于离线批量推理，`AsyncLLMEngine` 类封装了
    此类用于在线服务。

    NOTE: 配置参数派生自 `EngineArgs` 类。完整的参数列表请参见
    `EngineArgs`。

    参数：
        model: 在 vllm 外部初始化的 actor 模型（为 verl 添加）
        tokenizer: 已初始化的 tokenizer（为 verl 添加）
        model_config: 与 LLM 模型相关的配置。
        cache_config: 与 KV cache 内存管理相关的配置。
        parallel_config: 与分布式执行相关的配置。
        scheduler_config: 与请求调度器相关的配置。
        distributed_init_method: 分布式执行的初始化方法。
            详情参见 `torch.distributed.init_process_group`。
        placement_group: 用于分布式执行的 Ray placement group。
            分布式执行时必需。
        log_stats: 是否记录统计信息。
    """

    def __init__(
        self,
        # NOTE(sgm): 前两个参数是为 verl 添加的
        model: Union[nn.Module, Dict], # 模型本身或其参数字典
        tokenizer: nn.Module,
        # NOTE(sgm): vllm 原有参数
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> None:
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, speculative_config=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "tokenizer_revision=%s, trust_remote_code=%s, dtype=%s, "
            "max_seq_len=%d, download_dir=%r, load_format=%s, "
            "tensor_parallel_size=%d, disable_custom_all_reduce=%s, "
            "quantization=%s, enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "decoding_config=%r, seed=%d, served_model_name=%s)",
            vllm.__version__,
            model_config.model,
            speculative_config,
            model_config.tokenizer,
            model_config.skip_tokenizer_init,
            # model_config.tokenizer_mode,
            model_config.revision,
            model_config.tokenizer_revision,
            # model_config.trust_remote_code,
            model_config.dtype,
            model_config.max_model_len,
            load_config.download_dir,
            load_config.load_format,
            parallel_config.tensor_parallel_size,
            parallel_config.disable_custom_all_reduce,
            model_config.quantization,
            model_config.enforce_eager,
            cache_config.cache_dtype,
            model_config.quantization_param_path,
            device_config.device,
            decoding_config,
            model_config.seed,
            # model_config.served_model_name,
        )
        # TODO(woosuk): 在调试模式下打印更多配置。

        self.model_config = model_config  # TODO: 当前是 hfconfig
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.vision_language_config = vision_language_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.speculative_config = speculative_config
        self.load_config = load_config
        self.decoding_config = decoding_config or DecodingConfig()
        self.log_stats = log_stats

        # self.model = model # 不应存储模型，应将其删除
        # TODO(shengguangming): 也许可以选择在这里初始化或从参数传入
        if not self.model_config.skip_tokenizer_init:
            # TODO: 检查 tokenizer 类
            self._init_tokenizer(tokenizer)
            self.detokenizer = Detokenizer(self.tokenizer)
        else:
            self.detokenizer = None
            self.tokenizer = None

        self.seq_counter = Counter()
        # TODO: 不清楚它的用途
        self.generation_config_fields = _load_generation_config_dict(model_config)

        self.model_executor = executor_class(
            model=model, # 为 spmd_gpu_executor 添加
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            vision_language_config=vision_language_config,
            speculative_config=speculative_config,
            load_config=load_config,
        )

        # 分析内存占用并初始化 cache。
        self._initialize_kv_caches()

        # 如果启用了使用统计，则收集相关信息。
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import (get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(model_config),
                usage_context,
                extra_kvs={
                    # 通用配置
                    "dtype": str(model_config.dtype),
                    "tensor_parallel_size": parallel_config.tensor_parallel_size,
                    "block_size": cache_config.block_size,
                    "gpu_memory_utilization": cache_config.gpu_memory_utilization,

                    # 量化
                    "quantization": model_config.quantization,
                    "kv_cache_dtype": cache_config.cache_dtype,

                    # 功能开关
                    "enable_lora": bool(lora_config),
                    "enable_prefix_caching": cache_config.enable_prefix_caching,
                    "enforce_eager": model_config.enforce_eager,
                    "disable_custom_all_reduce": parallel_config.disable_custom_all_reduce,
                })

        if self.tokenizer:
            # 如果 tokenizer 运行在
            # 不同的进程中，则通过 ping 来确保其存活。
            self.tokenizer.ping()

        # 创建调度器。
        # NOTE: 这里的 cache_config 已更新为 GPU 和 CPU block 的数量，
        # 它们是在分布式 executor 中进行 profiling 得到的。
        # NOTE(shengguangming): 每个进程都拥有独立的调度器
        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

        # 指标日志记录。
        if self.log_stats:
            self.stat_logger = StatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                                          labels=dict(model_name=model_config.served_model_name),
                                          max_model_len=self.model_config.max_model_len)
            self.stat_logger.info("cache_config", self.cache_config)

        # 创建序列输出处理器，例如用于 beam search 或
        # 投机解码。
        self.output_processor = (SequenceGroupOutputProcessor.create_output_processor(
            self.scheduler_config,
            self.detokenizer,
            self.scheduler,
            self.seq_counter,
            self.get_tokenizer_for_seq,
            stop_checker=StopChecker(
                self.scheduler_config.max_model_len,
                self.get_tokenizer_for_seq,
            ),
        ))

    # TODO(sgm): 为 verl 添加，但 Rollout 中可能不需要 tokenizer
    def _init_tokenizer(self, tokenizer, **tokenizer_init_kwargs):
        init_kwargs = dict(enable_lora=bool(self.lora_config),
                           max_num_seqs=self.scheduler_config.max_num_seqs,
                           max_input_length=None)
        init_kwargs.update(tokenizer_init_kwargs)
        self.tokenizer: TokenizerGroup = TokenizerGroup(tokenizer, **init_kwargs)

    def init_cache_engine(self):
        # TODO: 检查在 offload/load KVCache 时是否应该每次迭代重建 CUDAGraph
        # 重新捕获 CUDAGraph 会非常耗时
        self.model_executor.init_cache_engine()

    def free_cache_engine(self):
        self.model_executor.free_cache_engine()

    # NOTE(sgm): 目前我们只支持 GPU executor
    # GPUExecutor 移除了对 Ray 的依赖
    @classmethod
    def from_engine_args(
        cls,
        model,
        tokenizer,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "LLMEngine":
        """根据引擎参数创建 LLM 引擎。"""
        # 创建引擎配置。
        engine_config = engine_args.create_engine_config()

        # 初始化集群并指定 executor 类。
        assert engine_config.device_config.device_type == "cuda", \
            "Currently, the vllm in verl only support running on GPU"

        if engine_config.parallel_config.world_size == 1:
            engine_config.load_config.load_format = "dummy_hf"

        from .spmd_gpu_executor import SPMDGPUExecutor
        executor_class = SPMDGPUExecutor

        # 创建 LLM 引擎。
        engine = cls(
            model,
            tokenizer,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
        )
        return engine

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        self.model_executor.sync_model_weights(actor_weights=actor_weights, load_format=load_format)

    def offload_model_weights(self) -> None:
        self.model_executor.offload_model_weights()
