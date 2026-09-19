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
# 改编自 https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py

import torch
from typing import Dict, Optional, Union, Type

import vllm.envs as envs
from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, EngineConfig, LoRAConfig, MultiModalConfig,
                         ObservabilityConfig, ParallelConfig, PromptAdapterConfig, SchedulerConfig, SpeculativeConfig)
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import (SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import INPUT_REGISTRY, LLMInputs, PromptInputs
from vllm.logger import init_logger
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.engine.metrics import (LoggingStatLogger, PrometheusStatLogger, StatLoggerBase, Stats)
from vllm.tracing import (SpanAttributes, SpanKind, extract_trace_context, init_tracer)
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled, usage_message)
from vllm.utils import Counter
from vllm.engine.llm_engine import _load_generation_config_dict
from vllm.engine.llm_engine import LLMEngine
from vllm.version import __version__ as VLLM_VERSION

import torch.nn as nn
from .arg_utils import EngineArgs
from .tokenizer import TokenizerGroup
from .config import ModelConfig, LoadConfig

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


class LLMEngine(LLMEngine):
    """接收请求并生成文本的 LLM 引擎。

    这是 vLLM 引擎的主类。它接收来自客户端的请求并从 LLM 生成文本。
    它包含一个 tokenizer、一个语言模型（可能通过张量并行分布在多个
    GPU 上），以及为中间状态（即 KV cache）分配的 GPU 显存空间。该类
    利用迭代级调度和高效的内存管理来最大化 serving 吞吐量。

    `LLM` 类封装该类用于离线批量推理，`AsyncLLMEngine` 类封装该类
    用于在线 serving。

    NOTE: 配置参数派生自 `EngineArgs` 类。完整的参数列表请参见
    `EngineArgs`。

    Args:
        model: 在 vllm 之外初始化的 actor 模型（为 verl 添加）
        tokenizer: 已初始化的 tokenizer（为 verl 添加）
        model_config: 与 LLM 模型相关的配置。
        cache_config: 与 KV cache 内存管理相关的配置。
        parallel_config: 与分布式执行相关的配置。
        scheduler_config: 与请求调度器相关的配置。
        distributed_init_method: 分布式执行的初始化方法。详见
            `torch.distributed.init_process_group`。
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
        multimodal_config: Optional[MultiModalConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        observability_config: Optional[ObservabilityConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> None:
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, speculative_config=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, revision=%s, "
            "rope_scaling=%r, rope_theta=%r, tokenizer_revision=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, tensor_parallel_size=%d, "
            "pipeline_parallel_size=%d, "
            "disable_custom_all_reduce=%s, quantization=%s, "
            "enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "decoding_config=%r, observability_config=%r, "
            "seed=%d, served_model_name=%s, use_v2_block_manager=%s, "
            "enable_prefix_caching=%s)",
            VLLM_VERSION,
            model_config.model,
            speculative_config,
            model_config.tokenizer,
            model_config.skip_tokenizer_init,
            model_config.revision,
            model_config.rope_scaling,
            model_config.rope_theta,
            model_config.tokenizer_revision,
            model_config.trust_remote_code,
            model_config.dtype,
            model_config.max_model_len,
            load_config.download_dir,
            load_config.load_format,
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
            parallel_config.disable_custom_all_reduce,
            model_config.quantization,
            model_config.enforce_eager,
            cache_config.cache_dtype,
            model_config.quantization_param_path,
            device_config.device,
            decoding_config,
            observability_config,
            model_config.seed,
            model_config.served_model_name,
            scheduler_config.use_v2_block_manager,
            cache_config.enable_prefix_caching,
        )
        # TODO(woosuk): 在调试模式下打印更多配置。

        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.multimodal_config = multimodal_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.speculative_config = speculative_config
        self.load_config = load_config
        self.decoding_config = decoding_config or DecodingConfig()
        self.prompt_adapter_config = prompt_adapter_config
        self.observability_config = observability_config or ObservabilityConfig()
        self.log_stats = log_stats

        # self.model = model # 不应存储模型，应当删除
        # TODO(shengguangming): 或许可以选择在这里初始化或从参数传入
        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer(tokenizer)
            self.detokenizer = Detokenizer(self.tokenizer)
        else:
            self.tokenizer = None
            self.detokenizer = None

        self.seq_counter = Counter()
        self.generation_config_fields = _load_generation_config_dict(model_config)

        self.input_processor = INPUT_REGISTRY.create_input_processor(self.model_config)

        self.model_executor = executor_class(
            model=model, # 为 spmd_gpu_executor 添加
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            multimodal_config=multimodal_config,
            speculative_config=speculative_config,
            load_config=load_config,
            prompt_adapter_config=prompt_adapter_config,
        )

        # 分析内存使用情况并初始化缓存。
        if not self.model_config.embedding_mode:
            self._initialize_kv_caches()

        # 若启用了使用统计，则收集相关信息。
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import (get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(model_config),
                usage_context,
                extra_kvs={
                    # 常规配置
                    "dtype": str(model_config.dtype),
                    "tensor_parallel_size": parallel_config.tensor_parallel_size,
                    "block_size": cache_config.block_size,
                    "gpu_memory_utilization": cache_config.gpu_memory_utilization,

                    # 量化
                    "quantization": model_config.quantization,
                    "kv_cache_dtype": str(cache_config.cache_dtype),

                    # 功能开关
                    "enable_lora": bool(lora_config),
                    "enable_prompt_adapter": bool(prompt_adapter_config),
                    "enable_prefix_caching": cache_config.enable_prefix_caching,
                    "enforce_eager": model_config.enforce_eager,
                    "disable_custom_all_reduce": parallel_config.disable_custom_all_reduce,
                })

        if self.tokenizer:
            # 若 tokenizer 运行在不同进程中，通过 ping 确保其存活。
            self.tokenizer.ping()

        # 创建调度器。
        # NOTE: 这里的 cache_config 已更新为 GPU 和 CPU block 的数量，
        # 它们是在分布式 executor 中分析得到的。
        self.scheduler = [
            Scheduler(scheduler_config, cache_config, lora_config, parallel_config.pipeline_parallel_size)
            for _ in range(parallel_config.pipeline_parallel_size)
        ]

        # 指标日志记录。
        if self.log_stats:
            if stat_loggers is not None:
                self.stat_loggers = stat_loggers
            else:
                self.stat_loggers = {
                    "logging":
                        LoggingStatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC),
                    "prometheus":
                        PrometheusStatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                                             labels=dict(model_name=model_config.served_model_name),
                                             max_model_len=self.model_config.max_model_len),
                }
                self.stat_loggers["prometheus"].info("cache_config", self.cache_config)

        self.tracer = None
        if self.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer("vllm.llm_engine", self.observability_config.otlp_traces_endpoint)

        # 创建序列输出处理器，例如用于 beam search 或投机解码。
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
        return TokenizerGroup(tokenizer, **init_kwargs)

    def init_cache_engine(self):
        # TODO: 检查 offload/load KVCache 时是否应该每轮迭代都重建 CUDAGraph
        # 重新捕获 CUDAGraph 非常耗时
        self.model_executor.init_cache_engine()

    def free_cache_engine(self):
        self.model_executor.free_cache_engine()

    # NOTE(sgm): 目前我们仅支持 GPU executor
    # GPUExecutor 去除了对 Ray 的依赖
    @classmethod
    def _get_executor_cls(cls, engine_config: EngineConfig) -> Type[ExecutorBase]:
        assert engine_config.device_config.device_type == "cuda", \
            "Currently, the vllm in verl only support running on GPU"

        if engine_config.parallel_config.world_size == 1:
            engine_config.load_config.load_format = "dummy_hf"

        from .spmd_gpu_executor import SPMDGPUExecutor
        executor_class = SPMDGPUExecutor
        return executor_class

    @classmethod
    def from_engine_args(
        cls,
        model,
        tokenizer,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "LLMEngine":
        """根据引擎参数创建 LLM 引擎。"""
        # 创建引擎配置。
        engine_config = engine_args.create_engine_config()
        executor_class = cls._get_executor_cls(engine_config)
        # 初始化集群并指定 executor 类。
        assert engine_config.device_config.device_type == "cuda", \
            "Currently, the vllm in verl only support running on GPU"

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
            stat_loggers=stat_loggers,
        )
        return engine

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        self.model_executor.sync_model_weights(actor_weights=actor_weights, load_format=load_format)

    def offload_model_weights(self) -> None:
        self.model_executor.offload_model_weights()
