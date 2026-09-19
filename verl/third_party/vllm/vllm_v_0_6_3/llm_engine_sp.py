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

from functools import partial
from typing import Callable, Dict, Optional, Type, Union

import torch
import torch.nn as nn
from vllm.config import (
    CacheConfig,
    DecodingConfig,
    DeviceConfig,
    EngineConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    PromptAdapterConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm.core.scheduler import Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine, SchedulerContext, SchedulerOutputState, _load_generation_config_dict
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.sequence import Sequence
from vllm.tracing import init_tracer
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled, usage_message
from vllm.utils import Counter, weak_bind
from vllm.version import __version__ as VLLM_VERSION

from .arg_utils import EngineArgs
from .config import LoadConfig, ModelConfig
from .tokenizer import TokenizerGroup

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


class LLMEngine(LLMEngine):
    """接收请求并生成文本的 LLM 引擎。

    这是 vLLM 引擎的主类。它接收来自客户端的请求并从 LLM 生成文本。
    它包含一个 tokenizer、一个语言模型（可能通过张量并行分布在多个
    GPU 上），以及为中间状态（即 KV cache）分配的 GPU 显存空间。该类
    利用迭代级调度和高效的内存管理来最大化 serving 吞吐量。

    :class:`~vllm.LLM` 类封装该类用于离线批量推理，:class:`AsyncLLMEngine`
    类封装该类用于在线 serving。

    配置参数派生自 :class:`~vllm.EngineArgs`。（参见
    :ref:`engine_args`）

    Args:
        model_config: 与 LLM 模型相关的配置。
        cache_config: 与 KV cache 内存管理相关的配置。
        parallel_config: 与分布式执行相关的配置。
        scheduler_config: 与请求调度器相关的配置。
        device_config: 与设备相关的配置。
        lora_config (Optional): 与多 LoRA serving 相关的配置。
        speculative_config (Optional): 与投机解码（speculative decoding）
            相关的配置。
        executor_class: 用于管理分布式执行的模型 executor 类。
        prompt_adapter_config (Optional): 与 prompt adapter serving 相关
            的配置。
        log_stats: 是否记录统计信息。
        usage_context: 指定的入口点，用于收集使用信息。
    """

    def __init__(
        self,
        # NOTE(sgm): 前两个参数是为 verl 添加的
        model: Union[nn.Module, Dict],  # 模型本身或其参数字典
        tokenizer: nn.Module,
        # NOTE(sgm): vllm 原有参数
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        observability_config: Optional[ObservabilityConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
    ) -> None:
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, speculative_config=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "override_neuron_config=%s, "
            "rope_scaling=%r, rope_theta=%r, tokenizer_revision=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, tensor_parallel_size=%d, "
            "pipeline_parallel_size=%d, "
            "disable_custom_all_reduce=%s, quantization=%s, "
            "enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "decoding_config=%r, observability_config=%r, "
            "seed=%d, served_model_name=%s, use_v2_block_manager=%s, "
            "num_scheduler_steps=%d, chunked_prefill_enabled=%s "
            "multi_step_stream_outputs=%s, enable_prefix_caching=%s, "
            "use_async_output_proc=%s, use_cached_outputs=%s, "
            "mm_processor_kwargs=%s)",
            VLLM_VERSION,
            model_config.model,
            speculative_config,
            model_config.tokenizer,
            model_config.skip_tokenizer_init,
            model_config.tokenizer_mode,
            model_config.revision,
            model_config.override_neuron_config,
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
            scheduler_config.num_scheduler_steps,
            scheduler_config.chunked_prefill_enabled,
            scheduler_config.multi_step_stream_outputs,
            cache_config.enable_prefix_caching,
            model_config.use_async_output_proc,
            use_cached_outputs,
            model_config.mm_processor_kwargs,
        )
        # TODO(woosuk): 在调试模式下打印更多配置。
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.speculative_config = speculative_config
        self.load_config = load_config
        self.decoding_config = decoding_config or DecodingConfig()
        self.prompt_adapter_config = prompt_adapter_config
        self.observability_config = observability_config or ObservabilityConfig()
        self.log_stats = log_stats
        self.use_cached_outputs = use_cached_outputs

        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer(tokenizer)
            self.detokenizer = Detokenizer(self.tokenizer)
            tokenizer_group = self.get_tokenizer_group()
        else:
            self.tokenizer = None
            self.detokenizer = None
            tokenizer_group = None

        # 确保该函数不包含对 self 的引用，
        # 以避免引擎的 GC 问题
        def get_tokenizer_for_seq(sequence: Sequence) -> AnyTokenizer:
            assert tokenizer_group, "tokenizer_group cannot be None, " "make sure skip_tokenizer_init is False"
            return tokenizer_group.get_lora_tokenizer(sequence.lora_request)

        self.seq_counter = Counter()
        self.generation_config_fields = _load_generation_config_dict(model_config)

        self.input_preprocessor = InputPreprocessor(model_config, self.tokenizer)

        self.input_registry = input_registry
        self.input_processor = input_registry.create_input_processor(model_config)

        self.model_executor = executor_class(
            model=model,  # 为 spmd_gpu_executor 添加
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            speculative_config=speculative_config,
            load_config=load_config,
            prompt_adapter_config=prompt_adapter_config,
            observability_config=self.observability_config,
        )

        if not self.model_config.embedding_mode:
            self._initialize_kv_caches()

        # 若启用了使用统计，则收集相关信息。
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import get_architecture_class_name

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
                },
            )

        if self.tokenizer:
            # 若 tokenizer 运行在不同进程中，通过 ping 确保其存活。
            self.tokenizer.ping()

        self.cached_scheduler_outputs = [
            SchedulerOutputState() for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        self.scheduler_contexts = [
            SchedulerContext(multi_step_stream_outputs=self.scheduler_config.multi_step_stream_outputs)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        if model_config.use_async_output_proc:
            process_model_outputs = weak_bind(self._process_model_outputs)

            self.async_callbacks = [
                partial(process_model_outputs, ctx=self.scheduler_contexts[v_id])
                for v_id in range(self.parallel_config.pipeline_parallel_size)
            ]
        else:
            self.async_callbacks = []

        # AsyncLLMEngine 目前使用它来确保将请求输出快速追加
        # 到 asyncio 队列中
        self.process_request_outputs_callback: Optional[Callable] = None

        # 创建调度器。
        # NOTE: 这里的 cache_config 已更新为 GPU 和 CPU block 的数量，
        # 它们是在分布式 executor 中分析得到的。
        self.scheduler = [
            Scheduler(
                scheduler_config,
                cache_config,
                lora_config,
                parallel_config.pipeline_parallel_size,
                self.async_callbacks[v_id] if model_config.use_async_output_proc else None,
            ) for v_id in range(parallel_config.pipeline_parallel_size)
        ]

        # 指标日志记录。
        if self.log_stats:
            if stat_loggers is not None:
                self.stat_loggers = stat_loggers
            else:
                # 延迟导入 prometheus multiprocessing。
                # 需要在导入 prometheus_client 之前设置 PROMETHEUS_MULTIPROC_DIR
                # 环境变量。
                # 参见 https://prometheus.github.io/client_python/multiprocess/
                from vllm.engine.metrics import LoggingStatLogger, PrometheusStatLogger

                self.stat_loggers = {
                    "logging":
                        LoggingStatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC),
                    "prometheus":
                        PrometheusStatLogger(
                            local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                            labels=dict(model_name=model_config.served_model_name),
                            max_model_len=self.model_config.max_model_len,
                        ),
                }
                self.stat_loggers["prometheus"].info("cache_config", self.cache_config)

        self.tracer = None
        if self.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer("vllm.llm_engine", self.observability_config.otlp_traces_endpoint)

        # 创建序列输出处理器，例如用于 beam search 或投机解码。
        self.output_processor = SequenceGroupOutputProcessor.create_output_processor(
            self.scheduler_config,
            self.detokenizer,
            self.scheduler,
            self.seq_counter,
            get_tokenizer_for_seq,
            stop_checker=StopChecker(
                self.scheduler_config.max_model_len,
                get_tokenizer_for_seq,
            ),
        )

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
        distributed_executor_backend = engine_config.parallel_config.distributed_executor_backend
        # 初始化集群并指定 executor 类。]
        assert (engine_config.device_config.device_type == "cuda"
               ), "Currently, the vllm in verl only support running on GPU"

        # print('Waiting for debugger'); import os,debugpy; debugpy.listen(('localhost', 5678 + int(os.getenv('RANK', '0')))); debugpy.wait_for_client()
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
        assert (engine_config.device_config.device_type == "cuda"
               ), "Currently, the vllm in verl only support running on GPU"

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
