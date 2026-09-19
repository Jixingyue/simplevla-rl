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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/config.py

from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import torch
from transformers import PretrainedConfig
from packaging.version import Version

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, get_nvcc_cuda_version

logger = init_logger(__name__)

_GB = 1 << 30


class ModelConfig:
    """模型配置。

    Args:
        model: 要使用的 huggingface 模型的名称或路径。
        tokenizer: 要使用的 huggingface tokenizer 的名称或路径。
        tokenizer_mode: Tokenizer 模式。"auto" 会在可用时使用快速 tokenizer，
            "slow" 则始终使用慢速 tokenizer。
        trust_remote_code: 下载模型和 tokenizer 时是否信任远程代码
            （例如来自 HuggingFace 的代码）。
        download_dir: 下载和加载权重的目录，默认为 huggingface 的
            默认缓存目录。
        load_format: 要加载的模型权重格式：
            "auto" 会尝试以 safetensors 格式加载权重，如果该格式
                不可用则回退到 pytorch bin 格式。
            "pt" 会以 pytorch bin 格式加载权重。
            "safetensors" 会以 safetensors 格式加载权重。
            "npcache" 会以 pytorch 格式加载权重并存储一个 numpy 缓存
                以加快加载速度。
            "dummy" 会用随机值初始化权重，主要用于性能分析。
        dtype: 模型权重和激活值的数据类型。"auto" 选项对 FP32 和
            FP16 模型使用 FP16 精度，对 BF16 模型使用 BF16 精度。
        seed: 用于可复现性的随机种子。
        revision: 要使用的具体模型版本。可以是分支名、标签名或
            commit id。若未指定，则使用默认版本。
        tokenizer_revision: 要使用的具体 tokenizer 版本。可以是分支名、
            标签名或 commit id。若未指定，则使用默认版本。
        max_model_len: 序列的最大长度（包括 prompt 和输出）。
            若为 None，则从模型推导。
        quantization: 用于量化模型权重的量化方法。若为 None，
            则假设模型权重未被量化。
        enforce_eager: 是否强制使用 eager 模式执行。若为 True，
            将禁用 CUDA graph 并始终以 eager 模式执行模型。
            若为 False，将混合使用 CUDA graph 和 eager 执行。
        max_context_len_to_capture: CUDA graph 覆盖的最大上下文长度。
            当序列的上下文长度超过该值时，回退到 eager 模式。
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        dtype: str,
        seed: int,
        load_format: str = 'model',
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        trust_remote_code: Optional[bool] = True,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
    ) -> None:
        self.model = hf_config._name_or_path
        self.tokenizer = hf_config._name_or_path
        self.load_format = load_format
        self.seed = seed
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self.max_context_len_to_capture = max_context_len_to_capture

        # self.hf_config = get_config(model, trust_remote_code, revision)
        self.hf_config = hf_config
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.max_model_len = _get_and_verify_max_len(self.hf_config, max_model_len)
        # self._verify_load_format()
        # self._verify_tokenizer_mode()
        self._verify_quantization()
        self._verify_cuda_graph()

    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        if load_format not in ["auto", "pt", "safetensors", "npcache", "dummy", "model"]:
            raise ValueError(f"Unknown load format: {self.load_format}. Must be one of "
                             "'auto', 'pt', 'safetensors', 'npcache', 'dummy' or 'model'.")
        self.load_format = load_format

    # def _verify_tokenizer_mode(self) -> None:
    #     tokenizer_mode = self.tokenizer_mode.lower()
    #     if tokenizer_mode not in ["auto", "slow"]:
    #         raise ValueError(
    #             f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
    #             "either 'auto' or 'slow'.")
    #     self.tokenizer_mode = tokenizer_mode

    def _verify_quantization(self) -> None:
        supported_quantization = ["awq", "gptq", "squeezellm"]
        rocm_not_supported_quantization = ["awq", "gptq"]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # 从 HF 模型配置中解析量化方法（如果可用）。
        hf_quant_config = getattr(self.hf_config, "quantization_config", None)
        if hf_quant_config is not None:
            hf_quant_method = str(hf_quant_config["quant_method"]).lower()
            if self.quantization is None:
                self.quantization = hf_quant_method
            elif self.quantization != hf_quant_method:
                raise ValueError("Quantization method specified in the model config "
                                 f"({hf_quant_method}) does not match the quantization "
                                 f"method specified in the `quantization` argument "
                                 f"({self.quantization}).")

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(f"Unknown quantization method: {self.quantization}. Must "
                                 f"be one of {supported_quantization}.")
            if is_hip() and self.quantization in rocm_not_supported_quantization:
                raise ValueError(f"{self.quantization} quantization is currently not supported "
                                 f"in ROCm.")
            logger.warning(f"{self.quantization} quantization is not fully "
                           "optimized yet. The speed can be slower than "
                           "non-quantized models.")

    def _verify_cuda_graph(self) -> None:
        if self.max_context_len_to_capture is None:
            self.max_context_len_to_capture = self.max_model_len
        self.max_context_len_to_capture = min(self.max_context_len_to_capture, self.max_model_len)
        if (self.quantization in ["gptq", "squeezellm"] and not self.enforce_eager):
            # 相关 issue：https://github.com/vllm-project/vllm/issues/2147
            logger.warning(f"{self.quantization} does not support CUDA graph "
                           "yet. Disabling CUDA graph.")
            self.enforce_eager = True

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(f"Total number of attention heads ({total_num_attention_heads})"
                             " must be divisible by tensor parallel size "
                             f"({tensor_parallel_size}).")

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(f"Total number of hidden layers ({total_num_hidden_layers}) "
                             "must be divisible by pipeline parallel size "
                             f"({pipeline_parallel_size}).")

    def get_sliding_window(self) -> Optional[int]:
        return getattr(self.hf_config, "sliding_window", None)

    def get_vocab_size(self) -> int:
        return self.hf_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        # FIXME(woosuk): 这可能并不对所有模型都成立。
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_total_num_kv_heads(self) -> int:
        """返回 KV 头的总数。"""
        # 针对 GPTBigCode 与 Falcon：
        # NOTE: 对于 falcon，当 new_decoder_architecture 为 True 时，
        # multi_query 标志会被忽略，我们使用 n_head_kv 作为
        # KV 头的数量。
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (self.hf_config.model_type in falcon_model_types and
                                   getattr(self.hf_config, "new_decoder_architecture", False))
        if not new_decoder_arch_falcon and getattr(self.hf_config, "multi_query", False):
            # 多查询注意力（Multi-query attention），只有一个 KV 头。
            # 目前这种情况不支持张量并行。
            return 1

        attributes = [
            # 针对 Falcon：
            "n_head_kv",
            "num_kv_heads",
            # 针对 LLaMA-2：
            "num_key_value_heads",
            # 针对 ChatGLM：
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # 对于非分组查询注意力（non-grouped-query attention）模型，
        # KV 头的数量等于注意力头的数量。
        return self.hf_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """返回每个 GPU 上的 KV 头数量。"""
        total_num_kv_heads = self.get_total_num_kv_heads()
        # 如果使用张量并行，我们将 KV 头的数量除以张量并行大小。
        # 当 KV 头的数量小于张量并行大小时，会复制 KV 头，
        # 以确保每个 GPU 至少有一个 KV 头。
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size


class CacheConfig:
    """KV cache 的配置。

    Args:
        block_size: 缓存块的大小（以 token 数计）。
        gpu_memory_utilization: vLLM 执行所使用的 GPU 内存比例。
        swap_space: 每个 GPU 的 CPU 交换空间大小（GiB）。
        cache_dtype: KV cache 存储的数据类型。
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
        cache_dtype: str,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GB
        self.cache_dtype = cache_dtype
        self.sliding_window = sliding_window
        self._verify_args()
        self._verify_cache_dtype()

        # 将在性能分析（profiling）之后设置。
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError("GPU memory utilization must be less than 1.0. Got "
                             f"{self.gpu_memory_utilization}.")

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype == "fp8_e5m2":
            nvcc_cuda_version = get_nvcc_cuda_version()
            if nvcc_cuda_version < Version("11.8"):
                raise ValueError("FP8 is not supported when cuda version is lower than 11.8.")
            device_name = torch.cuda.get_device_name()
            if "AMD" in device_name:
                raise NotImplementedError("FP8_E5M2 KV Cache on AMD GPU has not been supported yet.")
            logger.info("Using fp8_e5m2 data type to store kv cache. It reduces "
                        "the GPU memory footprint and boosts the performance. "
                        "But it may cause slight accuracy drop. "
                        "Currently we only support fp8 without scaling factors and "
                        "make e5m2 as a default format.")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): 这里假设张量并行组中的 GPU 位于同一个节点上。
        # 然而，GPU 可能分布在多个节点上。
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (f"{cpu_memory_usage / _GB:.2f} GiB out of "
               f"the {total_cpu_memory / _GB:.2f} GiB total CPU memory is "
               "allocated for the swap space.")
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. " + msg)


class ParallelConfig:
    """分布式执行的配置。

    Args:
        pipeline_parallel_size: 流水线并行组的数量。
        tensor_parallel_size: 张量并行组的数量。
        worker_use_ray: 是否对模型 worker 使用 Ray。当
            pipeline_parallel_size 或 tensor_parallel_size 大于 1 时
            会被设置为 True。
        max_parallel_loading_workers: 顺序加载模型时每批的最大数量。
            用于避免在张量并行加载大模型时出现 RAM OOM。
        disable_custom_all_reduce: 禁用自定义 all-reduce kernel，
            回退到 NCCL。
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: bool,
        max_parallel_loading_workers: Optional[int] = None,
        disable_custom_all_reduce: bool = False,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_use_ray = worker_use_ray
        self.max_parallel_loading_workers = max_parallel_loading_workers
        self.disable_custom_all_reduce = disable_custom_all_reduce

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError("Pipeline parallelism is not supported yet.")
        if not self.disable_custom_all_reduce and self.world_size > 1:
            if is_hip():
                self.disable_custom_all_reduce = True
                logger.info("Disabled the custom all-reduce kernel because it is not "
                            "supported on AMD GPUs.")
            elif self.pipeline_parallel_size > 1:
                self.disable_custom_all_reduce = True
                logger.info("Disabled the custom all-reduce kernel because it is not "
                            "supported with pipeline parallelism.")

        # FIXME(woosuk): 修复稳定性问题后重新启用自定义
        # all-reduce kernel。
        if not self.disable_custom_all_reduce and self.world_size > 1:
            self.disable_custom_all_reduce = True
            logger.info("Custom all-reduce kernels are temporarily disabled due to "
                        "stability issues. We will re-enable them once the issues are "
                        "resolved.")


class SchedulerConfig:
    """调度器（Scheduler）配置。

    Args:
        max_num_batched_tokens: 单次迭代中可处理的最大 token 数。
        max_num_seqs: 单次迭代中可处理的最大序列数。
        max_model_len: 序列的最大长度（包括 prompt 和生成的文本）。
        max_paddings: 一个 batch 中可添加的最大 padding 数。
    """

    def __init__(
        self,
        max_num_batched_tokens: Optional[int],
        max_num_seqs: int,
        max_model_len: int,
        max_paddings: int,
    ) -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            # 如果 max_model_len 太短，为了获得更高吞吐量，
            # 使用 2048 作为默认值。
            self.max_num_batched_tokens = max(max_model_len, 2048)
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_paddings = max_paddings
        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                             f"smaller than max_model_len ({self.max_model_len}). "
                             "This effectively limits the maximum sequence length to "
                             "max_num_batched_tokens and makes vLLM reject longer "
                             "sequences. Please increase max_num_batched_tokens or "
                             "decrease max_model_len.")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                             "be greater than or equal to max_num_seqs "
                             f"({self.max_num_seqs}).")


class DeviceConfig:

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)


@dataclass
class LoRAConfig:
    max_lora_rank: int
    max_loras: int
    max_cpu_loras: Optional[int] = None
    lora_dtype: Optional[torch.dtype] = None
    lora_extra_vocab_size: int = 256
    # 这是一个常量。
    lora_vocab_padding_size: ClassVar[int] = 256

    def __post_init__(self):
        # 与 csrc/punica/bgmv/bgmv_config.h 保持同步
        possible_max_ranks = (8, 16, 32, 64)
        possible_lora_extra_vocab_size = (0, 256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(f"max_lora_rank ({self.max_lora_rank}) must be one of "
                             f"{possible_max_ranks}.")
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(f"lora_extra_vocab_size ({self.lora_extra_vocab_size}) "
                             f"must be one of {possible_lora_extra_vocab_size}.")
        if self.max_loras < 1:
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                             f"max_loras ({self.max_loras})")

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
        if model_config.quantization is not None:
            raise ValueError("LoRA is not supported with quantized models yet.")

    def verify_with_scheduler_config(self, scheduler_config: SchedulerConfig):
        if scheduler_config.max_num_batched_tokens > 65528:
            raise ValueError("Due to limitations of the custom LoRA CUDA kernel, "
                             "max_num_batched_tokens must be <= 65528 when "
                             "LoRA is enabled.")


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_ROCM_NOT_SUPPORTED_DTYPE = ["float", "float32"]


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) 是不正确的，
    # 因为 config.torch_dtype 可能为 None。
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                # 遵循常见做法，对 float32 模型使用 float16。
                torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    if is_hip() and torch_dtype == torch.float32:
        rocm_supported_dtypes = [
            k for k, v in _STR_DTYPE_TO_TORCH_DTYPE.items() if (k not in _ROCM_NOT_SUPPORTED_DTYPE)
        ]
        raise ValueError(f"dtype \'{dtype}\' is not supported in ROCm. "
                         f"Supported dtypes are {rocm_supported_dtypes}")

    # 校验 dtype。
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # 向上转换为 float32 是允许的。
            pass
        elif config_dtype == torch.float32:
            # 从 float32 向下转换为 float16 或 bfloat16 是允许的。
            pass
        else:
            # 在 float16 和 bfloat16 之间转换是允许的，会给出警告。
            logger.warning(f"Casting {config_dtype} to {torch_dtype}.")

    return torch_dtype


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
) -> int:
    """获取并校验模型的最大长度。"""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT 模型
        "max_position_embeddings",
        # GPT-2 模型
        "n_positions",
        # MPT 模型
        "max_seq_len",
        # ChatGLM2 模型
        "seq_length",
        # 其他模型
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    for key in possible_keys:
        max_len_key = getattr(hf_config, key, None)
        if max_len_key is not None:
            derived_max_model_len = min(derived_max_model_len, max_len_key)
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # 如果指定了 max_model_len，则使用它。
            return max_model_len

        default_max_len = 2048
        logger.warning("The model's config.json does not contain any of the following "
                       "keys to determine the original maximum length of the model: "
                       f"{possible_keys}. Assuming the model's maximum length is "
                       f"{default_max_len}.")
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        assert "factor" in rope_scaling
        scaling_factor = rope_scaling["factor"]
        if rope_scaling["type"] == "yarn":
            derived_max_model_len = rope_scaling["original_max_position_embeddings"]
        derived_max_model_len *= scaling_factor

    if max_model_len is None:
        max_model_len = derived_max_model_len
    elif max_model_len > derived_max_model_len:
        raise ValueError(f"User-specified max_model_len ({max_model_len}) is greater than "
                         f"the derived max_model_len ({max_len_key}={derived_max_model_len}"
                         " in model's config.json). This may lead to incorrect model "
                         "outputs or CUDA errors. Make sure the value is correct and "
                         "within the model context size.")
    return int(max_model_len)
