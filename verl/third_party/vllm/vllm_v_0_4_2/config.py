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

import enum
import json
from typing import List, Optional, Union
from dataclasses import dataclass, field, fields

from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils import is_hip
# 为 verl 添加
from vllm.config import ModelConfig, _get_and_verify_dtype, _get_and_verify_max_len

GPTQMarlinConfig = get_quantization_config("gptq_marlin")

logger = init_logger(__name__)

_GB = 1 << 30


class ModelConfig(ModelConfig):
    """模型的配置。

    参数:
        model: 要使用的 huggingface 模型的名称或路径。
        tokenizer: 要使用的 huggingface tokenizer 的名称或路径。
        tokenizer_mode: Tokenizer 模式。"auto" 会使用 fast tokenizer（如果
            可用），"slow" 则始终使用 slow tokenizer。
        trust_remote_code: 下载模型和 tokenizer 时信任远程代码（例如来自
            HuggingFace 的代码）。
        download_dir: 下载和加载权重的目录，默认为 huggingface 的
            默认缓存目录。
        load_format: 要加载的模型权重格式：
            "auto" 会尝试以 safetensors 格式加载权重，如果
                safetensors 格式不可用，则回退到 pytorch bin 格式。
            "pt" 会以 pytorch bin 格式加载权重。
            "safetensors" 会以 safetensors 格式加载权重。
            "npcache" 会以 pytorch 格式加载权重，并存储
                一个 numpy 缓存以加快加载速度。
            "dummy" 会用随机值初始化权重，主要用于
                profiling。
        dtype: 模型权重和激活值的数据类型。"auto" 选项
            会为 FP32 和 FP16 模型使用 FP16 精度，为 BF16 模型
            使用 BF16 精度。
        seed: 用于可复现性的随机种子。
        revision: 要使用的具体模型版本。可以是分支名、
            标签名或 commit id。如果未指定，将使用默认
            版本。
        code_revision: Hugging Face Hub 上模型代码要使用的具体版本。
            可以是分支名、标签名或
            commit id。如果未指定，将使用默认版本。
        tokenizer_revision: 要使用的具体 tokenizer 版本。可以是
            分支名、标签名或 commit id。如果未指定，将使用
            默认版本。
        max_model_len: 序列的最大长度（包括 prompt 和
            输出）。如果为 None，将从模型推导得出。
        quantization: 用于量化模型权重的量化方法。
            如果为 None，我们假设模型权重未被量化。
        quantization_param_path: 包含 scaling factor 的 JSON 文件路径。
            当 KV cache 类型为 ROCm（AMD GPU）上的 FP8_E4M3 时，
            用于将 KV cache scaling factor 加载到模型中。未来当
            模型 dtype 为 ROCm 上的 FP8_E4M3 时，这些也将用于
            加载激活值和权重的 scaling factor。
        enforce_eager: 是否强制使用 eager 执行。如果为 True，我们将
            禁用 CUDA graph，并始终以 eager 模式执行模型。
            如果为 False，我们将混合使用 CUDA graph 和 eager 执行。
        max_context_len_to_capture: CUDA graph 覆盖的最大上下文长度。
            当序列的上下文长度超过此值时，我们回退到
            eager 模式（已弃用。请改用 max_seq_len_to_capture）。
        max_seq_len_to_capture: CUDA graph 覆盖的最大序列长度。
            当序列的上下文长度超过此值时，我们回退到
            eager 模式
        skip_tokenizer_init: 如果为 true，则跳过 tokenizer 和
            detokenizer 的初始化。
        served_model_name: 用于指标标签 `model_name` 的模型名称，
            与通过 API 暴露的模型名称一致。如果提供了多个模型
            名称，将使用第一个名称。如果未指定，
            模型名称将与 `model` 相同。
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        dtype: str,
        seed: int,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        quantization_param_path: Optional[str] = None,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: Optional[int] = None,
        max_logprobs: int = 5,
        skip_tokenizer_init: bool = False,
        served_model_name: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self.model = hf_config._name_or_path
        self.tokenizer = hf_config._name_or_path
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.enforce_eager = enforce_eager
        self.max_context_len_to_capture = max_context_len_to_capture
        if self.max_context_len_to_capture is not None:
            raise ValueError("`max_context_len_to_capture` is deprecated. "
                             "Use `max_seq_len_to_capture` instead.")
        self.max_seq_len_to_capture = (max_seq_len_to_capture or max_context_len_to_capture)
        self.max_logprobs = max_logprobs
        self.skip_tokenizer_init = skip_tokenizer_init

        # self.hf_config = get_config(model, trust_remote_code, revision)
        self.hf_config = hf_config
        self.hf_text_config = get_hf_text_config(hf_config)
        # TODO: 针对多模态模型
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.max_model_len = _get_and_verify_max_len(self.hf_config, max_model_len)
        # self.served_model_name = get_served_model_name(model,
        #                                                served_model_name)
        # self._verify_load_format()
        # self._verify_tokenizer_mode()
        self._verify_quantization()
        self._verify_cuda_graph()


class LoadFormat(str, enum.Enum):
    AUTO = 'auto'
    MEGATRON = "megatron"
    HF = "hf"
    DTENSOR = 'dtensor'
    DUMMY_HF = 'dummy_hf'
    DUMMY_MEGATRON = 'dummy_megatron'
    DUMMY_DTENSOR = 'dummy_dtensor'


@dataclass
class LoadConfig:
    """
        download_dir: 下载和加载权重的目录，默认为 huggingface 的
            默认缓存目录。
        load_format: 要加载的模型权重格式：
            "auto" 会尝试以 safetensors 格式加载权重，如果
                safetensors 格式不可用，则回退到 pytorch bin 格式。
            "pt" 会以 pytorch bin 格式加载权重。
            "safetensors" 会以 safetensors 格式加载权重。
            "npcache" 会以 pytorch 格式加载权重，并存储
                一个 numpy 缓存以加快加载速度。
            "dummy" 会用随机值初始化权重，主要用于
                profiling。
            "tensorizer" 会使用 CoreWeave 的 tensorizer 库来
                快速加载权重。
    """

    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(default_factory=dict)

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(model_loader_extra_config)
        self._verify_load_format()

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return

        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)

        rocm_not_supported_load_format: List[str] = []
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f for f in LoadFormat.__members__ if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(f"load format '{load_format}' is not supported in ROCm. "
                             f"Supported load formats are "
                             f"{rocm_supported_load_format}")
