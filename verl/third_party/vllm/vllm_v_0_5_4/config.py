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
# 改编自 https://github.com/vllm-project/vllm/blob/main/vllm/config.py

import enum
import json
from typing import List, Optional, Union
from dataclasses import dataclass, field, fields

import torch
from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils import is_hip, print_warning_once
# 为 verl 添加
from vllm.config import ModelConfig, _get_and_verify_dtype, _get_and_verify_max_len, get_served_model_name

GPTQMarlinConfig = get_quantization_config("gptq_marlin")

logger = init_logger(__name__)

_GB = 1 << 30


class ModelConfig(ModelConfig):
    """模型配置。

    Args:
        model: 要使用的 huggingface 模型的名称或路径。
        tokenizer: 要使用的 huggingface tokenizer 的名称或路径。
        tokenizer_mode: tokenizer 模式。"auto" 会在可用时使用快速 tokenizer，
            "slow" 则始终使用慢速 tokenizer。
        trust_remote_code: 下载模型和 tokenizer 时是否信任远程代码
            （例如来自 HuggingFace 的代码）。
        download_dir: 下载和加载权重的目录，默认为 huggingface 的
            默认缓存目录。
        load_format: 要加载的模型权重格式：
            "auto" 会尝试以 safetensors 格式加载权重，若 safetensors 格式
                不可用则回退到 pytorch bin 格式。
            "pt" 会以 pytorch bin 格式加载权重。
            "safetensors" 会以 safetensors 格式加载权重。
            "npcache" 会以 pytorch 格式加载权重，并存储 numpy 缓存以
                加快加载速度。
            "dummy" 会用随机值初始化权重，主要用于性能分析。
        dtype: 模型权重和激活值的数据类型。"auto" 选项会对 FP32 和
            FP16 模型使用 FP16 精度，对 BF16 模型使用 BF16 精度。
        seed: 用于保证可复现性的随机种子。
        revision: 要使用的具体模型版本。可以是分支名、标签名或
            commit id。若未指定，将使用默认版本。
        code_revision: Hugging Face Hub 上模型代码使用的具体版本。
            可以是分支名、标签名或 commit id。若未指定，将使用
            默认版本。
        tokenizer_revision: 要使用的具体 tokenizer 版本。可以是分支名、
            标签名或 commit id。若未指定，将使用默认版本。
        max_model_len: 序列的最大长度（包括 prompt 和输出）。若为
            None，将从模型推导。
        quantization: 量化模型权重时使用的量化方法。若为 None，则
            假定模型权重未量化。
        quantization_param_path: 包含缩放因子的 JSON 文件路径。用于在
            KV cache 类型为 FP8_E4M3 时（ROCm/AMD GPU 上）将 KV cache
            缩放因子加载到模型中。未来当模型 dtype 为 FP8_E4M3 时
            （ROCm 上），这些因子也将用于加载激活值和权重的缩放因子。
        enforce_eager: 是否强制 eager 执行。若为 True，将禁用
            CUDA graph，始终以 eager 模式执行模型；若为 False，将
            混合使用 CUDA graph 和 eager 执行。
        max_context_len_to_capture: CUDA graph 覆盖的最大上下文长度。
            当序列的上下文长度超过该值时，回退到 eager 模式
            （已弃用。请改用 max_seq_len_to_capture）。
        max_seq_len_to_capture: CUDA graph 覆盖的最大序列长度。当
            序列的上下文长度超过该值时，回退到 eager 模式。
        skip_tokenizer_init: 若为 true，跳过 tokenizer 和
            detokenizer 的初始化。
        served_model_name: 用于指标标签 `model_name` 的模型名称，
            与通过 API 暴露的模型名称一致。若提供多个模型名称，
            将使用第一个名称。若未指定，模型名称将与 `model`
            相同。
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        tokenizer_mode: str,
        trust_remote_code: bool,
        dtype: Union[str, torch.dtype],
        seed: int,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        rope_scaling: Optional[dict] = None,
        rope_theta: Optional[float] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        quantization_param_path: Optional[str] = None,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: Optional[int] = None,
        max_logprobs: int = 20,
        disable_sliding_window: bool = False,
        skip_tokenizer_init: bool = False,
        served_model_name: Optional[Union[str, List[str]]] = None,
        multimodal_config: Optional["MultiModalConfig"] = None,
    ) -> None:
        self.model = hf_config._name_or_path
        self.tokenizer = hf_config._name_or_path
        # NOTE(sgm): 与开源版本相同
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        # 默认情况下 tokenizer 版本与模型版本保持一致。
        if tokenizer_revision is None:
            self.tokenizer_revision = revision
        else:
            self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.enforce_eager = enforce_eager
        if max_context_len_to_capture is not None:
            raise ValueError("`max_context_len_to_capture` is deprecated. "
                             "Use `max_seq_len_to_capture` instead.")
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.skip_tokenizer_init = skip_tokenizer_init

        # self.hf_config = get_config(model, trust_remote_code, revision)
        self.hf_config = hf_config
        self.hf_text_config = get_hf_text_config(hf_config)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)
        # self.served_model_name = get_served_model_name(model,
        #                                                served_model_name)
        # self._verify_load_format()
        # self._verify_tokenizer_mode()
        if (not self.disable_sliding_window and self.hf_text_config.model_type == "gemma2" and
                self.hf_text_config.sliding_window is not None):
            print_warning_once("Gemma 2 uses sliding window attention for every odd layer, "
                               "which is currently not supported by vLLM. Disabling sliding "
                               "window and capping the max length to the sliding window size "
                               f"({self.hf_text_config.sliding_window}).")
            self.disable_sliding_window = True

        self.max_model_len = _get_and_verify_max_len(hf_config=self.hf_text_config,
                                                     max_model_len=max_model_len,
                                                     disable_sliding_window=self.disable_sliding_window,
                                                     sliding_window_len=self.get_hf_config_sliding_window())
        self.served_model_name = get_served_model_name(
            self.model,  # str
            served_model_name)
        self.multimodal_config = multimodal_config

        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()
        self._verify_embedding_mode()
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


# TODO: 检查这是否有必要
@dataclass
class LoadConfig:
    """
        download_dir: 下载和加载权重的目录，默认为 huggingface 的
            默认缓存目录。
        load_format: 要加载的模型权重格式：
            "auto" 会尝试以 safetensors 格式加载权重，若 safetensors 格式
                不可用则回退到 pytorch bin 格式。
            "pt" 会以 pytorch bin 格式加载权重。
            "safetensors" 会以 safetensors 格式加载权重。
            "npcache" 会以 pytorch 格式加载权重，并存储 numpy 缓存以
                加快加载速度。
            "dummy" 会用随机值初始化权重，主要用于性能分析。
            "tensorizer" 会使用 CoreWeave 的 tensorizer 库进行
                快速权重加载。
            "bitsandbytes" 会加载 nf4 类型的权重。
        ignore_patterns: 加载模型时要忽略的模式列表。默认为
            "original/**/*" 以避免重复加载 llama 的 checkpoint。
            
    """

    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(default_factory=dict)
    ignore_patterns: Optional[Union[List[str], str]] = None

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(model_loader_extra_config)
        self._verify_load_format()

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info("Ignoring the following patterns when downloading weights: %s", self.ignore_patterns)
        else:
            self.ignore_patterns = ["original/**/*"]

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
