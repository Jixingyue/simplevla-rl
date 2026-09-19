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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

from transformers import PretrainedConfig

# 为 verl 添加
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.loader import BaseModelLoader

logger = init_logger(__name__)


class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    MEGATRON = "megatron"
    HF = "hf"
    DTENSOR = "dtensor"
    DUMMY_HF = "dummy_hf"
    DUMMY_MEGATRON = "dummy_megatron"
    DUMMY_DTENSOR = "dummy_dtensor"


class ModelConfig(ModelConfig):

    def __init__(self, hf_config: PretrainedConfig, *args, **kwargs) -> None:
        super().__init__(model=hf_config._name_or_path, tokenizer=hf_config._name_or_path, *args, **kwargs)
        self.hf_config = hf_config


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
        "tensorizer" 会使用 CoreWeave 的 tensorizer 库来快速加载
            权重。
        "bitsandbytes" 会加载 nf4 类型的权重。
    ignore_patterns: 加载模型时要忽略的模式列表。
        默认为 "original/**/*"，以避免重复加载 llama 的
        checkpoint。

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
