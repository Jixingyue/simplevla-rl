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
# 改编自 https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
"""选择和加载模型的工具函数。"""
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from vllm.config import CacheConfig, DeviceConfig, LoadConfig, LoRAConfig, ModelConfig, ParallelConfig, SchedulerConfig
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.model_executor.model_loader import BaseModelLoader
from vllm.model_executor.model_loader.loader import _initialize_model
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

from .config import LoadConfig, LoadFormat, ModelConfig
from .dtensor_weight_loaders import load_dtensor_weights, update_dtensor_weight_loader
from .hf_weight_loader import update_hf_weight_loader
from .megatron_weight_loaders import load_megatron_weights, update_megatron_weight_loader


def get_model(
    actor_model: Union[PreTrainedModel, Dict],
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
    parallel_config: ParallelConfig,
    scheduler_config: SchedulerConfig,
    lora_config: Optional[LoRAConfig],
    cache_config: CacheConfig = None,
) -> nn.Module:
    loader = get_model_loader(load_config)
    if load_config.load_format.startswith("dummy"):
        return loader.load_model(
            model_config=model_config,
            device_config=device_config,
            lora_config=lora_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            cache_config=cache_config,
        )
    else:
        return loader.load_model(
            actor_model=actor_model,
            model_config=model_config,
            device_config=device_config,
            lora_config=lora_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            cache_config=cache_config,
        )


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """根据加载格式获取模型加载器。"""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.AUTO:
        update_megatron_weight_loader()
        return MegatronLoader(load_config)

    # NOTE(sgm): 在运行时替换 weight_loader 函数
    if load_config.load_format == LoadFormat.MEGATRON:
        update_megatron_weight_loader()
        return MegatronLoader(load_config)

    if load_config.load_format == LoadFormat.HF:
        update_hf_weight_loader()
        return HFLoader(load_config)

    if load_config.load_format == LoadFormat.DTENSOR:
        update_dtensor_weight_loader()
        return DTensorLoader(load_config)

    if load_config.load_format == LoadFormat.DUMMY_HF:
        update_hf_weight_loader()
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.DUMMY_MEGATRON:
        update_megatron_weight_loader()
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.DUMMY_DTENSOR:
        update_dtensor_weight_loader()
        return DummyModelLoader(load_config)

    raise ValueError("load format not supported in verl: {}, only support {} and {}".format(
        load_config.load_format, LoadFormat.MEGATRON, LoadFormat.HF))


class DummyModelLoader(BaseModelLoader):
    """将模型权重设置为随机值的模型加载器。"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)
            # NOTE(woosuk): 为了准确评估性能，我们为权重分配了
            # 随机值。
            # initialize_dummy_weights(model)
        return model.eval()


class MegatronLoader(BaseModelLoader):
    """可以从分片的 megatron 模型加载模型权重的模型加载器。"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # 没有需要下载的内容

    def _get_weights_iterator(actor_model: Union[PreTrainedModel, Dict]):
        # NOTE(shengguangming) 从 actor 模型加载权重
        pass
        # if isinstance(actor_model, nn.Module):
        #     load_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)), vllm_model=model)
        # else:
        #     load_weights(actor_weights=actor_model, vllm_model=model)
        # return actor_model

    def load_model(
        self,
        actor_model: Union[PreTrainedModel, Dict],
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)

            # TODO(sgm): 这是一种权宜做法，我们需要为 vllm 中的每个模型注册 load_weight() 函数
            if isinstance(actor_model, nn.Module):
                load_megatron_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)),
                                      vllm_model=model)
            else:
                load_megatron_weights(actor_weights=actor_model, vllm_model=model)

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                # FIXME: 在 Mixtral 更新为使用 quant_method 后
                # 移除此段。
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()
        # NOTE(sgm) 有些权重已指向 gpu，但仍然需要这一步。
        model = model.cuda()  # NOTE (zhangchi.usc1992) vllm 分析内存占用时需要这一步
        return model.eval()


class HFLoader(BaseModelLoader):
    """可以从模型的完整参数加载模型权重的模型加载器。"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def _get_weights_iterator(self, actor_model: Union[PreTrainedModel, Dict]):
        if isinstance(actor_model, Dict):
            return actor_model.items()
        elif isinstance(actor_model, nn.Module):
            return dict(actor_model.named_parameters()).items()
        else:
            raise ValueError(f"actor model should be Dict or nn.Module, but get {type(actor_model)}")

    def load_model(
        self,
        actor_model: Union[PreTrainedModel, Dict],
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            # with torch.device(device_config.device):
            # NOTE(sgm): 在 cpu 上初始化模型
            model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)
            model.load_weights(self._get_weights_iterator(actor_model))
            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                # FIXME: 在 Mixtral 更新为使用 quant_method 后
                # 移除此段。
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()
        # NOTE(sgm) 有些权重已指向 gpu，但仍然需要这一步。
        model = model.cuda()  # NOTE (zhangchi.usc1992) vllm 分析内存占用时需要这一步
        return model.eval()


class DTensorLoader(BaseModelLoader):
    """可以从分片的 megatron 模型加载模型权重的模型加载器。"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def _get_weights_iterator(actor_model: Union[PreTrainedModel, Dict]):
        # NOTE(shengguangming) 从 actor 模型加载权重
        pass
        # if isinstance(actor_model, nn.Module):
        #     load_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)), vllm_model=model)
        # else:
        #     load_weights(actor_weights=actor_model, vllm_model=model)
        # return actor_model

    def load_model(
        self,
        actor_model: Union[PreTrainedModel, Dict],
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)

            # TODO(sgm): 这是一种权宜做法，我们需要为 vllm 中的每个模型注册 load_weight() 函数
            if isinstance(actor_model, nn.Module):
                load_dtensor_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)),
                                     vllm_model=model)
            else:
                load_dtensor_weights(actor_weights=actor_model, vllm_model=model)

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                # FIXME: 在 Mixtral 更新为使用 quant_method 后
                # 移除此段。
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()
        # NOTE(sgm) 有些权重已指向 gpu，但仍然需要这一步。
        model = model.cuda()  # NOTE (zhangchi.usc1992) vllm 分析内存占用时需要这一步
        return model.eval()


# FIXME(sgm): 对 vllm v0.4.2 中的 _get_logits 函数做一个权宜修改
# 由于它们使用 ray，_get_logits 的结果只需要返回到 driver 节点，
# 因此 gather 就足够了。然而我们使用 SPMD 而不是中央调度器，
# 所以需要 all_gather（与 v0.2.6 对齐）
def _get_logits(self, hidden_states: torch.Tensor, embedding: torch.Tensor,
                embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
    # 获取下一个 token 的 logits。
    logits = torch.matmul(hidden_states, embedding.t())
    if embedding_bias is not None:
        logits += embedding_bias
    logits = tensor_model_parallel_all_gather(logits)
    # 移除词表中的 padding（如果有）。
    if logits is not None:
        logits = logits[:, :self.org_vocab_size]
    return logits


from vllm.model_executor.layers.logits_processor import LogitsProcessor


def logitsprocessor_init(
    self,
    vocab_size: int,
    org_vocab_size: Optional[int] = None,
    scale: float = 1.0,
    logits_as_input: bool = False,
    soft_cap: Optional[float] = None,
) -> None:
    """
    Args:
        scale: 应用于 logits 的缩放因子。
    """
    super(LogitsProcessor, self).__init__()
    self.scale = scale
    self.vocab_size = vocab_size
    # 输入是否为 logits（默认为 hidden states）。
    self.logits_as_input = logits_as_input
    # 原始词表大小（不含 LoRA）。
    self.org_vocab_size = org_vocab_size or vocab_size
    # 对 logits 做 soft cap。用于 Gemma 2。
    self.soft_cap = soft_cap
    # 使用 gather 还是 all-gather 来收集 logits。
    self.use_gather = False


LogitsProcessor.__init__ = logitsprocessor_init  # 使用 all_gather
