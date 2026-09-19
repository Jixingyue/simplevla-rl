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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader
"""用于选择和加载模型的工具函数。"""
import contextlib
from typing import Dict, Type, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from megatron.core.tensor_parallel.utils import VocabUtility

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.weight_utils import (get_quant_config, initialize_dummy_weights)

from .config import ModelConfig
from vllm.config import DeviceConfig, LoRAConfig
from .weight_loaders import *
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sequence import SamplerOutput
from typing import Optional
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.sampler import _prune_hidden_states, _apply_logits_processors, _apply_penalties, _apply_top_k_top_p, _apply_min_p, _apply_penalties, _sample, _get_logprobs, _build_sampler_output


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """将默认的 torch dtype 设置为给定的 dtype。"""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(f"Model architectures {architectures} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


from vllm.model_executor.layers.linear import *
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.layers.activation import ScaledActivation

__LAYER_WEIGHT_LOADER_REGISTRY__ = {
    ColumnParallelLinear: parallel_weight_loader,
    MergedColumnParallelLinear: parallel_weight_loader,
    QKVParallelLinear: parallel_weight_loader,
    RowParallelLinear: parallel_weight_loader,
    VocabParallelEmbedding: parallel_weight_loader,
    ParallelLMHead: parallel_weight_loader
    # "ScaledActivation.weight_loader": ScaledActivation, # TODO(shengguangming): vllm 的最新提交修复了该函数的 awq 问题并添加了 load_weights
    # "default_weight_loader": default_weight_loader
}

# NOTE(gmsheng): 在运行时更换 weight_loader 函数
for layer_class, weight_loader in __LAYER_WEIGHT_LOADER_REGISTRY__.items():
    layer_class.weight_loader = weight_loader

__MODEL_WEIGHT_LOADER_REGISTRY__ = {
    'GPT2LMHeadModel': gpt2_weight_loader,
    'LlamaForCausalLM': llama_weight_loader,
    'LLaMAForCausalLM': llama_weight_loader,
    'MistralForCausalLM': mistral_weight_loader,
}

# FIXME(shengguangming): vLLM 的词表会填充到 64，可能导致越界
# 因此我们需要重写 vocab 的 init 函数
DEFAULT_VOCAB_PADDING_SIZE = 64


def vocab_init(self,
               num_embeddings: int,
               embedding_dim: int,
               params_dtype: Optional[torch.dtype] = None,
               org_num_embeddings: Optional[int] = None,
               padding_size: int = DEFAULT_VOCAB_PADDING_SIZE):
    super(VocabParallelEmbedding, self).__init__()

    # 保留输入维度。
    # TODO (填充到能被 4 整除)
    self.num_embeddings = num_embeddings
    self.org_vocab_size = org_num_embeddings or num_embeddings

    # self.num_embeddings_padded = pad_vocab_size(num_embeddings,
    #                                             padding_size)
    self.embedding_dim = embedding_dim
    if params_dtype is None:
        params_dtype = torch.get_default_dtype()
    self.tp_size = get_tensor_model_parallel_world_size()
    # 沿词表维度切分权重矩阵。

    self.vocab_start_index, self.vocab_end_index = (VocabUtility.vocab_range_from_global_vocab_size(
        self.num_embeddings, get_tensor_model_parallel_rank(), self.tp_size))
    self.num_embeddings_per_partition = (self.vocab_end_index - self.vocab_start_index)
    self.weight = Parameter(
        torch.empty(
            self.num_embeddings_per_partition,
            self.embedding_dim,
            # device=torch.cuda.current_device(),
            dtype=params_dtype))
    set_weight_attrs(self.weight, {"parallel_dim": 0, "weight_loader": self.weight_loader})


VocabParallelEmbedding.__init__ = vocab_init


def _get_model_weight_loader(arch: str):
    if arch in __MODEL_WEIGHT_LOADER_REGISTRY__:
        return __MODEL_WEIGHT_LOADER_REGISTRY__[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(actor_model: Union[PreTrainedModel, Dict],
              model_config: ModelConfig,
              device_config: DeviceConfig,
              lora_config: Optional[LoRAConfig] = None) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    # 获取量化配置。
    linear_method = None
    quant_config = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config.quantization, model_config.model, model_config.hf_config,
                                        model_config.download_dir)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(f"The quantization method {model_config.quantization} is not "
                             "supported for the current GPU. "
                             f"Minimum capability: {quant_config.get_min_capability()}. "
                             f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(f"{model_config.dtype} is not supported for quantization "
                             f"method {model_config.quantization}. Supported dtypes: "
                             f"{supported_dtypes}")
        linear_method = quant_config.get_linear_method()

    with _set_default_torch_dtype(model_config.dtype):
        # 创建模型实例。
        # 权重将被初始化为空张量。
        # with torch.device(device_config.device):
        # NOTE(sgm): 在 cpu 上初始化模型
        model = model_class(model_config.hf_config, linear_method)

        if model_config.load_format == "dummy":
            model = model.cuda()
            # NOTE(woosuk): 为了准确评估性能，我们
            # 为权重赋予随机值。
            initialize_dummy_weights(model)
        elif model_config.load_format == 'model' or model_config.load_format == 'auto':
            # NOTE(shengguangming) 从 actor 模型加载权重
            if isinstance(actor_model, nn.Module):
                load_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)), vllm_model=model)
            else:
                load_weights(actor_weights=actor_model, vllm_model=model)

        # NOTE(sgm) 一些权重指向 gpu，但仍然需要这一步。
        model = model.cuda()  # NOTE (zhangchi.usc1992) 我们需要这一步以便 vllm 统计内存使用
    return model.eval()


# actor 模型是 .state_dict()
def load_weights(actor_weights: Dict, vllm_model: nn.Module):
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    weight_loader(actor_weights, vllm_model)
    # NOTE(sgm) 为了降低峰值内存占用，我们在初始化后将 vllm 模型卸载到 cpu
    # 并且在第一次迭代同步模型权重之后需要这一步。
    vllm_model = vllm_model.cuda()


# FIXME(sgm): hack vllm v0.3.1 的 Sampler 函数
# 因为他们使用 ray，采样结果只需要返回给 driver 节点，
# 因此 gather 就足够了。然而我们使用 SPMD 而不是中央调度器，
# 需要使用 all_gather（与 v0.2.6 对齐）
def _get_logits(self, hidden_states: torch.Tensor, embedding: torch.Tensor,
                embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
    # 获取下一个 token 的 logits。
    logits = torch.matmul(hidden_states, embedding.t())
    if embedding_bias is not None:
        logits += embedding_bias
    logits = tensor_model_parallel_all_gather(logits)
    # 移除词表中的填充（如果有）。
    if logits is not None:
        logits = logits[:, :self.org_vocab_size]
    return logits


def forward(
    self,
    embedding: torch.Tensor,
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    embedding_bias: Optional[torch.Tensor] = None,
) -> Optional[SamplerOutput]:
    # 获取用于采样的 hidden states。
    hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)

    # 获取下一个 token 的 logits。
    logits = self._get_logits(hidden_states, embedding, embedding_bias)
    # 为 sampler_output 保存原始 logprobs
    origin_logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

    # 仅在 driver worker 中执行采样。
    # Note: `_get_logits` 仍然分布在各 TP worker 上，因为
    # `embedding` 权重分布在各 TP worker 上。
    # TODO(zhuohan): 将 get_logits 部分改为一个单独的阶段。
    if not sampling_metadata.perform_sampling:
        return None

    assert logits is not None
    _, vocab_size = logits.shape

    # 应用 logits 处理器（如果有）。
    logits = _apply_logits_processors(logits, sampling_metadata)

    # 使用锁页内存准备采样张量以避免阻塞。
    (sampling_tensors, do_penalties, do_top_p_top_k,
     do_min_p) = SamplingTensors.from_sampling_metadata(sampling_metadata, vocab_size, logits.device, logits.dtype)

    # 应用 presence 和 frequency 惩罚。
    if do_penalties:
        logits = _apply_penalties(logits, sampling_tensors.prompt_tokens, sampling_tensors.output_tokens,
                                  sampling_tensors.presence_penalties, sampling_tensors.frequency_penalties,
                                  sampling_tensors.repetition_penalties)

    # 应用温度缩放。
    # 使用原地除法以避免创建新张量。
    logits.div_(sampling_tensors.temperatures.unsqueeze_(dim=1))

    if do_top_p_top_k:
        logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps, sampling_tensors.top_ks)

    if do_min_p:
        logits = _apply_min_p(logits, sampling_tensors.min_ps)

    # 我们使用 float32 存储概率和对数概率。
    # 计算概率。
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    # 计算对数概率。
    # 使用 log_softmax 确保数值稳定性。
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

    # 采样下一个 token。
    sample_results = _sample(probs, logprobs, sampling_metadata)

    # 获取 logprobs 查询结果。
    # prompt_logprobs, sample_logprobs = _get_logprobs(
    #     logprobs, sampling_metadata, sample_results)
    prompt_logprobs, sample_logprobs = _get_logprobs(origin_logprobs, sampling_metadata, sample_results)

    return _build_sampler_output(sample_results, sampling_metadata, prompt_logprobs, sample_logprobs)


from vllm.model_executor.layers.sampler import Sampler

Sampler._get_logits = _get_logits
Sampler.forward = forward
