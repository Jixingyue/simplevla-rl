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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models

from typing import Dict
import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import *
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.layers.activation import ScaledActivation
from vllm.model_executor.models import ModelRegistry


# NOTE(shengguangming): 替换类中原有的 weight loader 函数
def parallel_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """并行线性层的 weight loader。"""
    assert param.size() == loaded_weight.size(
    ), 'the parameter size is not align with the loaded weight size, param size: {}, loaded_weight size: {}'.format(
        param.size(), loaded_weight.size())
    assert param.data.dtype == loaded_weight.data.dtype, "if we want to shared weights, the data type should also be the same"

    param.data = loaded_weight.data


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """默认的 weight loader。"""
    assert param.size() == loaded_weight.size()
    assert param.data.dtype == loaded_weight.data.dtype, "if we want to shared weights, the data type should also be the same"

    param.data = loaded_weight.data


def gpt2_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "lm_head.weight" in name:
            # GPT-2 将 embedding 层与最终的
            # 线性层的权重绑定在一起。
            continue
        if ".attn.bias" in name or ".attn.masked_bias" in name:
            # 跳过 attention mask。
            # 注意："c_attn.bias" 不应被跳过。
            continue
        if not name.startswith("transformer."):
            name = "transformer." + name
        param = params_dict[name]
        # HF 的 GPT-2 实现使用 Conv1D 而不是 Linear。
        # 因此，我们需要对权重进行转置。
        # 注意(zhuohan)：下面的逻辑可能会破坏量化模型。
        for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
            if conv1d_weight_name not in name:
                continue
            if not name.endswith(".weight"):
                continue
            # TODO: 检查 megatron
            loaded_weight = loaded_weight.t()
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)


def llama_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # NOTE(shengguangming): megatron llama 可能带有此前缀
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def llama_megatron_core_te_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_mapping = [
        # (megatron core gpt 模型名, vllm 模型名)
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", 'self_attn.o_proj'),
        ('pre_mlp_layernorm', 'post_attention_layernorm'),
        ('mlp.linear_fc1.layer_norm_weight', 'post_attention_layernorm.weight'),
        ('mlp.linear_fc1.layer_norm_bias', 'post_attention_layernorm.bias'),
        ('mlp.linear_fc1', 'mlp.gate_up_proj'),
        ('mlp.linear_fc2', 'mlp.down_proj'),
        ('decoder.final_layernorm', 'model.norm'),
        ('output_layer', 'lm_head'),
    ]
    # NOTE(shengguangming): megatron llama 可能带有此前缀
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        name = _replace_name(name, params_mapping)
        if name.endswith('.bias') and name not in params_dict:
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def llama_megatron_core_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_mapping = [
        # (megatron core gpt 模型名, vllm 模型名)
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", 'self_attn.o_proj'),
        (
            'input_layernorm',
            'input_layernorm',
        ),
        ('pre_mlp_layernorm', 'post_attention_layernorm'),
        ('mlp.linear_fc1', 'mlp.gate_up_proj'),
        ('mlp.linear_fc2', 'mlp.down_proj'),
        ('decoder.final_layernorm', 'model.norm'),
        ('output_layer', 'lm_head'),
    ]
    # NOTE(shengguangming): megatron llama 可能带有此前缀
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        name = _replace_name(name, params_mapping)
        if name.endswith('.bias') and name not in params_dict:
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def _replace_name(megatron_name, name_mapping):
    for m_name, v_name in name_mapping:
        if m_name not in megatron_name:
            continue
        if 'layers' in megatron_name:  # 处理 decoder 层
            megatron_name = megatron_name.replace('decoder', 'model')
            megatron_name_list = megatron_name.split('.')
            if 'layer_norm_weight' in megatron_name_list or 'layer_norm_bias' in megatron_name_list:
                param_name_list = megatron_name_list[:3]
                param_name_list.append(v_name)
                param_name = '.'.join(param_name_list)
            else:
                param_name_list = megatron_name_list[:3]
                weight_or_bias = megatron_name_list[-1]
                param_name_list.append(v_name)
                param_name_list.append(weight_or_bias)
                param_name = '.'.join(param_name_list)
            return param_name
        else:
            param_name = megatron_name.replace(m_name, v_name)
            return param_name


def llama_megatron_core_te_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_mapping = [
        # (megatron core gpt 模型名, vllm 模型名)
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", 'self_attn.o_proj'),
        ('pre_mlp_layernorm', 'post_attention_layernorm'),
        ('mlp.linear_fc1.layer_norm_weight', 'post_attention_layernorm.weight'),
        ('mlp.linear_fc1.layer_norm_bias', 'post_attention_layernorm.bias'),
        ('mlp.linear_fc1', 'mlp.gate_up_proj'),
        ('mlp.linear_fc2', 'mlp.down_proj'),
        ('decoder.final_layernorm', 'model.norm'),
        ('output_layer', 'lm_head'),
    ]
    # NOTE(shengguangming): megatron llama 可能带有此前缀
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        name = _replace_name(name, params_mapping)
        if name.endswith('.bias') and name not in params_dict:
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def llama_megatron_core_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_mapping = [
        # (megatron core gpt 模型名, vllm 模型名)
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", 'self_attn.o_proj'),
        (
            'input_layernorm',
            'input_layernorm',
        ),
        ('pre_mlp_layernorm', 'post_attention_layernorm'),
        ('mlp.linear_fc1', 'mlp.gate_up_proj'),
        ('mlp.linear_fc2', 'mlp.down_proj'),
        ('decoder.final_layernorm', 'model.norm'),
        ('output_layer', 'lm_head'),
    ]
    # NOTE(shengguangming): megatron llama 可能带有此前缀
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        name = _replace_name(name, params_mapping)
        if name.endswith('.bias') and name not in params_dict:
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def _replace_name(megatron_name, name_mapping):
    for m_name, v_name in name_mapping:
        if m_name not in megatron_name:
            continue
        if 'layers' in megatron_name:  # 处理 decoder 层
            megatron_name = megatron_name.replace('decoder', 'model')
            megatron_name_list = megatron_name.split('.')
            if 'layer_norm_weight' in megatron_name_list or 'layer_norm_bias' in megatron_name_list:
                param_name_list = megatron_name_list[:3]
                param_name_list.append(v_name)
                param_name = '.'.join(param_name_list)
            else:
                param_name_list = megatron_name_list[:3]
                weight_or_bias = megatron_name_list[-1]
                param_name_list.append(v_name)
                param_name_list.append(weight_or_bias)
                param_name = '.'.join(param_name_list)
            return param_name
        else:
            param_name = megatron_name.replace(m_name, v_name)
            return param_name


def mistral_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # TODO: 需要实现一种通用的方式来处理前缀
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


__LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__ = {
    ColumnParallelLinear: parallel_weight_loader,
    MergedColumnParallelLinear: parallel_weight_loader,
    QKVParallelLinear: parallel_weight_loader,
    RowParallelLinear: parallel_weight_loader,
    VocabParallelEmbedding: parallel_weight_loader,
    ParallelLMHead: parallel_weight_loader
    # "ScaledActivation.weight_loader": ScaledActivation, # TODO(shengguangming): vllm 的最新提交修复了该函数的 awq 问题并添加了 load_weights
    # "default_weight_loader": default_weight_loader
}

# for layer_class, weight_loader in __LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__.items():
#     # setattr(layer_class, 'megatron_weight_loader', weight_loader)
#     layer_class.weight_loader = weight_loader

__MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__ = {
    'GPT2LMHeadModel': gpt2_weight_loader,
    'LlamaForCausalLM': llama_megatron_core_te_weight_loader,  # 开源 megatron 使用 te 后端
    'LLaMAForCausalLM': llama_megatron_core_te_weight_loader,
    'MistralForCausalLM': mistral_megatron_weight_loader,
}


# actor 模型是 .state_dict()
# 加载 megatron 权重
def load_megatron_weights(actor_weights: Dict, vllm_model: nn.Module):
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    weight_loader(actor_weights, vllm_model)
    # NOTE(sgm) 为了降低峰值内存占用，我们在初始化后将 vllm 模型卸载到 cpu，
    # 而在第一次迭代同步模型权重之后需要用到这里的功能。
    vllm_model = vllm_model.cuda()


def _get_model_weight_loader(arch: str):
    if arch in __MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__:
        return __MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def update_megatron_weight_loader():
    for layer_class, weight_loader in __LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__.items():
        layer_class.weight_loader = weight_loader
    VocabParallelEmbedding.__init__ = vocab_init


# FIXME(shengguangming): vLLM 的词表会填充到 64，可能导致越界
# 因此我们需要重写 vocab 的初始化函数
DEFAULT_VOCAB_PADDING_SIZE = 64


def vocab_init(self,
               num_embeddings: int,
               embedding_dim: int,
               params_dtype: Optional[torch.dtype] = None,
               org_num_embeddings: Optional[int] = None,
               padding_size: int = DEFAULT_VOCAB_PADDING_SIZE):
    super(VocabParallelEmbedding, self).__init__()

    # 保持输入维度。
    # TODO（填充到能被 4 整除）
    self.num_embeddings = num_embeddings
    self.org_vocab_size = org_num_embeddings or num_embeddings

    # self.num_embeddings_padded = pad_vocab_size(num_embeddings,
    #                                             padding_size)
    self.embedding_dim = embedding_dim
    if params_dtype is None:
        params_dtype = torch.get_default_dtype()
    self.tp_size = get_tensor_model_parallel_world_size()
    # 沿词表维度切分权重矩阵。

    # TODO: 移除对 megatron 的依赖
    from megatron.core.tensor_parallel.utils import VocabUtility
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
