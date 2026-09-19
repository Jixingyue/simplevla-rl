# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
"""
用于从 huggingface 创建常见模型的工具函数
"""
import os
import warnings
from typing import Dict, Type

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from verl.models.registry import ModelRegistry


class LambdaLayer(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def squeeze(x):
    return torch.squeeze(x, dim=-1)


def update_model_config(module_config, override_config_kwargs):
    for key, val in override_config_kwargs.items():
        setattr(module_config, key, val)


def get_huggingface_actor_config(model_name: str, override_config_kwargs=None, trust_remote_code=False) -> Dict:
    if override_config_kwargs is None:
        override_config_kwargs = {}
    assert isinstance(override_config_kwargs, Dict), \
        f'override_config_kwargs must be a dict, got {type(override_config_kwargs)}'
    module_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    update_model_config(module_config, override_config_kwargs)

    return module_config


def create_huggingface_actor(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    """

    Args:
        model_name:
        actor_override_config_kwargs:

    Returns:

    """
    if override_config_kwargs is None:
        override_config_kwargs = {}
    if automodel_kwargs is None:
        automodel_kwargs = {}
    assert isinstance(override_config_kwargs, Dict), \
        f'override_config_kwargs must be a dict, got {type(override_config_kwargs)}'
    module_config = get_huggingface_actor_config(model_name,
                                                 override_config_kwargs,
                                                 trust_remote_code=automodel_kwargs.get('trust_remote_code', False))
    module: nn.Module = AutoModelForCausalLM.from_config(module_config, **automodel_kwargs)
    return module


def create_huggingface_critic(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    """

    Args:
        model_name:
        override_config_kwargs:

    Returns:

    """
    critic_module: nn.Module = create_huggingface_actor(model_name,
                                                        override_config_kwargs=override_config_kwargs,
                                                        automodel_kwargs=automodel_kwargs)
    if automodel_kwargs is None:
        automodel_kwargs = {}
    torch_dtype = automodel_kwargs.get('torch_dtype', torch.float32)
    critic_module.lm_head = nn.Sequential(nn.Linear(critic_module.config.hidden_size, 1, dtype=torch_dtype),
                                          LambdaLayer(fn=squeeze))
    return critic_module


def get_model_size(model: nn.Module, scale='auto'):
    n_params = sum(p.numel() for p in model.parameters())

    if scale == 'auto':
        if n_params > 1e9:
            scale = 'B'
        elif n_params > 1e6:
            scale = 'M'
        elif n_params > 1e3:
            scale = 'K'
        else:
            scale = ''

    if scale == 'B':
        n_params = n_params / 1e9
    elif scale == 'M':
        n_params = n_params / 1e6
    elif scale == 'K':
        n_params = n_params / 1e3
    elif scale == '':
        pass
    else:
        raise NotImplemented(f'Unknown scale {scale}')

    return n_params, scale


def print_model_size(model: nn.Module, name: str = None):
    n_params, scale = get_model_size(model, scale='auto')
    if name is None:
        name = model.__class__.__name__
    print(f'{name} contains {n_params:.2f}{scale} parameters')


def create_random_mask(input_ids: torch.Tensor,
                       max_ratio_of_valid_token: float,
                       max_ratio_of_left_padding: float,
                       min_ratio_of_valid_token: float = 0):
    """给定 input_ids 创建随机掩码。支持左填充和右填充。
    流程：
    - 采样有效 token 长度
    - 采样 left_padding 长度
    - 生成填充

    Args:
        input_ids:
            shape (batch_size, seq_len)

    Returns:

    """
    assert max_ratio_of_valid_token > 0 and max_ratio_of_valid_token <= 1.
    assert max_ratio_of_left_padding >= 0 and max_ratio_of_left_padding < 1.
    assert min_ratio_of_valid_token <= max_ratio_of_valid_token

    batch_size, sequence_length = input_ids.shape
    max_num_valid_tokens = int(sequence_length * max_ratio_of_valid_token)
    min_num_valid_tokens = max(1, int(sequence_length * min_ratio_of_valid_token))
    max_left_padding = int(sequence_length * max_ratio_of_left_padding)
    assert max_num_valid_tokens + max_left_padding <= sequence_length
    assert max_num_valid_tokens > 0 and max_ratio_of_valid_token <= sequence_length
    masks = torch.ones_like(input_ids, dtype=torch.int64)
    # TODO: 可以进一步优化这里的速度
    for i in range(batch_size):
        num_left_padding = np.random.randint(low=0, high=max_left_padding + 1, dtype=np.int64)
        num_valid = np.random.randint(low=min_num_valid_tokens, high=max_num_valid_tokens + 1, dtype=np.int64)

        for index in range(num_left_padding):
            masks[i, index] = 0

        for index in range(num_left_padding + num_valid, sequence_length):
            masks[i, index] = 0
    return masks


def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


def normalize_pp_vpp_params(params, num_hidden_layers, layer_name='layers'):
    """
    将 pp/vpp 参数归一化为完整的命名参数。
    这在从 pp rank 收集参数并传递给无 pp 的模型时很有用

    params: List[List[Dict[str, param]]]
        params 包含一个 pp 列表，每个 vpp chunk 中含有一个 vpp named_parameters 列表。
    output: Dict[str, param]

    """

    def normalize_model_name(name, pp_rank, vpp_rank, pp_size, vpp_size, num_layers):
        """
        将每个 pp stage 中每个 model_chunk 的模型名称转换为推理引擎中的名称
        """
        if vpp_size > 1:
            # print(f'try to bind vpp params to inference engine...')
            layers_per_pp = num_layers // pp_size
            layers_per_vpp = layers_per_pp // vpp_size
            pp_offset = layers_per_vpp * pp_rank
            vpp_offset = (layers_per_vpp * pp_size) * vpp_rank
            layer_offset = pp_offset + vpp_offset
        else:
            layers_per_pp = num_layers // pp_size
            layer_offset = layers_per_pp * pp_rank

        if layer_name in name:  # 属于中间层
            split_name = name.split('.')
            # 找到 split_name 后面的数字
            for i, name in enumerate(split_name):
                if name == layer_name:
                    break
            layer_num_idx = i + 1
            # 检查名称
            assert len(split_name) >= layer_num_idx + 1, f'split_name = {split_name}'
            assert split_name[layer_num_idx].isdigit(), f'split_name = {split_name}'
            # 将 layer_num_idx 加上 layer_offset
            split_name[layer_num_idx] = str(int(split_name[layer_num_idx]) + layer_offset)
            name = '.'.join(split_name)  # inference_tp_model 中的权重名称
        return name

    pp_size = len(params)
    normalized_name_to_param = {}
    for pp_rank in range(len(params)):
        vpp_size = len(params[pp_rank])
        for vpp_rank in range(vpp_size):
            for name, param in params[pp_rank][vpp_rank].items():
                normalized_name = normalize_model_name(name, pp_rank, vpp_rank, pp_size, vpp_size, num_hidden_layers)
                normalized_name_to_param[normalized_name] = param

    return normalized_name_to_param


def get_parallel_model_from_config(config, megatron_config, pre_process=None, post_process=None, value=False):
    from megatron.core import ModelParallelConfig
    assert isinstance(megatron_config, ModelParallelConfig)
    model_class = _get_parallel_model_architecture_from_config(config, value)

    model = model_class(config, megatron_config, pre_process=pre_process, post_process=post_process)
    return model


def _get_parallel_model_architecture_from_config(config: PretrainedConfig, value=False) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch, value)
        if model_cls is not None:
            return model_cls
    raise ValueError(f"Model architectures {architectures} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def load_megatron_model_weights(config,
                                model_config,
                                parallel_model,
                                params_dtype,
                                is_value_model=False,
                                local_cache_path='~/.cache/verl/rlhf'):
    assert hasattr(model_config, "architectures"), "architectures cannot be empty when load weight!"
    architectures = getattr(model_config, "architectures", [])
    local_cache_path = os.path.expanduser(local_cache_path)

    if config.model.path.startswith("hdfs:"):
        from verl.utils.fs import copy_local_path_from_hdfs
        print(f'start download from {config.model.path}')
        local_model_path = copy_local_path_from_hdfs(src=config.model.path, cache_dir=local_cache_path)
        print('finish download')
    else:
        print(f"load from local dir {config.model.path}")
        local_model_path = config.model.path

    model = AutoModelForCausalLM.from_pretrained(local_model_path)
    state_dict = model.state_dict()

    from verl.models.weight_loader_registry import get_weight_loader
    print(f'before weight loader: architectures = {architectures}...')
    for arch in architectures:
        print(f'call weight loader arch = {arch}, model config = {model.config}')
        weight_loader = get_weight_loader(arch)
        weight_loader(state_dict=state_dict,
                      wrapped_models=parallel_model,
                      config=model.config,
                      params_dtype=params_dtype,
                      is_value_model=is_value_model)


# 将 input_ids_rmpad、cu_seqlens 和 max_seqlen_in_batch 填充至可被 tp 整除
def pad_packed_inputs(unpad_tokens: torch.Tensor, cu_seqlens, max_seqlen_in_batch, size):
    """填充 token 使总长度为 size 的倍数。
    该函数在应用序列并行和上下文并行时很有用

    Args:
        unpad_tokens: (total_nnz, ...)。去除填充后的 token
        cu_seqlens: (total_nnz + 1,)
        max_seqlen_in_batch: int

    Returns:

    """
    F = nn.functional

    total_nnz = unpad_tokens.shape[0]

    if total_nnz % size == 0:
        pad_size = 0
    else:
        pad_size = size - total_nnz % size

    # 我们假设在 batch 中添加一个 seqlen 为 pad_size 的新数据
    if pad_size > 0:
        if unpad_tokens.ndim == 1:
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        elif unpad_tokens.ndim == 2:
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        else:
            raise NotImplementedError(f'Padding dim {unpad_tokens.ndim()} is not supported')

        cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_size + cu_seqlens[-1])
        max_seqlen_in_batch = max(max_seqlen_in_batch, pad_size)

    return unpad_tokens, cu_seqlens, max_seqlen_in_batch
