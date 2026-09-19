# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from typing import Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig
from megatron.core import ModelParallelConfig

from .parallel_attention import ParallelLlamaAttention, ParallelLlamaAttentionRmPad
from .parallel_mlp import ParallelLlamaMLP
from .parallel_rmsnorm import ParallelLlamaRMSNorm


class ParallelLlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = ParallelLlamaAttention(config=config, megatron_config=megatron_config)

        self.mlp = ParallelLlamaMLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelLlamaRMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelLlamaRMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        参数:
            hidden_states (`torch.FloatTensor`): 形状为 `(batch, seq_len, embed_dim)` 的层输入
            attention_mask (`torch.FloatTensor`, *optional*): 形状为
                `(batch, 1, tgt_len, src_len)` 的注意力掩码，其中 padding 元素用非常大的负值表示。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的 attention 张量。详见返回张量中的 `attentions`。
            use_cache (`bool`, *optional*):
                若设为 `True`，则返回 `past_key_values` 键值状态，可用于加速解码
                （参见 `past_key_values`）。
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): 缓存的过去 key 和 value 投影状态
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # 注意: sequence parallel 隐藏在 ColumnParallelLinear 内部
        # reduce scatter 隐藏在 RowParallelLinear 内部

        # 自注意力
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # TODO: 在此处添加 sequence parallel 算子 reduce_scatter

        hidden_states = residual + hidden_states

        # 全连接层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # TODO: 在此处添加 sequence parallel 算子 all_gather

        hidden_states = self.mlp(hidden_states)

        # TODO: 在此处添加 sequence parallel 算子 reduce_scatter

        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs


class ParallelLlamaDecoderLayerRmPad(nn.Module):

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.hidden_size = config.hidden_size
        self.self_attn = ParallelLlamaAttentionRmPad(config=config, megatron_config=megatron_config)

        self.mlp = ParallelLlamaMLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelLlamaRMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelLlamaRMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states  # (total_nnz // sp, 1, hidden_size)

        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力
        # (total_nnz // sp, 1, hidden_size) -> all-gather (total_nnz, 1, hidden_size)
        # -> col + row -> reduce-scatter -> (total_nnz // sp, 1, hidden_size)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       position_ids=position_ids,
                                       sequence_length=sequence_length,
                                       indices=indices,
                                       cu_seqlens=cu_seqlens,
                                       max_seqlen_in_batch=max_seqlen_in_batch)

        hidden_states = residual + hidden_states

        # 全连接层
        # 形状变化与 attn 相同
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs
