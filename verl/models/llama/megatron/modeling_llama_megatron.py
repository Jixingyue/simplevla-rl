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
""" PyTorch LLaMA 模型。"""

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from megatron.core import tensor_parallel
from megatron.core import ModelParallelConfig
from torch import nn
from torch.nn import init
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast

from verl.utils.megatron import sequence_parallel as sp_utils
from verl.utils.megatron import tensor_parallel as tp_utils
from .layers import ParallelLlamaDecoderLayer, ParallelLlamaRMSNorm, ParallelLlamaDecoderLayerRmPad
"""
TODO: 
1. 添加权重初始化。这里我们需要小心处理 TP 权重初始化。
2. 添加 sequence parallel
3. 从 meta LLama 预训练 checkpoint 加载权重
"""


# 复制自 transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """
    构建用于双向自注意力的因果掩码（causal mask）。
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# 复制自 transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    将 attention_mask 从 `[bsz, seq_len]` 扩展为 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class ParallelLlamaModel(nn.Module):
    """
    由 *config.num_hidden_layers* 层组成的 Transformer decoder。每一层都是一个 [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            assert embedding_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                                   embedding_dim=config.hidden_size,
                                                                   **embedding_kwargs)

        self.layers = nn.ModuleList(
            [ParallelLlamaDecoderLayer(config, megatron_config) for _ in range(config.num_hidden_layers)])
        self.norm = ParallelLlamaRMSNorm(config, megatron_config)

    # 复制自 transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        # 创建因果掩码（causal mask）
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype,
                                              tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                       combined_attention_mask)

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """

        Args:
            input_ids: 输入 id。形状 (batch_size, seq_length)
            attention_mask: 注意力掩码。形状 (batch_size, seq_length)
            position_ids: 位置 id。形状 (batch_size, seq_length)

        Returns:

        """
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        # 位置嵌入

        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds)

        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelLlamaForCausalLM(nn.Module):

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.model = ParallelLlamaModel(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)

        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=config.hidden_size,
                                                            output_size=config.vocab_size,
                                                            bias=False,
                                                            gather_output=False,
                                                            skip_bias_add=False,
                                                            **column_kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                用于计算掩码语言建模损失（masked language modeling loss）的标签。索引应在 `[0, ...,
                config.vocab_size]` 内或为 -100（见 `input_ids` 的 docstring）。索引被设为 `-100` 的 token 会被忽略
                （masked），损失只针对标签在 `[0, ..., config.vocab_size]` 内的 token 计算。

        Returns:
        ```"""

        # decoder 输出由 (dec_features, layer_state, dec_hidden, dec_attn) 组成
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)[0]

        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)

        logits = logits.float()
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


class ParallelLlamaModelRmPad(nn.Module):
    """
    由 *config.num_hidden_layers* 层组成的 Transformer decoder。每一层都是一个 [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        self.megatron_config = megatron_config
        if megatron_config is not None:
            assert embedding_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                                   embedding_dim=config.hidden_size,
                                                                   **embedding_kwargs)

        self.layers = nn.ModuleList(
            [ParallelLlamaDecoderLayerRmPad(config, megatron_config) for _ in range(config.num_hidden_layers)])
        self.norm = ParallelLlamaRMSNorm(config, megatron_config)

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: Optional[torch.LongTensor] = None,
                sequence_length: int = None,
                indices: torch.Tensor = None,
                cu_seqlens: int = None,
                max_seqlen_in_batch: int = None) -> Union[Tuple, BaseModelOutputWithPast]:
        """

        Args:
            input_ids: 输入 id。形状 (1, totol_nnz)
            position_ids: 位置 id。形状 (batch_size, seq_length)

        Returns:

        """
        inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

        # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
        inputs_embeds = inputs_embeds.transpose(0, 1)
        if self.megatron_config.sequence_parallel:
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(hidden_states,
                                          position_ids=position_ids,
                                          sequence_length=sequence_length,
                                          indices=indices,
                                          cu_seqlens=cu_seqlens,
                                          max_seqlen_in_batch=max_seqlen_in_batch)

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelLlamaForCausalLMRmPad(nn.Module):

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.model = ParallelLlamaModelRmPad(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size
        self._init_head()

    def _init_head(self):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=self.config.hidden_size,
                                                            output_size=self.config.vocab_size,
                                                            bias=False,
                                                            gather_output=False,
                                                            skip_bias_add=False,
                                                            **column_kwargs)

    def _forward_head(self, hidden_states):
        # 从 sequence parallel 区域进行 all_gather 的操作在 lm_head 内部完成
        logits = self.lm_head(hidden_states)[0]
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)  # (total_nnz_padded, 1, vocab_size)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                用于计算掩码语言建模损失（masked language modeling loss）的标签。索引应在 `[0, ...,
                config.vocab_size]` 内或为 -100（见 `input_ids` 的 docstring）。索引被设为 `-100` 的 token 会被忽略
                （masked），损失只针对标签在 `[0, ..., config.vocab_size]` 内的 token 计算。

        Returns:
        ```"""
        batch_size, sequence_length = input_ids.shape

        # 在这里去除 padding
        input_ids, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(input_ids.unsqueeze(dim=-1),
                                                                          attention_mask)  # (total_nnz, 1)

        # 将 input_ids 填充（pad）到所有 tp rank 的 tp 倍数
        # TODO: 为了更好的性能，sp padding 应该在每一层被移除。不确定性能差距有多大
        if self.megatron_config.sequence_parallel:
            input_ids = sp_utils.pad_to_sequence_parallel(input_ids)

        input_ids = input_ids.transpose(0, 1)  # (1, total_nnz+pad)

        outputs = self.model(input_ids=input_ids,
                             position_ids=position_ids,
                             sequence_length=sequence_length,
                             indices=indices,
                             cu_seqlens=cu_seqlens,
                             max_seqlen_in_batch=max_seqlen_in_batch)

        hidden_states = outputs

        logits = self._forward_head(hidden_states)

        # 从 sequence parallel 中去除 padding
        if self.megatron_config.sequence_parallel:
            totol_nnz = cu_seqlens[-1]
            logits = logits[:totol_nnz]  # (total_nnz_padded)

        logits = torch.squeeze(logits, dim=1)  # 移除人为添加的 batch 维度
        # 把移除的 padding 加回去
        logits = pad_input(logits, indices, batch_size,
                           seqlen=sequence_length)  # (batch_size, sequence_length, vocab_size)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class ParallelLlamaForValueRmPad(ParallelLlamaForCausalLMRmPad):

    def _init_head(self):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = nn.Linear(in_features=self.config.hidden_size, out_features=1, bias=False)
        # lm_head 实际上等同于 sequence parallel
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    def _forward_head(self, hidden_states):
        logits = self.lm_head(hidden_states)  # (total_nnz_padded // tp, 1, 1)
        logits = logits.float()
        if self.megatron_config.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output = super().forward(input_ids, attention_mask, position_ids)
        output.logits = torch.squeeze(output.logits, dim=-1)
        return output


"""
支持流水线并行（pipeline parallelism）
"""


class ParallelLlamaModelRmPadPP(nn.Module):
    """
    由 *config.num_hidden_layers* 层组成的 Transformer decoder。每一层都是一个 [`LlamaDecoderLayer`]
    该模型定义支持流水线并行（pipeline parallelism）。为了支持 pp 和 vpp：
    - 该模型只包含当前 pp stage 和 vpp chunk 中的层
    - 当在 Megatron 中调用 get_model 时，该 rank 会实例化当前 pp 中的所有 vpp chunk。
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig, pre_process, post_process):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.megatron_config = megatron_config
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            assert embedding_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        if pre_process:
            self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                                       embedding_dim=config.hidden_size,
                                                                       **embedding_kwargs)
        else:
            self.embed_tokens = None

        # pp_rank = megatron_config.pipeline_model_parallel_rank
        pp_size = megatron_config.pipeline_model_parallel_size
        self.num_layer_per_pp = config.num_hidden_layers // pp_size
        vpp_size = megatron_config.virtual_pipeline_model_parallel_size

        if vpp_size is not None:
            self.num_layer_vpp_chunk = self.num_layer_per_pp // vpp_size
            self.num_layer_this_model = self.num_layer_vpp_chunk
            # vpp_rank = megatron_config.virtual_pipeline_model_parallel_rank
            # self.offset = vpp_rank * (
            #         config.num_hidden_layers // megatron_config.virtual_pipeline_model_parallel_size) + \
            #             (megatron_config.pipeline_model_parallel_rank * self.num_layer_vpp_chunk)
        else:
            self.num_layer_this_model = self.num_layer_per_pp
            # self.offset = pp_rank * self.num_layer_per_pp

        layers = []
        for i in range(self.num_layer_this_model):
            layer = ParallelLlamaDecoderLayerRmPad(config, megatron_config)
            # setattr(layer, 'hidden_layer_index', self.offset + i)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if post_process:
            self.norm = ParallelLlamaRMSNorm(config, megatron_config)
        else:
            self.norm = None

    def set_input_tensor(self, input_tensor):
        """设置用于替代 forward() 输入的 input tensor。

        在进行流水线并行时，来自上一个 stage 的输入是通过通信获得的，
        而不是来自输入，因此模型的 forward_step_func 拿不到它。内部代码
        因此使用该函数来绕过 forward_step_func 提供的输入"""
        self.input_tensor = input_tensor

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: Optional[torch.LongTensor] = None,
                sequence_length: int = None,
                indices: torch.Tensor = None,
                cu_seqlens: int = None,
                max_seqlen_in_batch: int = None) -> Union[Tuple, BaseModelOutputWithPast]:
        """

        Args:
            input_ids: 输入 id。形状 (1, totol_nnz)
            position_ids: 位置 id。形状 (batch_size, seq_length)

        Returns:

        """
        if self.pre_process:
            # if torch.cuda.current_device() == 0:
            #     print(f'rank {torch.cuda.current_device()}: input_ids shape before embedding: {input_ids.shape}')
            inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

            # 在开源 megatron 中，vocab parallel embedding 不会做 sequence parallel 的 reduce-scatter
            # 所以需要在这里手动处理：
            # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
            inputs_embeds = inputs_embeds.transpose(0, 1)
            if self.megatron_config.sequence_parallel:
                inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

            # if torch.cuda.current_device() == 0:
            #     print(f'rank {torch.cuda.current_device()}: input_embeds shape after embedding: {inputs_embeds.shape}')
            hidden_states = inputs_embeds
        else:
            # self.hidden_states 应由 Megatron 传入
            hidden_states = self.input_tensor

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(hidden_states,
                                          position_ids=position_ids,
                                          sequence_length=sequence_length,
                                          indices=indices,
                                          cu_seqlens=cu_seqlens,
                                          max_seqlen_in_batch=max_seqlen_in_batch)

            hidden_states = layer_outputs

        if self.post_process:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelLlamaForCausalLMRmPadPP(nn.Module):

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig, pre_process, post_process):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.model = ParallelLlamaModelRmPadPP(config,
                                               megatron_config=megatron_config,
                                               pre_process=pre_process,
                                               post_process=post_process)
        self.share_embeddings_and_output_weights = None  # 变通处理（workaround），megatron 要求有该属性
        self.vocab_size = config.vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        if post_process:
            self._init_head()

    def set_input_tensor(self, input_tensor):
        """设置用于替代 forward() 输入的 input tensor。

        在进行流水线并行时，来自上一个 stage 的输入是通过通信获得的，
        而不是来自输入，因此模型的 forward_step_func 拿不到它。内部代码
        因此使用该函数来绕过 forward_step_func 提供的输入"""
        assert len(input_tensor) == 1
        self.model.set_input_tensor(input_tensor[0])

    def _init_head(self):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=self.config.hidden_size,
                                                            output_size=self.config.vocab_size,
                                                            bias=False,
                                                            gather_output=False,
                                                            skip_bias_add=False,
                                                            **column_kwargs)

    def _forward_head(self, hidden_states):
        # 从 sequence parallel 区域进行 all_gather 的操作在 lm_head 内部完成
        # print(f'logits shape before forward_head: {hidden_states.shape}, vocab_size = {self.config.vocab_size}') # [4, 32, 4096]
        logits = self.lm_head(hidden_states)[0]
        # print(f'logits shape after forward_head: {logits.shape}') # [8, 32, 8]
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        return logits

    def forward(
        self,
        # 原始输入
        *,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                用于计算掩码语言建模损失（masked language modeling loss）的标签。索引应在 `[0, ...,
                config.vocab_size]` 内或为 -100（见 `input_ids` 的 docstring）。索引被设为 `-100` 的 token 会被忽略
                （masked），损失只针对标签在 `[0, ..., config.vocab_size]` 内的 token 计算。

        Returns:
        ```"""

        # 注意 input_ids、attention_mask 和 position_ids 应传递给每一个 pp 层。
        # 在第一个 pp 阶段会使用 input_ids，在其他 pp 层中 self.model 内部会使用 hidden_states
        batch_size, sequence_length = input_ids.shape
        # 在这里去除 padding
        input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(input_ids.unsqueeze(dim=-1),
                                                                                attention_mask)  # (total_nnz, 1)
        # print(f'input_ids.shape = {input_ids.shape}, input_ids_rmpad.shape = {input_ids_rmpad.shape}, indices.shape = {indices.shape}, cu_seqlens[-1] = {cu_seqlens[-1]}')
        # 将 input_ids 填充（pad）到所有 tp rank 的 tp 倍数
        # TODO: 为了更好的性能，sp padding 应该在每一层被移除。不确定性能差距有多大
        if self.megatron_config.sequence_parallel:
            input_ids_rmpad = sp_utils.pad_to_sequence_parallel(input_ids_rmpad)

        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz+pad)

        outputs = self.model(input_ids=input_ids_rmpad,
                             position_ids=position_ids,
                             sequence_length=sequence_length,
                             indices=indices,
                             cu_seqlens=cu_seqlens,
                             max_seqlen_in_batch=max_seqlen_in_batch)

        if self.post_process:
            hidden_states = outputs
            # print(f'hidden_states.shape = {hidden_states.shape}') # torch.Size([4, 32, 4096])
            logits = self._forward_head(hidden_states)
            # print(f'logits.shape = {logits.shape}')
            logits = torch.squeeze(logits, dim=1)  # 移除人为添加的 batch 维度 # torch.Size([8, 32, 16])

            # 从 sequence parallel 中去除 padding
            if self.megatron_config.sequence_parallel:
                totol_nnz = cu_seqlens[-1]
                logits = logits[:totol_nnz]  # (total_nnz_padded)
            # 把移除的 padding 加回去。如果输入已经是 rmpad，我们让调用方来做 pad_input
            # print(f'logits.shape = {logits.shape}, indices.shape = {indices.shape}, batch_size = {batch_size}, seq_len = {sequence_length}')
            logits = pad_input(logits, indices, batch_size,
                               seqlen=sequence_length)  # (batch_size, sequence_length, vocab_size)

            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
        else:
            return outputs


class ParallelLlamaForValueRmPadPP(ParallelLlamaForCausalLMRmPadPP):

    def _init_head(self):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = nn.Linear(in_features=self.config.hidden_size, out_features=1, bias=False)
        # lm_head 实际上等同于 sequence parallel
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    def _forward_head(self, hidden_states):
        logits = self.lm_head(hidden_states)  # (total_nnz_padded // tp, 1, 1)
        logits = logits.float()
        if self.megatron_config.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits

    def forward(
        self,
        *,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if self.post_process:
            output.logits = torch.squeeze(output.logits, dim=-1)
            return output
        else:
            return output
