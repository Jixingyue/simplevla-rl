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
包含一些小型 torch 工具函数
"""

from typing import Dict, Union, List, Optional

import os
import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


def gather_from_labels(data, label):
    """从 data 中收集（gather）label 对应的值。label 中的值应在 [0, vocab_size) 范围内

    Args:
        data: (..., vocab_size)
        label (torch.IntTensor) : (...,)

    Returns:

    """

    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    return output


def logprobs_from_logits(logits, labels):
    """
    参见: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_naive(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels):
    output = -cross_entropy_loss(logits, labels)[0]
    return output


def logprobs_from_logits_naive(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logpy = gather_from_labels(logp, labels)
    return logpy


def logprobs_of_labels_v2(logits: torch.FloatTensor, labels):
    """
    logprobs_from_logits 的省内存实现
    """
    assert logits.dtype == torch.float32, 'Using bf16 logits with logprobs_of_labels_v2 may lead to divergence'
    logprobs_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1))
    logprobs_labels = logprobs_labels - torch.logsumexp(logits, dim=-1, keepdim=True)
    return logprobs_labels.squeeze(-1)


def clip_by_value(x, tensor_min, tensor_max):
    """
    torch.clamp 的张量扩展版本
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits: torch.Tensor):
    """从 logits 计算熵。"""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def masked_sum(values, mask, axis=None):
    """计算带掩码值张量的和。"""
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """计算带掩码值张量的均值。"""
    if (mask == False).all():
        return (values * mask).sum(axis=axis) 
    else: 
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_var(values, mask, unbiased=True):
    """计算带掩码值张量的方差。"""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # 注意，如果 mask_sum == 1，则会出现除以零的问题
        # 为避免该问题，只需使用更大的 minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """对带掩码值的张量做白化。"""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def get_eos_mask(response_id: torch.Tensor, eos_token: int = 2, dtype=torch.int64):
    '''
    例如句子结束 token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    '''
    eos_mask = response_id.eq(eos_token).long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def compute_grad_norm(model: nn.Module):
    total_grad_square = 0
    total_params = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_square += torch.sum(torch.square(param.grad.detach())).item()
    return total_grad_square


def broadcast_dict_tensor(tensors: Union[Dict[str, torch.Tensor], TensorDict], src, group):
    """
    TODO: 优化此处。理论上我们只需要一次 broadcast
    """

    for key in tensors.sorted_keys:
        torch.distributed.broadcast(tensors[key], src=src, group=group, async_op=False)


def allgather_dict_tensors(tensors: Union[Dict[str, torch.Tensor], TensorDict], size, group, dim=0):
    """
    TODO: 优化此处。
    - 我们可以使用异步操作
    - 我们可以只用一次 allgather
    Args:
        tensors:
        size:
        group:

    Returns:

    """
    if isinstance(tensors, TensorDict):
        is_tensor_dict = True
        tensors_as_dict = tensors.to_dict()
    else:
        tensors_as_dict = tensors
        is_tensor_dict = False

    output = {}
    sorted_keys = sorted(tensors_as_dict.keys())
    for key in sorted_keys:
        val = tensors_as_dict[key]
        output[key] = [torch.empty_like(val) for _ in range(size)]
        torch.distributed.all_gather(output[key], val, group=group, async_op=False)
        output[key] = torch.cat(output[key], dim=dim)

    if is_tensor_dict:
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)

    return output


def split_dict_tensor_into_batches(tensors: TensorDict, batch_size) -> List[TensorDict]:
    assert tensors.batch_size[0] % batch_size == 0, \
        f'input data batch size: {tensors.batch_size[0]}, split batch size: {batch_size}'
    return tensors.split(batch_size)


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    在最后一维将 2D 张量（例如 responses、logprobs）填充到 max_seq_length。
    输入形状: [bs, seq_length]
    输出形状: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) 表示在右侧填充到 max_seq_length 且不做左侧填充
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, 'constant', pad_token_id)


from transformers import PreTrainedTokenizer


def tokenize_and_postprocess_data(prompt: str,
                                  tokenizer: PreTrainedTokenizer,
                                  max_length: int,
                                  pad_token_id: int,
                                  left_pad=True,
                                  truncation='error'):
    """
    输入数据是 tokenizer 的输出。
    """
    assert truncation in ['left', 'right', 'error']

    input_data = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

    input_ids = input_data['input_ids']
    attention_mask = input_data['attention_mask']

    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(input_ids,
                                           max_seq_len=max_length,
                                           pad_token_id=pad_token_id,
                                           left_pad=left_pad)
        attention_mask = pad_sequence_to_length(attention_mask,
                                                max_seq_len=max_length,
                                                pad_token_id=0,
                                                left_pad=left_pad)
    elif sequence_length > max_length:
        if truncation == 'left':
            # 实际上，左侧截断可能并不合理
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == 'right':
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == 'error':
            raise NotImplementedError(f'{sequence_length} is larger than {max_length}')
        else:
            raise NotImplementedError(f'Unknown truncation method {truncation}')

    return input_ids, attention_mask


def postprocess_rob_data(input_ids,
                        attention_mask,
                        max_length: int,
                        pad_token_id: int,
                        left_pad=True,
                        truncation='error'):
    """
    输入数据是 tokenizer 的输出。
    """
    assert truncation in ['left', 'right', 'error']

    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(input_ids,
                                           max_seq_len=max_length,
                                           pad_token_id=pad_token_id,
                                           left_pad=left_pad)
        attention_mask = pad_sequence_to_length(attention_mask,
                                                max_seq_len=max_length,
                                                pad_token_id=0,
                                                left_pad=left_pad)
    elif sequence_length > max_length:
        if truncation == 'left':
            # 实际上，左侧截断可能并不合理
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == 'right':
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == 'error':
            raise NotImplementedError(f'{sequence_length} is larger than {max_length}')
        else:
            raise NotImplementedError(f'Unknown truncation method {truncation}')

    return input_ids, attention_mask


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """ 移除 pad token。

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[List[int]]): 包含每条 query 的去 pad token id。
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        no_padding_batch.append((ids[len(ids) - mask.sum():]).cpu().numpy().tolist())
    return no_padding_batch


def log_probs_from_logits_response(input_ids, logits, response_length):
    """从完整的 logits 中计算 response 部分的 log_probs。注意 logits = model(input_ids)
    
    Args:
        input_ids: [batch_size, seqlen]
        logits: [batch_size, seqlen, vocab_size]
    
    Returns:
        response_log_prob: 
    """
    response_logits = logits[:, -response_length - 1:-1]
    response = input_ids[:, -response_length:]
    response_log_prob = logprobs_from_logits(logits=response_logits, labels=response)
    return response_log_prob


def log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    """基于 rmpad 的 logits 和 pad 的输入计算 log_probs。注意
    logits_rmpad = model(input_ids_rmpad)。对于每个句子，logits 与 input_ids 之间存在一个偏移。
    之所以需要这个函数，是为了在 rmpad 模式下计算 logprobs_from_logits，因为当 vocab_size 很大时
    这样做更省内存
    
    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size]
        response_length: int
    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(input_ids.unsqueeze(-1),
                                                                            attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen)
    output = full_output.squeeze(-1)[:, -response_length - 1:-1]  # [batch_size, response_length]
    return output


def log_probs_from_logits_all_rmpad(input_ids_rmpad, logits_rmpad, indices, batch_size, seqlen, response_length):
    """基于 rmpad 的 input_ids 和 logits 计算 log_probs。注意
    logits_rmpad = model(input_ids_rmpad)。对于每个句子，logits 与 input_ids 之间存在一个偏移。
    之所以需要这个函数，是为了在 rmpad 模式下计算 logprobs_from_logits，因为当 vocab_size 很大时
    这样做更省内存
    
    Args:
        input_ids_rmpad: [1, total_nnz]
        logits_rmpad: [total_nnz, vocab_size]
        indices: [total_nnz]
        batch_size: int
        seqlen: int
        response_length: int
    """
    from flash_attn.bert_padding import pad_input
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # 转置回 [total_nnz, 1]
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen)
    output = full_output.squeeze(-1)[:, -response_length - 1:-1]  # [batch_size, response_length]
    return output


from transformers.generation.logits_process import (TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)


def post_process_logits(input_ids, logits, temperature, top_k, top_p):
    if temperature != 1.:
        logits = logits.div_(temperature)  # 原地操作以避免 OOM
    # TODO: 将它们加回来
    # if top_k is not None and top_k > 0:
    #     logits = TopKLogitsWarper(top_k=top_k)(input_ids, logits)
    # if top_p is not None and top_p < 1.0 and top_p > 0.0:
    #     logits = TopPLogitsWarper(top_p=top_p)(input_ids, logits)
    return logits


"""
与优化器相关的工具
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    创建一个学习率调度器：先经过一段 warmup 期间（期间学习率在 0 与优化器中设置的初始 lr 之间线性增长），
    之后学习率按照余弦函数的值从优化器中设置的初始 lr 衰减到 0。
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            要调度学习率的优化器。
        num_warmup_steps (:obj:`int`):
            warmup 阶段的步数。
        num_training_steps (:obj:`int`):
            训练总步数。
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            相对于最大值的最小 lr 比例。
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            余弦调度中的波数（默认只是按半个余弦从最大值衰减到 0）。
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            恢复训练时上一个 epoch 的索引。
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR`，带有相应的调度。
    """
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(0.0, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):

    def lr_lambda(current_step):
        return min(1, float(current_step) / float(max(1, num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    # 创建因果掩码
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


# 复制自 transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """
    构造用于双向自注意力的因果掩码。
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


def get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )