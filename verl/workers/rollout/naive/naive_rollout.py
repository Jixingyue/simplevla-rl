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
在单 GPU rollout 中，序列通过直接从模型采样来生成。
输出将包含
1. output_ids
2. attention_masks（左填充）
3. eos_masks
4. log_probs
"""
from typing import Iterable, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import logprobs_from_logits
from ..base import BaseRollout

__all__ = ['NativeRollout']


class NaiveRollout(BaseRollout):

    def __init__(self, module: nn.Module, config):
        """一个简单的 rollout。它要求模块与 huggingface API 兼容，即：
        模块应定义 __call__ 来接收 input_ids、attention_mask 和 position_ids，
        并输出一个包含 logits 字段的结构。

        Args:
            module: 这里的 module 遵循 huggingface API
            config: DictConfig
        """
        super().__init__()
        self.config = config
        self.module = module

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """生成序列"""
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']  # 左填充的 attention_mask
        position_ids = prompts.batch['position_ids']

        # 用于构建 attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()

        prev_attention_mask = torch.ones(size=(batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)

        logits_lst = []
        for _ in range(self.config.response_length):
            # 如果序列上下文增长过长，则需要在 block_size 处进行截断
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            idx_cond = idx
            # 前向模型，得到序列中对应索引的 logits
            # 这里我们使用 huggingface API
            output = self.module(input_ids=idx_cond, attention_mask=attention_mask, position_ids=position_ids)
            logits = output.logits
            # 取最后一步的 logits 并按给定温度进行缩放
            logits = logits[:, -1, :] / self.config.temperature  # (bs, vocab_size)
            # 可选：将 logits 裁剪为仅保留 top k 选项
            if self.config.top_k is not None:
                v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用 softmax 将 logits 转换为（归一化的）概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样
            if self.config.do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            attention_mask = torch.cat((attention_mask, prev_attention_mask), dim=-1)

            prev_attention_mask = torch.logical_and(idx_next != eos_token_id, prev_attention_mask.bool())
            prev_attention_mask.to(attention_mask.dtype)

            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)

            # 将采样得到的索引追加到当前序列并继续
            idx = torch.cat((idx, idx_next), dim=1)
            logits_lst.append(logits)

        logits = torch.stack(logits_lst, dim=1)  # (bs, response_length, vocab_size)
        prompts = idx[:, :prompt_length]  # (bs, prompt_length)
        response = idx[:, prompt_length:]  # (bs, response_length)
        log_probs = logprobs_from_logits(logits=logits, labels=response)
        batch = TensorDict(
            {
                'input_ids': prompts,
                'responses': response,
                'sequences': idx,
                'old_log_probs': log_probs,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            },
            batch_size=batch_size)

        self.module.train()

        return DataProto(batch=batch)
