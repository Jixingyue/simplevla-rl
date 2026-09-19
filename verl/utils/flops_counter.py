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

import torch
from transformers import PretrainedConfig, Qwen2Config, LlamaConfig

VALID_CONFIG_TYPE = (Qwen2Config, LlamaConfig)


def get_device_flops(unit="T"):

    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device_name = torch.cuda.get_device_name()
    flops = float("inf")  # 未知 GPU 类型时 FLOPS 取无穷大
    if "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    flops_unit = unit_convert(flops, unit)
    return flops_unit


class FlopsCounter:
    """
    用于在训练循环中统计 MFU。

    示例：
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        if not isinstance(config, VALID_CONFIG_TYPE):
            print(f"Only support config type of {VALID_CONFIG_TYPE}, but got {type(config)}. "
                  f"MFU will always be zero.")

        self.estimate_func = {"qwen2": self._estimate_qwen2_flops, 'llama': self._estimate_qwen2_flops}
        self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        return 0

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        assert isinstance(self.config, (Qwen2Config, LlamaConfig))
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # 非 attention 部分每层的参数
        # Qwen2/LLama 使用 SwiGelu，mlp 中包含 gate、up 和 down 线性层
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # 非 attention 部分所有层的参数
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # 非 attention 部分所有层、所有 token 的前向与反向 FLOPS
        dense_N_flops = 6 * dense_N * tokens_sum

        # attention 部分所有层、所有 token 的前向与反向 FLOPS
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # 所有层、所有 token 的前向与反向 FLOPS
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens, delta_time):
        """
        基于当前 batch 中的有效 token 数量及所耗时间估算 FLOPS。

        Args:
            batch_seqlens (List[int]): 一个列表，每个元素表示当前 batch 中的有效 token 数量。
            delta_time (float): 处理该 batch 所耗的时间，单位为秒。

        Returns:
            estimated_flops (float): 基于输入 token 和时间估算出的 FLOPS。
            promised_flops (float): 当前设备的理论 FLOPS。
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
