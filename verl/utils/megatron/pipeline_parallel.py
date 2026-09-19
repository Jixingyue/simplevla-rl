# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
from megatron.core import parallel_state as mpu

from .sequence_parallel import pad_to_sequence_parallel


def compute_transformers_input_shapes(batches, meta_info):
    from flash_attn.bert_padding import unpad_input  # Megatron 必须使用 flash 2
    # 预先计算每个流水线并行阶段各个 micro-batch 的输入形状
    input_shapes = []
    for model_inputs in batches:
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        input_ids_rmpad = unpad_input(input_ids.unsqueeze(dim=-1), attention_mask)[0]  # (total_nnz, 1)
        if meta_info['sequence_parallel']:
            input_ids_rmpad = pad_to_sequence_parallel(input_ids_rmpad)
            # 为 model_inputs 计算形状
            input_shapes.append(
                torch.Size([
                    input_ids_rmpad.shape[0] // mpu.get_tensor_model_parallel_world_size(), 1, meta_info['hidden_size']
                ]))
        else:
            # 为 model_inputs 计算形状
            input_shapes.append(torch.Size([input_ids_rmpad.shape[0], 1, meta_info['hidden_size']]))
    return input_shapes


def make_batch_generator(batches, vpp_size):
    if vpp_size > 1:
        # 有 vpp
        batch_generator = [batches] * vpp_size  # vpp 块的数量
        batch_generator = [iter(b) for b in batch_generator]
    else:
        # 无 vpp
        batch_generator = iter(batches)
    return batch_generator
