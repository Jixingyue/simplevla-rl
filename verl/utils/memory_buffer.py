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
该文件包含操作 torch 内存缓冲区的工具函数
"""

from typing import Dict, List

import torch
from torch import nn


class MemoryBuffer:
    """
    内存缓冲区是一个连续的 torch 张量，可以组合多个共享底层内存的张量。
    它必须具有唯一的 dtype 才能支持这种行为。
    """

    def __init__(self, numel: int, numel_padded: int, dtype: torch.dtype):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        self.data = torch.zeros(self.numel_padded, dtype=self.dtype, device='cuda', requires_grad=False)

    def zero(self):
        """将缓冲区重置为零。"""
        self.data.zero_()

    def get(self, shape, start_index):
        """返回一个以输入 `shape` 为形状的张量视图，指向从 `start_index` 开始的一维数据。"""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor


def calc_padded_numel(shape: torch.Size, dtype: torch.dtype):
    """为了 CUDA 内存对齐，确保按 128 位对齐"""
    align_numel = 128 // torch.finfo(dtype).bits
    numel = shape.numel()
    return (numel + align_numel - 1) // align_numel * align_numel


def get_weight_buffer_meta_from_module(module: nn.Module) -> Dict[str, Dict]:
    """
    返回一个字典，将名称映射到形状和 dtype。
    """
    weight_buffer_meta = {}
    for name, param in sorted(module.named_parameters()):
        weight_buffer_meta[name] = {'shape': param.shape, 'dtype': param.dtype}
    return weight_buffer_meta


def build_memory_buffer(weight_buffer_meta: Dict[str, Dict]) -> Dict[torch.dtype, MemoryBuffer]:
    """根据 weight_buffer_meta 构建内存缓冲区

    Args:
        weight_buffer_meta: 包含从名称到张量形状和 dtype 字典的映射

    Returns: 每个 dtype 一个大的内存缓冲区，可容纳所有张量

    """
    memory_buffers = {}
    total_numel_map = {}  # 从 dtype 到总元素数的映射
    for name, meta_info in sorted(weight_buffer_meta.items()):
        shape = meta_info['shape']
        dtype = meta_info['dtype']

        assert isinstance(shape, torch.Size)
        assert isinstance(dtype, torch.dtype)

        if dtype not in total_numel_map:
            total_numel_map[dtype] = 0

        total_numel_map[dtype] += calc_padded_numel(shape, dtype)

    for dtype, total_numel in total_numel_map.items():
        memory_buffers[dtype] = MemoryBuffer(total_numel, total_numel, dtype)

    return memory_buffers


def build_memory_reference_from_module(module: torch.nn.Module,
                                       memory_buffers: Dict[torch.dtype, MemoryBuffer],
                                       maintain_weight=True):
    start_index = {}
    for dtype in memory_buffers.keys():
        start_index[dtype] = 0
    for name, param in sorted(module.named_parameters()):
        memory_buffer = memory_buffers[param.dtype]
        buffer = memory_buffer.get(shape=param.shape, start_index=start_index[param.dtype])
        # 需要递增 start_index
        start_index[param.dtype] += calc_padded_numel(param.shape, dtype)
        if maintain_weight:
            buffer.copy_(param.data)
        param.data = buffer


def build_memory_reference(weight_buffer_meta: Dict[str, Dict], memory_buffers: Dict[torch.dtype, MemoryBuffer]):
    """构建内存引用。内存缓冲区通过 build_memory_buffer API 构建。
    该 API 会根据 weight_buffer_meta 将权重缓冲区指针分配到内存缓冲区中。

    Args:
        weight_buffer_meta:
        memory_buffers:

    Returns:

    """
    start_idx = {}
    weight_buffers = {}
    for dtype in memory_buffers.keys():
        start_idx[dtype] = 0

    for name, meta_info in sorted(weight_buffer_meta.items()):
        shape = meta_info['shape']
        dtype = meta_info['dtype']

        buffer = memory_buffers[dtype].get(shape, start_index=start_idx[dtype])
        start_idx[dtype] += calc_padded_numel(shape, dtype)
        weight_buffers[name] = buffer

    return weight_buffers


class MemoryBufferModuleWrapper:
    """
    注意，我们没有将 MemoryBufferModuleWrapper 设计为 nn.Module，原因是
    - 这会改变 checkpoint 的名称
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.weight_buffer_meta = get_weight_buffer_meta_from_module(self.module)
        self.memory_buffers = build_memory_buffer(self.weight_buffer_meta)
        build_memory_reference_from_module(self.module, self.memory_buffers)

    def get_memory_buffers(self):
        return self.memory_buffers

    def get_weight_buffer_meta(self):
        return self.weight_buffer_meta


class MegatronMemoryBufferForRollout(object):
    """
    我们假设
    - 推理引擎采用 tp + dp
    - actor 采用 tp + pp + dp
    - 推理引擎与 actor 之间的 tp 应当相同
    - memory_buffers: 包含一个 memory_buffers 列表，每个元素是从 dtype 到 MemoryBuffer 的字典
    - weight_buffers: 包含一个 weight_buffers 列表，每个元素是从名称到参数的字典
    - named_parameters: 从名称到参数的字典，对来自 pp 和 vpp 的名称做了归一化。注意，
        named_parameters 可能无法直接与推理引擎兼容。用户需要自行处理
        这部分内容，例如布局不匹配的情况（如 qkv 转置）。
    - 注意 weight_buffer、named_parameters 和 memory_buffers 共享同一块底层 GPU 内存。
    - 在进行权重同步时，数据通过内存缓冲区传输
    """

    def __init__(self, transform_memory_param_fn):
        self._memory_buffers = []
        self._weight_buffers = []
        self._named_parameters = {}
        self.transform_memory_param_fn = transform_memory_param_fn

    def initialize_weight_buffer(self, weight_buffer_meta_pp: List[Dict[str, Dict]]):
        """
        初始化权重缓冲区。权重缓冲区根据 actor 获得。我们将为 weight_buffer 中的
        每个 dtype 构建一个大缓冲区。

        Args:
            weight_buffer_meta: 包含各个 pp 模型，每个 pp 模型包含一个映射字典

        Returns: None

        """
        self.weight_buffer_meta_pp = weight_buffer_meta_pp

        for weight_buffer_meta in self.weight_buffer_meta_pp:
            memory_buffer = build_memory_buffer(weight_buffer_meta)
            self._memory_buffers.append(memory_buffer)
            self._weight_buffers.append(None)

    def build_memory_reference(self):
        for i, weight_buffer_meta in enumerate(self.weight_buffer_meta_pp):
            self._weight_buffers[i] = build_memory_reference(weight_buffer_meta, self._memory_buffers[i])
        self._named_parameters = self.transform_memory_param_fn(self._weight_buffers)

    @property
    def named_parameters(self):
        return self._named_parameters

    @property
    def weight_buffers(self):
        return self._weight_buffers

    @property
    def memory_buffers(self):
        return self._memory_buffers
