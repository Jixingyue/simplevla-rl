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
实现任意两个函数、模块之间的基础数据传输协议。
我们可以通过继承 Protocol 来定义更详细的 batch 信息及特定 key
"""

import numpy as np
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union

import torch
import tensordict
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from verl.utils.py_functional import union_two_dict

__all__ = ['DataProto', 'union_tensor_dict',]

try:
    tensordict.set_lazy_legacy(False).set()
except:
    pass


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """合并两个 tensordict。"""
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, \
        f'Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}'
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(tensor_dict2[key]), \
                f'{key} in tensor_dict1 and tensor_dict2 are not the same object'

    return tensor_dict1


def union_numpy_dict(tensor_dict1: dict[np.ndarray], tensor_dict2: dict[np.ndarray]) -> dict[np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            assert np.all(tensor_dict2[key] == tensor_dict1[key]), \
                f'{key} in tensor_dict1 and tensor_dict2 are not the same object'
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output


def collate_fn(x: list['DataProtoItem']):
    batch = []
    non_tensor_batch = []
    for data in x:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.array(val, dtype=object)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@dataclass
class DataProtoItem:
    # TODO(zhangchi.usc1992) 添加一致性检查
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)


@dataclass
class DataProto:
    """
    DataProto 是一种数据结构，旨在为函数之间的数据交换提供标准协议。
    它包含一个 batch（TensorDict）和一个 meta_info（Dict）。batch 是一个 TensorDict https://pytorch.org/tensordict/。
    TensorDict 允许你像操作单个 Tensor 一样操作一个由 Tensor 组成的字典。理想情况下，具有相同 batch size 的
    张量应放入 batch 中。
    """
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        # 执行必要的检查
        self.check_consistency()

    def __len__(self):
        return self.batch.batch_size[0]

    def __getitem__(self, item):
        tensor_data = self.batch[item]
        non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
        return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

    def slice(self, index):
        tensor_data = self.batch[index]
        non_tensor_data = {key: val[index] for key, val in self.non_tensor_batch.items()}
        return DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

    def slice_batch(self, start, length, dim=0):
        """
        注意该操作是原地（in-place）进行的
        """
        for key, val in self.batch.items():
            self.batch[key] = val.narrow(start=start, length=length, dim=dim)


    def __getstate__(self):
        import io
        buffer = io.BytesIO()
        if tensordict.__version__ >= '0.5.0' and self.batch is not None:
            self.batch = self.batch.contiguous()
            self.batch = self.batch.consolidate()
        torch.save(self.batch, buffer)
        return buffer, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        batch_deserialized, non_tensor_batch, meta_info = data
        batch_deserialized.seek(0)
        batch = torch.load(batch_deserialized,
                           weights_only=False,
                           map_location='cpu' if not torch.cuda.is_available() else None)
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def check_consistency(self):
        """检查 DataProto 的一致性。主要针对 batch 和 non_tensor_batch。
        我们将该函数公开，以便用户可以直接自行调用。
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, 'only support num_batch_dims=1'

        if len(self.non_tensor_batch) != 0:
            # TODO: 如果需要，我们实际上可以解除这个限制
            assert len(self.batch.batch_size) == 1, 'only support num_batch_dims=1 when non_tensor_batch is not empty.'

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert isinstance(
                    val, np.ndarray
                ) and val.dtype == object, 'data in the non_tensor_batch must be a numpy.array with dtype=object'
                assert val.shape[
                    0] == batch_size, f'key {key} length {len(val)} is not equal to batch size {batch_size}'

    @classmethod
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None):
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f'Unsupported type in data {type(val)}')

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @classmethod
    def from_dict(cls, tensors: Dict[str, torch.Tensor], non_tensors=None, meta_info=None, num_batch_dims=1):
        """从一个张量字典创建 DataProto。这里假设：
        1. tensors 中所有张量具有相同的 dim0
        2. 只有 dim0 是 batch 维度
        """
        assert len(tensors) > 0, 'tensors must not be empty'
        assert num_batch_dims > 0, 'num_batch_dims must be greater than zero'
        if non_tensors is not None:
            assert num_batch_dims == 1, 'only support num_batch_dims=1 when non_tensors is not None.'

        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # 获取并检查 batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, \
                    f'Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}'

        for key, val in non_tensors.items():
            non_tensors[key] = np.array(val, dtype=object)

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size)
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device) -> 'DataProto':
        """将 batch 移动到指定设备

        Args:
            device (torch.device, str): torch 设备

        Returns:
            DataProto: 当前的 DataProto

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> 'DataProto':
        """通过 batch_keys 和 meta_info_keys 选择 DataProto 的一个子集

        Args:
            batch_keys (list, optional): 字符串列表，指示要选择的 batch 中的键
            meta_info_keys (list, optional): 键列表，指示要选择的 meta info

        Returns:
            DataProto: 包含所选 batch_keys 和 meta_info_keys 的 DataProto
        """
        # TODO (zhangchi.usc1992) 是否要进行复制
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return DataProto(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> 'DataProto':
        """通过 `batch_keys` 和 `meta_info_keys` 从 DataProto 中弹出（pop）一个子集

        Args:
            batch_keys (list, optional): 字符串列表，指示要从 batch 中弹出的键
            meta_info_keys (list, optional): 键列表，指示要弹出的 meta info

        Returns:
            DataProto: 包含被弹出的 batch_keys 和 meta_info_keys 的 DataProto
        """
        assert batch_keys is not None
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []

        tensors = {}
        # 张量 batch
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # 非张量 batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(self, old_keys=None, new_keys=None) -> 'DataProto':
        """
        注意该函数仅重命名 batch 中的键
        """

        def validate_input(keys):
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f'keys must be a list or a string, but got {type(keys)}')
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f'new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}')

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: 'DataProto') -> 'DataProto':
        """与另一个 DataProto 求并集（union）。分别对 batch 和 meta_info 进行 union。
        在以下情况下抛出错误：
        - batch 中存在冲突的键且它们不相等
        - 两个 data batch 的 batch size 不相同
        - meta_info 中存在冲突的键且它们不相同。

        Args:
            other (DataProto): 用于 union 的另一个 DataProto

        Returns:
            DataProto: union 之后的 DataProto
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        """从 DataProto 构造一个迭代器。这建立在 TensorDict 可以作为普通 Pytorch
        dataset 使用的基础上。详见 https://pytorch.org/tensordict/tutorials/data_fashion。

        Args:
            mini_batch_size (int): 遍历数据集时的 mini-batch 大小。我们要求
                ``batch.batch_size[0] % mini_batch_size == 0``
            epochs (int): 遍历数据集时的 epoch 数。
            dataloader_kwargs: 内部会返回一个基于 batch 的 DataLoader。
                dataloader_kwargs 是传递给 DataLoader 的 kwargs

        Returns:
            Iterator: 每次产出一个 mini-batch 数据的迭代器。总迭代步数为
            ``self.batch.batch_size * epochs // mini_batch_size``
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # 我们可以直接从 TensorDict 创建 dataloader
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, Dict)
        train_dataloader = DataLoader(dataset=self,
                                      batch_size=mini_batch_size,
                                      collate_fn=collate_fn,
                                      generator=generator,
                                      **dataloader_kwargs)

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def chunk(self, chunks: int) -> List['DataProto']:
        """沿 dim=0 将 batch 切分成若干块。切分后 meta_info 会传递给每个 DataProto。

        Args:
            chunks (int): 沿 dim=0 切分的块数

        Returns:
            List[DataProto]: 切分后得到的 DataProto 列表
        """
        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            non_tensor_lst = np.array_split(val, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        assert len(batch_lst) == chunks, (f"len(batch_lst) ({len(batch_lst)}) != chunks ({chunks})")
        assert len(non_tensor_batch_lst) == chunks, (f"len(non_tensor_batch_lst) ({len(non_tensor_batch_lst)}) != chunks ({chunks})")
        for i in range(chunks):
            output.append(
                DataProto(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info))

        return output

    @staticmethod
    def concat(data: List['DataProto']) -> 'DataProto':
        """拼接一个 DataProto 列表。batch 沿 dim=0 进行拼接。
        假定各 meta_info 完全相同，将使用第一个 DataProto 的 meta_info。

        Args:
            data (List[DataProto]): DataProto 列表

        Returns:
            DataProto: 拼接后的 DataProto
        """
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        if batch_lst[0] is not None:
            new_batch = torch.cat(batch_lst, dim=0)
        else:
            new_batch = None

        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        return DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    def reorder(self, indices):
        """
        注意该操作是原地（in-place）进行的
        """
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}


import ray


@dataclass
class DataProtoFuture:
    """
    DataProtoFuture 旨在消除 driver 上的实际数据获取（fetching）。这样一来，driver 无需等待
    数据，从而可以实现异步执行。 
    DataProtoFuture 包含一个来自另一个 WorkerGroup 的 future 列表，其大小为 world_size。
    - collect_fn 是一个 Callable，用于将 future 列表归约为一个 DataProto
    - dispatch_fn 是一个 Callable，用于将 DataProto 划分为大小为 world_size 的 DataProto 列表，然后再进行选择

    潜在问题：我们可以优化 dispatch_fn(collect_fn)，使得只在目的地获取所需的数据
    - DataProtoFuture 只支持从某个方法的输出直接传递到另一个输入。你不能在 driver 中对
    DataProtoFuture 执行任何操作。
    """
    collect_fn: Callable
    futures: List[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: List[ray.ObjectRef]) -> 'DataProtoFuture':
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> List['DataProtoFuture']:
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # 注意我们不能直接传递 i 和 chunks
            def dispatch_fn(x, i, chunks):
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(collect_fn=self.collect_fn,
                                         dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks),
                                         futures=self.futures)
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        output = ray.get(self.futures)  # 得到 dp_size 个结果。
        for o in output:
            assert isinstance(o, DataProto)
        output = self.collect_fn(output)  # 选择 dp，concat
        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)  # 沿 batch 维度切分，使用 dp 进行选择
        return output
