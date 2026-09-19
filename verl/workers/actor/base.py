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
Actor 的基类
"""
from abc import ABC, abstractmethod
from typing import Iterable, Dict

from verl import DataProto
import torch

__all__ = ['BasePPOActor']


class BasePPOActor(ABC):

    def __init__(self, config):
        """PPO actor 的基类

        参数:
            config (DictConfig): 传递给 PPOActor 的配置。我们期望其类型为
                DictConfig（https://omegaconf.readthedocs.io/），但一般情况下也可以是任何 namedtuple。
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """根据一批数据计算 logits。

        参数:
            data (DataProto): 以 DataProto 表示的一批数据。它必须包含键 ```input_ids```、
                ```attention_mask``` 和 ```position_ids```。

        返回:
            DataProto: 包含键 ```log_probs``` 的 DataProto


        """
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> Dict:
        """使用 DataProto 迭代器更新策略

        参数:
            data (DataProto): 由 ```make_minibatch_iterator``` 返回的
                DataProto 迭代器

        返回:
            Dict: 一个包含任意内容的字典。通常包含更新模型过程中的统计信息，
            如 ```loss```、```grad_norm``` 等。

        """
        pass
