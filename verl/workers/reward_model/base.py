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
reward model 的基类
"""

from abc import ABC, abstractmethod

from verl import DataProto


class BasePPORewardModel(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def compute_reward(self, data: DataProto) -> DataProto:
        """给定 input_ids 计算 reward。transformers 应输出一个形状为
           [batch_size, sequence_length] 的张量，并且应收集 [EOS] mask 处的值。

        参数:
            data: 必须包含键 "input_ids"、"attention_mask" 和 "position_ids"。
                - input_ids: [batch_size, sequence_length]
                - attention_mask: [batch_size, sequence_length]
                - position_ids: [batch_size, sequence_length]

        返回: 一个包含 "reward" 的 data pass protocol。只有 [EOS] 位置包含 reward。
            其他位置的 reward 应为零。注意，如果我们使用 dense reward，这在将来可能会改变。
            因此，我们为一般情况保留此接口。
            - reward: [batch_size, sequence_length]。

        """
        pass
