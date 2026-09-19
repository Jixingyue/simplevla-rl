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
基础 tokenizer 类，任何基于混合引擎的 rollout 或使用 vLLM 的推理都需要它。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Union

__all__ = ['HybridEngineBaseTokenizer']


class HybridEngineBaseTokenizer(ABC):
    """tokenizer 的属性名和函数名应与 HF 保持一致，以满足 vllm 的要求"""

    @property
    @abstractmethod
    def vocab_size(self):
        """
        `int`: 基础词表的大小（不含新增 token）。
        """
        pass

    @property
    @abstractmethod
    def pad_token_id(self):
        """
        `Optional[int]`: 词表中 padding token 的 id。如果该 token 尚未设置，则返回 `None`。
        """
        pass

    @property
    @abstractmethod
    def eos_token_id(self):
        """
        `Optional[int]`: 词表中句子结束 token 的 id。如果该 token 尚未设置，则返回 `None`。
        """
        pass

    @property
    @abstractmethod
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: 列出映射到类属性的特殊 token（`'<unk>'`、`'<cls>'` 等）的 id。
        """
        pass

    @property
    @abstractmethod
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: 由去重后的特殊 token（`'<unk>'`、`'<cls>'` 等）组成的列表。

        将 `tokenizers.AddedToken` 类型的 token 转换为字符串。
        """
        pass

    @abstractmethod
    def encode(self, text):
        """
        使用 tokenizer 和词表将字符串转换为 id（整数）序列。

        Args:
            text (`str`, `List[str]` or `List[int]`):
                待编码的第一个序列。可以是字符串、字符串列表（用 `tokenize` 方法分词后的
                字符串）或整数列表。

            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                可选的第二个待编码序列。可以是字符串、字符串列表（用 `tokenize` 方法分词后的
                字符串）或整数列表。
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        使用 tokenizer 和词表将 id 序列转换为字符串，并可选择移除特殊
        token 以及清理分词产生的空格。

        类似于执行 `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`。

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                已分词的输入 id 列表。可以通过 `__call__` 方法获得。
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                解码时是否移除特殊 token。
            clean_up_tokenization_spaces (`bool`, *optional*):
                是否清理分词产生的空格。如果为 `None`，将默认使用
                `self.clean_up_tokenization_spaces`。
            kwargs (additional keyword arguments, *optional*):
                将传递给底层模型特定的 decode 方法。

        Returns:
            `str`: 解码得到的句子。
        """
        pass

    @abstractmethod
    def convert_ids_to_tokens(self,
                              ids: Union[int, List[int]],
                              skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """
        使用词表和新增 token，将单个索引或索引序列转换为单个 token 或 token 序列。

        Args:
            ids (`int` or `List[int]`):
                要转换为 token 的 token id（或多个 token id）。
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                解码时是否移除特殊 token。

        Returns:
            `str` or `List[str]`: 解码得到的 token（或多个 token）。
        """
        pass

    @abstractmethod
    def get_added_vocab(self) -> Dict[str, int]:
        """
        以 token 到索引的字典形式返回词表中的新增 token。结果可能与快速调用有所不同，
        因为目前即使 token 已经在词表中，我们也总是将其添加进去。这一点应当改进。

        Returns:
            `Dict[str, int]`: 新增的 token。
        """
        pass

    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        将 token 序列转换为单个字符串。最简单的方式是 `" ".join(tokens)`，但通常
        我们还希望同时去除子词分词产生的痕迹。

        Args:
            tokens (`List[str]`): 要拼接成字符串的 token。

        Returns:
            `str`: 拼接后的 token。
        """
        pass

    @property
    def is_fast(self):
        return False
