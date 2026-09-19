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
"""用于 tokenization 的工具函数。"""
import warnings

__all__ = ['hf_tokenizer']


def set_pad_token_id(tokenizer):
    """当 pad_token_id 为 None 时，将其设置为 eos_token_id。

    Args:
        tokenizer (transformers.PreTrainedTokenizer): 要设置的 tokenizer。

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f'tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f'tokenizer.pad_token is None. Now set to {tokenizer.eos_token}')


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """创建一个 huggingface 预训练 tokenizer。

    Args:
        name (str): tokenizer 的名称。
        correct_pad_token (bool): 是否修正 pad token id。
        correct_gemma2 (bool): 是否修正 gemma2 tokenizer。
        **kwargs: 传递给 tokenizer 的关键字参数。

    Returns:
        transformers.PreTrainedTokenizer: 预训练 tokenizer。

    """
    from transformers import AutoTokenizer, AutoConfig, AutoProcessor
    if correct_gemma2 and isinstance(name_or_path, str) and 'gemma-2-2b-it' in name_or_path:
        # gemma2 的 EOS token 存在歧义，可能会恶化 RL 性能。
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn('Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.')
        kwargs['eos_token'] = '<end_of_turn>'
        kwargs['eos_token_id'] = 107
    
    model = kwargs.get("model",None)
    
    if model == "openvla-oft":   
        from verl.utils.vla_utils.openvla_oft.configuration_prismatic import OpenVLAConfig
        from verl.utils.vla_utils.openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        print("*********USE VLA tokenizer*************")
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        tokenizer=processor.tokenizer
    elif model == "openvla":
        from verl.utils.vla_utils.openvla.configuration_prismatic import OpenVLAConfig
        from verl.utils.vla_utils.openvla.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        print("*********USE VLA tokenizer*************")
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        tokenizer=processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
        
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer
