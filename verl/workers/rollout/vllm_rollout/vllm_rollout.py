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
可在不同后端中使用的 vllm_rollout
与 FSDP 配合使用时：
- 使用 DTensor 权重加载器（推荐）或 HF 权重加载器
- 利用 FSDP 的 state_dict 在 vLLM 的各个 tp rank 之间同步权重
与 Megatron 配合使用时：
- 使用 Megatron 权重加载器
- 训练期间，只有当前 pp 阶段持有参数
- 推理之前，将当前 pp rank 的参数广播给所有其他 pp rank（所有 pp rank 持有全部参数）
- 将参数绑定到推理引擎
- 在 tp 维度上做推理，pp 被视作额外的 dp
- 推理结束后，释放所有不属于当前 pp rank 的参数
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. 在 vllm 中支持 pp
# 2. 传入 tokenizer 是否必要？这里并没有发生编码/解码
# 3. 简化初始化逻辑


# NOTE(sgm): 为 verl 添加。可以通过让 dataloader 直接产出不含 padding 的 List[int] 来优化。
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # 去除 prompt token_id 中的左侧填充
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """一个 vLLM rollout。它要求模块受 vllm 支持。

        Args:
            module: 这里的 module 遵循 huggingface API
            config: DictConfig
            tokenizer: 任务/模型的 tokenizer
            model_hf_config: 用于在 vllm 中初始化生成模型的 huggingface 配置
            **kwargs: train_tp，供 Megatron 后端初始化混合引擎（零冗余）进程组使用
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"

        if kwargs.get('train_tp', None) is not None:
            # 使用 megatron 部署
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(actor_module,
                                    tokenizer=tokenizer,
                                    model_hf_config=model_hf_config,
                                    tensor_parallel_size=tensor_parallel_size,
                                    dtype=config.dtype,
                                    enforce_eager=config.enforce_eager,
                                    gpu_memory_utilization=config.gpu_memory_utilization,
                                    skip_tokenizer_init=False,
                                    max_model_len=config.prompt_length + config.response_length,
                                    load_format=config.load_format)

        # offload vllm 模型以降低峰值内存占用
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # 可以设为 0，让 actor 重新计算
            max_tokens=config.response_length,
        )

        # 之后我们可能会把结果整体 detokenize
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # 支持从配置文件添加任意采样参数
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # 更新采样参数
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # 回滚到之前的采样参数
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # 重建 vllm 缓存引擎
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        n_samples = prompts.meta_info.get('n_samples', 1)
        idx = prompts.batch['input_ids'].repeat_interleave(n_samples,dim=0)  # (bs, prompt_length)
        # 左填充的 attention_mask
        attention_mask = prompts.batch['attention_mask'].repeat_interleave(n_samples,dim=0)
        position_ids = prompts.batch['position_ids'].repeat_interleave(n_samples,dim=0)

        # 用于构建 attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # 将 idx 从 torch.Tensor 解析为 List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
            }

        # 用户可以在不同运行中自定义不同的 sampling_params
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # 因为我们已经将其转换为 prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        response = output[0].to(idx.device)  # (bs, response_length)
        log_probs = output[1].to(idx.device)  # (bs, response_length)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): 修复 right_pad 情况下的 position_ids
        # prompt: 左填充 + response: 右填充
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # 所有 tp rank 在这里应包含相同的数据。所有 rank 上的数据都是有效的
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # 这里 input_ids 变成了完整的句子
                # 'old_log_probs': log_probs, # 我们将用 actor 重新计算 old log prob
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # 释放 vllm 缓存引擎
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
