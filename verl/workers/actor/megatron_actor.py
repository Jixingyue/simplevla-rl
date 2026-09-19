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
Megatron Actor。
在 megatron actor 中，区别在于：
1. 我们只构建 minibatch

注意我们的模型不必是 `MegatronModule`，因为我们不在最后一层共享 embedding
"""

from functools import partial
from typing import Iterable, Dict

import torch
from torch import nn
import torch.distributed
# from megatron import get_args
from megatron.optimizer import DistributedOptimizer
from verl.utils.megatron.optimizer_config import OptimizerConfig
from megatron.core import parallel_state as mpu
from megatron.core import ModelParallelConfig
from megatron.core.pipeline_parallel import get_forward_backward_func
# from megatron.core.optimizer import DistributedOptimizer

from omegaconf import OmegaConf
from verl.utils.megatron.tensor_parallel import vocab_parallel_compute_entropy_loss, vocab_parallel_log_probs_from_logits
from verl.utils.megatron.pipeline_parallel import (compute_transformers_input_shapes, make_batch_generator)
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, broadcast_dict_tensor, split_dict_tensor_into_batches

__all__ = ['MegatronPPOActor']


class MegatronPPOActor(BasePPOActor):

    def __init__(self, config, model_config, megatron_config: ModelParallelConfig, actor_module: nn.ModuleList,
                 actor_optimizer: DistributedOptimizer, actor_optimizer_config: OptimizerConfig):
        """MeagtronPPOActor 类。该类实现了模型基于 Megatron 构建时的简单 PPO 逻辑。

        参数:
            config (OmegaConf): 包含 PPO Actor 超参数的基础配置。它必须包含

                ``ppo_micro_batch_size``: 更新 ppo 时的 minibatch 大小。

                ``ppo_mini_batch_size``: 使用 batch 数据更新 ppo 时的 minibatch 大小。

                ``ppo_epochs``: 使用 batch 数据更新 actor 的 epoch 数。

                ``shuffle``: 每个 ppo epoch 后是否打乱数据。

                ``clip_ratio``: ppo 算法的裁剪比例。参见 https://arxiv.org/abs/1707.06347。

                ``entropy_coeff``: PPO 损失的熵系数。参见 https://arxiv.org/abs/1707.06347。
            model_config (OmegaConf): 模型配置。必须包含 ``model_config.vocab_size`` 和
                ``model_config.hidden_size``
            megatron_config (OmegaConf): megatron 配置。必须包含

                ``sequence_parallel_enabled``: 是否启用序列并行。

                ``param_dtype``: 参数的数据类型。

                ``virtual_pipeline_model_parallel_size``: 虚拟流水线模型并行大小，即每个 pp stage 中的 chunk 数量。
            actor_module (nn.ModuleList): actor module 是一个 ModuleList，包含该 pp stage 中的 nn.Module 列表。
                该 rank 上的每个 nn.Module 持有一个 vpp module chunk。更多细节参见 https://arxiv.org/pdf/2104.04473.pdf。
                为了使用这里实现的更新逻辑，actor module 需要遵循以下约束

                1. 必须在任何计算之前实现 unpad_input，并在所有计算之后实现 pad_input。Remove padding 是一种
                移除填充 token 的优化。参见 flash-attn 中的 unpad_input 和 pad_input 函数
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py)。

                2. 每个 pp stage 必须返回形状相同为 [total_nnz, 1, hidden_size] 的 hidden state，
                其中 total_nnz 是该 batch 中有效 token 的数量。如果启用了序列并行，则
                hidden state 的大小为 [total_nnz // tp, 1, hidden_size]。
            actor_optimizer (DistributedOptimizer): 目前，我们只支持 Megatron 中的 DistributedOptimizer。它实现了
                zero1 优化器，将优化器状态分片到各个 dp rank 上。

        >>> def megatron_actor_model_provider(pre_process, post_process):
        >>>     vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        >>>     parallel_model = ParallelMistralForCausalLMRmPadPP(config=actor_model_config,
        >>>                                                        megatron_config=megatron_config,
        >>>                                                        pre_process=pre_process,
        >>>                                                        post_process=post_process).cuda()
        >>>     return parallel_model
        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          megatron_config=megatron_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        super().__init__(config)
        self.model_config = model_config
        self.megatron_config = megatron_config
        # self.megatron_args = get_args()
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.actor_optimizer_config = actor_optimizer_config

        self.optimizer_step_args = OmegaConf.create({
            'skip_grad': None,
            'overlap_dp_param_comm': False,
            'overlap_dp_grad_comm': False,
            'gradient_accumulation_steps': 1,
            'sequence_parallel': self.megatron_config.sequence_parallel,
            'DDP_impl': 'local',
            'layernorm_allreduce_bucket_threshold': 0,
            'pipeline_model_parallel_split_rank': None,
            'reduce_grads_use_alltoall': False
        })

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """给定 input_ids、attention_mask 和 position_ids，计算 responses 的对数概率

        参数:
            data (DataProto): 一个包含以下键的 DataProto

                ``input_ids``: 形状为 [batch_size, sequence_length] 的张量。torch.int64。注意 input_ids 是
                prompt 和 response 的拼接。注意 ``sequence_length = prompt_length + response_length``。

                ``attention_mask``: 形状为 [batch_size, sequence_length] 的张量。torch.int64。

                ``position_ids``: 形状为 [batch_size, sequence_length] 的张量。torch.int64。

                ``responses``:  形状为 [batch_size, response_length] 的张量。torch.int64。

        返回:
            DataProto: torch.Tensor: log_prob 张量
        """
        data.batch = data.batch.contiguous()

        def compute_logprobs_fn(output, data):
            response = data['responses']
            response_length = response.size(1)
            logits = output['logits']
            logits = logits[:, -response_length - 1:-1]
            log_probs = vocab_parallel_log_probs_from_logits(logits, response)
            return {'log_probs': log_probs}

        # 我们在这里默认进行 recompute_old_log_prob。
        # TODO (zhangchi.usc1992): 实际上，这个函数应该只返回 log_prob，这个逻辑应由用户在外部处理
        recompute_old_log_prob = self.config.get('recompute_old_log_prob', True)

        if recompute_old_log_prob or 'old_log_probs' not in data.batch.keys():
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
            batch = data.select(batch_keys=select_keys).batch
            input_ids = batch['input_ids']
            batch_size = input_ids.size(0)
            response = batch['responses']
            response_length = response.size(1)
            with torch.no_grad():
                output = self.forward_backward_batch(data, forward_only=True, post_process_fn=compute_logprobs_fn)
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # 仅在最后一个 rank 上。它应该在每个 tp rank 上
                    log_probs = torch.cat([o['log_probs'] for o in output], dim=0)  # (bs, seq_size)
                    log_probs = log_probs.to(torch.float32)
                else:
                    log_probs = torch.empty(size=(batch_size, response_length),
                                            dtype=torch.float32,
                                            device=input_ids.device)

                # 跨 pp rank 进行 broadcast
                torch.distributed.broadcast(tensor=log_probs,
                                            src=mpu.get_pipeline_model_parallel_last_rank(),
                                            group=mpu.get_pipeline_model_parallel_group(),
                                            async_op=False)

        # 每次计算后清空缓存
        torch.cuda.empty_cache()

        return log_probs

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """构建用于更新 actor 的 minibatch 迭代器

        参数:
            data (DataProto): 一个包含以下键的 DataProto

                ``input_ids``: 形状为 [batch_size, sequence_length] 的张量。torch.int64，其中 ``sequence_length = prompt_length + response_length``

                ``attention_mask``: 形状为 [batch_size, sequence_length] 的张量。torch.int64

                ``position_ids``: 形状为 [batch_size, sequence_length] 的张量。torch.int64

                ``responses``: 形状为 [batch_size, response_length] 的张量。torch.int64。注意 responses = input_ids[:, -response_length:]

                ``old_log_probs``: 形状为 [batch_size, response_length] 的张量。torch.float32。responses 的对数概率。

                ``advantages``: 形状为 [batch_size, response_length] 的张量。torch.float32。responses 的优势值。
                详情参见 PPO 论文 https://arxiv.org/abs/1707.06347

        返回:

        """
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs,
                                  dataloader_kwargs={'shuffle': self.config.shuffle})

    def forward_backward_batch(self, data: DataProto, forward_only=False, post_process_fn=None):
        """
        我们假设：
        - 模型接收输入: (input_ids, attention_mask, position_ids)。输入不做 rmpad
        - 如果启用了序列并行，通信形状为 (total_nnz_pad_to_sp // tp_size, 1, hidden_size)
        """
        # 从最后一个 pp rank broadcast 到所有其他 pp rank
        # TODO: 实际上，我们只需要控制采样顺序。
        broadcast_dict_tensor(data.batch,
                              src=mpu.get_pipeline_model_parallel_last_rank(),
                              group=mpu.get_pipeline_model_parallel_group())
        # 切分为 micro-batch
        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)

        if data.meta_info.get('micro_batch_size', None) is not None:
            batch_size = data.meta_info['micro_batch_size']
        else:
            batch_size = self.config.ppo_micro_batch_size
        batches = split_dict_tensor_into_batches(data.batch, batch_size=batch_size)
        # 计算 pp stage 的输入形状
        input_shapes = compute_transformers_input_shapes(
            batches,
            meta_info={
                'sequence_parallel': self.megatron_config.sequence_parallel,
                'hidden_size': self.model_config.hidden_size
            })
        n_micro_batch = len(batches)
        seq_len = batches[0]['input_ids'].shape[1]

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            if forward_only:
                if post_process_fn is None:
                    return 1.0, {'logits': output.logits}
                else:
                    return 1.0, post_process_fn(output, data)

            responses = data['responses']
            response_length = responses.size(1)
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            old_log_prob = data['old_log_probs']
            advantages = data['advantages']

            clip_ratio = meta_info['clip_ratio']
            entropy_coeff = meta_info['entropy_coeff']

            # 计算 policy loss
            logits = output.logits
            logits = logits[:, -response_length - 1:-1]
            log_prob = vocab_parallel_log_probs_from_logits(logits, responses)
            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                          log_prob=log_prob,
                                                                          advantages=advantages,
                                                                          eos_mask=response_mask,
                                                                          cliprange=clip_ratio)
            entropy_loss = vocab_parallel_compute_entropy_loss(logits, eos_mask=response_mask)
            policy_loss = pg_loss - entropy_loss * entropy_coeff
            # 返回 loss 和统计信息
            stats = {
                'actor/entropy_loss': entropy_loss.detach().item(),
                'actor/pg_loss': pg_loss.detach().item(),
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                'actor/ppo_kl': ppo_kl.detach().item()
            }
            return policy_loss, stats

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            position_ids = batch['position_ids']
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            if forward_only:
                meta_info = None
            else:
                meta_info = {'clip_ratio': self.config.clip_ratio, 'entropy_coeff': self.config.entropy_coeff}
            return output, partial(loss_func, data=batch, meta_info=meta_info)

        # batch 应该是 micro-batch 内部的 batch 列表
        batch_generator = make_batch_generator(batches, vpp_size=len(self.actor_module))

        # TODO: 我们可以改用新的 schedule
        # 对于 flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                input_shapes=input_shapes,  # 必须为 flash-attn 序列打包进行设置
                seq_length=batch_size * seq_len,  # 当设置了 input_shapes 时不使用
                hidden_size=self.model_config.hidden_size,  # 当设置了 input_shapes 时不使用
                micro_batch_size=1,  # 当设置了 input_shapes 时不使用
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=batch_size * seq_len,  # 在 pp = 1 时使用
                hidden_size=self.model_config.hidden_size,  # 在 pp = 1 时使用
                micro_batch_size=1,  # 在 pp = 1 时使用
                forward_only=forward_only,
            )
        # loss_reduces 包含 loss_func 返回的统计信息
        return losses_reduced

    def update_policy(self, dataloader: Iterable[DataProto]) -> Dict:
        """使用 DataProto 迭代器更新策略

        参数:
            dataloader (Iterable[DataProto]): 遍历 DataProto 的迭代器，由 ``make_minibatch_iterator`` 返回。
                每个 data batch 的键在 make_minibatch_iterator 中有描述。

        返回:
            Dict: 包含统计信息的字典。注意这些统计信息仅在最后一个 pp stage 有效，
            用户需要手动合并各 dp rank 上的输出。

        """
        metrics = {}
        for data in dataloader:
            # data = data.batch.to(self.actor_module.device)
            self.actor_optimizer.zero_grad()
            # 使用 use_contiguous_buffers_in_local_ddp 且不使用 overlap_dp_param_comm
            for chunk in self.actor_module:
                # 如果使用分布式优化器，zero grad buffer 将由优化器处理
                chunk.zero_grad_buffer(zero_buffer=(not self.actor_optimizer_config.use_distributed_optimizer))

            metric_micro_batch = self.forward_backward_batch(data)
            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)  # 将该 micro-batch 的指标追加到全局 metrics 中。

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step(
                self.megatron_config, self.megatron_config.timers)
            if update_successful:
                # 在新版 megatron 中，allgather 已在 optimizer.step 中执行
                pass
            else:
                raise NotImplementedError

            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)  # 将该 micro-batch 的指标追加到全局 metrics 中。

        # 每次计算后清空缓存
        torch.cuda.empty_cache()

        return metrics
