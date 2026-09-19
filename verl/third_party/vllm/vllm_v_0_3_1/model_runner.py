# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py

from typing import Dict, List, Optional, Tuple, Set, Union
import contextlib
import time
import numpy as np
import torch
import torch.nn as nn

from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import InputMetadata, SamplingMetadata
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.utils import in_wsl
from vllm.worker.model_runner import ModelRunner, CUDAGraphRunner, _async_h2d

from .model_loader import get_model

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]
_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
# 为 batch size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256 捕获图。
# NOTE: 如果修改此列表，需要同步更新 _get_graph_batch_size。
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]


class ModelRunner(ModelRunner):

    def __init__(
        self,
        model: Union[nn.Module, Dict], # 模型本身或其参数字典
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config

        # 在 tests/samplers/test_sampler.py 中 model_config 可能为 None。
        # FIXME(woosuk): 这是让测试正常运行的一个 hack，需要重构。
        self.sliding_window = (model_config.get_sliding_window() if model_config is not None else None)

        self.device_config = (device_config if device_config is not None else DeviceConfig())
        self.device = self.device_config.device

        self.model = model  # 这将被 get_model() 替换
        self.block_size = None  # 在初始 profiling 之后设置。
        self.lora_manager = None

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # 在图捕获期间设置。

        self.max_context_len_to_capture = (self.model_config.max_context_len_to_capture
                                           if self.model_config is not None else 0)
        # 使用 CUDA graph 时，输入的 block table 必须填充到
        # max_context_len_to_capture。然而，在 Python 中创建 block table
        # 可能开销很大。为了优化这一点，我们将 block table 缓存在 numpy 中，
        # 并且每次迭代只复制实际的输入内容。
        # 缓存的 block table 形状为
        # (max batch size to capture, max context len to capture / block size)。
        self.graph_block_tables = None  # 在初始 profiling 之后设置。
        # 缓存 in_wsl 的结果
        self.in_wsl = in_wsl()
        self.kv_cache_dtype = kv_cache_dtype

    def load_model(self) -> None:
        self.model = get_model(actor_model=self.model,
                               model_config=self.model_config,
                               device_config=self.device_config,
                               lora_config=self.lora_config)
        vocab_size = self.model.config.vocab_size

        if self.lora_config:
            assert hasattr(
                self.model,
                "supported_lora_modules") and self.model.supported_lora_modules, "Model does not support LoRA"
            assert hasattr(self.model, "embedding_modules"), "Model does not have embedding_modules"
            assert hasattr(self.model, "embedding_padding_modules"), "Model does not have embedding_padding_modules"
            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens + self.scheduler_config.max_paddings, vocab_size,
                self.lora_config, self.device, self.model.embedding_modules, self.model.embedding_padding_modules)
            self.model = self.lora_manager.create_lora_manager(self.model)

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
        subquery_lens: Optional[List[int]],
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0

        max_subquery_len = max(subquery_lens) if subquery_lens else 1
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                assert subquery_lens is not None
                subquery_len = subquery_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token 位置不需要采样，跳过
                    categorized_sample_indices_start_idx += subquery_len - 1

                categorized_sample_indices[sampling_params.sampling_type].append(categorized_sample_indices_start_idx)
                categorized_sample_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx, selected_token_start_idx + subquery_len - 1))
                selected_token_indices.append(selected_token_start_idx + subquery_len - 1)
                selected_token_start_idx += max_subquery_len
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(range(selected_token_start_idx, selected_token_start_idx + num_seqs))
                selected_token_start_idx += num_seqs

                categorized_sample_indices[sampling_params.sampling_type].extend(
                    range(categorized_sample_indices_start_idx, categorized_sample_indices_start_idx + num_seqs))
                categorized_sample_indices_start_idx += num_seqs

        selected_token_indices = _async_h2d(selected_token_indices,
                                            dtype=torch.long,
                                            target_device=self.device,
                                            pin_memory=not self.in_wsl)
        categorized_sample_indices = {
            t: _async_h2d(seq_ids, dtype=torch.int, target_device=self.device, pin_memory=not self.in_wsl)
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
        )
        return sampling_metadata

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, SamplingMetadata, Set[int], LoRAMapping]:
        # NOTE: 我们假设组内所有序列要么都是 prompt，要么都是 decode。
        is_prompt = seq_group_metadata_list[0].is_prompt
        # 准备输入张量。
        if is_prompt:
            (input_tokens, input_positions, input_metadata, prompt_lens, subquery_lens, lora_index_mapping,
             lora_prompt_mapping, lora_requests) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions, input_metadata, lora_index_mapping, lora_prompt_mapping,
             lora_requests) = self._prepare_decode(seq_group_metadata_list)
            prompt_lens = []
            subquery_lens = None
        sampling_metadata = self._prepare_sample(seq_group_metadata_list, prompt_lens, subquery_lens)
        if self.lora_config:
            flat_lora_index_mapping = [item for sublist in lora_index_mapping for item in sublist]
            lora_mapping = LoRAMapping(
                flat_lora_index_mapping,
                lora_prompt_mapping,
            )
        else:
            lora_mapping = None

        return (input_tokens, input_positions, input_metadata, sampling_metadata, lora_requests, lora_mapping)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, input_metadata, sampling_metadata, lora_requests,
         lora_mapping) = self.prepare_input_tensors(seq_group_metadata_list)

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        # 执行模型。
        if input_metadata.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model
        hidden_states = model_executable(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
        )

        # 采样下一个 token。
        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )
        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        # 启用 top-k 采样以反映准确的内存使用。
        vocab_size = self.model_config.get_vocab_size()
        # FIXME(sgm): 该采样参数会调用 cumsum()，导致确定性的 cumsum 抛出错误
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # 这表示将拥有各自不同 lora 的最大不同请求数，
        # 因此也就是最大内存消耗量。从传入的 lora request 复制创建
        # 虚拟 lora request，其中包含来自 lora 预热路径的 lora。
        dummy_lora_requests = []
        dummy_lora_requests_per_seq = []
        if self.lora_config:
            for idx in range(self.lora_config.max_loras):
                lora_id = idx + 1
                dummy_lora_request = LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_local_path="/not/a/real/path",
                )
                self.lora_manager.add_dummy_lora(dummy_lora_request, rank=LORA_WARMUP_RANK)
                dummy_lora_requests.append(dummy_lora_request)
            dummy_lora_requests_per_seq = [
                dummy_lora_requests[idx % len(dummy_lora_requests)] for idx in range(max_num_seqs)
            ]

        # 使用 max_num_sequences 个序列、总 token 数等于
        # max_num_batched_tokens 来统计内存使用。
        seqs: List[SequenceGroupMetadata] = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs + (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id] if dummy_lora_requests_per_seq else None,
            )
            seqs.append(seq)

        # 使用虚拟输入运行模型。
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [(None, None)] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return
