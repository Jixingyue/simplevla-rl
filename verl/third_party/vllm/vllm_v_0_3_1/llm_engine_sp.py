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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py

import os
import socket
import time
import torch
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

from vllm.lora.request import LoRARequest
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup, SequenceGroupMetadata, SequenceGroupOutput,
                           SequenceOutput, SequenceStatus)
from vllm.transformers_utils.tokenizer import detokenize_incrementally
from vllm.engine.metrics import StatLogger, Stats
from vllm.utils import Counter
import torch.nn as nn
from .arg_utils import EngineArgs
from .tokenizer import TokenizerGroup

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


class LLMEngine:
    """接收请求并生成文本的 LLM 引擎。

    这是 vLLM 引擎的主类。它接收来自客户端的请求并从 LLM 生成
    文本。它包含一个 tokenizer、一个语言模型（可能分布在多个 GPU
    上），以及为中间状态（即 KV cache）分配的 GPU 内存空间。该类
    利用迭代级调度和高效的内存管理来最大化服务吞吐量。

    `LLM` 类包装本类用于离线批量推理，`AsyncLLMEngine` 类包装
    本类用于在线服务。

    NOTE: 配置参数派生自 `EngineArgs` 类。完整参数列表请参见
    `EngineArgs`。

    Args:
        model_config: 与 LLM 模型相关的配置。
        cache_config: 与 KV cache 内存管理相关的配置。
        parallel_config: 与分布式执行相关的配置。
        scheduler_config: 与请求调度器相关的配置。
        distributed_init_method: 分布式执行的初始化方法。详见
            `torch.distributed.init_process_group`。
        placement_group: 用于分布式执行的 Ray placement group。
            分布式执行必需。
        log_stats: 是否记录统计信息。
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict], # 模型本身或其参数字典
        tokenizer: nn.Module,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        distributed_init_method: str,
        placement_group: Optional[None],
        log_stats: bool,
    ) -> None:
        logger.info("Initializing an LLM engine with config: "
                    f"model={model_config.model!r}, "
                    f"tokenizer={model_config.tokenizer!r}, "
                    # f"tokenizer_mode={model_config.tokenizer_mode}, "
                    f"revision={model_config.revision}, "
                    f"tokenizer_revision={model_config.tokenizer_revision}, "
                    # f"trust_remote_code={model_config.trust_remote_code}, "
                    f"dtype={model_config.dtype}, "
                    f"max_seq_len={model_config.max_model_len}, "
                    # f"download_dir={model_config.download_dir!r}, "
                    # f"load_format={model_config.load_format}, "
                    f"disable_custom_all_reduce={parallel_config.disable_custom_all_reduce}, "
                    f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
                    f"quantization={model_config.quantization}, "
                    f"seed={model_config.seed})")
        # TODO(woosuk): 在调试模式下打印更多配置。

        self.model_config = model_config  # TODO: 目前是 hfconfig
        self.cache_config = cache_config
        self.lora_config = lora_config
        assert self.cache_config.sliding_window == getattr(self.model_config.hf_config, "sliding_window", None)
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.log_stats = log_stats
        self._verify_args()

        # self.model = model # 不应存储模型，应将其删除
        # TODO(shengguangming): 或许可以选择在此初始化或从参数传入
        self._init_tokenizer(tokenizer)

        self.seq_counter = Counter()

        # 创建并行 GPU worker。
        self._init_workers_sp(model, distributed_init_method)

        # 分析内存使用情况并初始化缓存。
        self._init_cache_sp()

        # 创建调度器。
        # NOTE(shengguangming): 每个进程将拥有独立的调度器
        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

        # 指标记录。
        if self.log_stats:
            self.stat_logger = StatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC)

        # 日志记录。
        self.last_logging_time = 0.0
        # 列表元素为 (时间戳, token 数)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # 列表元素为 (时间戳, token 数)
        self.num_generation_tokens: List[Tuple[float, int]] = []

    def _init_tokenizer(self, tokenizer, **tokenizer_init_kwargs):
        init_kwargs = dict(enable_lora=bool(self.lora_config),
                           max_num_seqs=self.scheduler_config.max_num_seqs,
                           max_input_length=None)
        init_kwargs.update(tokenizer_init_kwargs)
        self.tokenizer: TokenizerGroup = TokenizerGroup(tokenizer, **init_kwargs)

    # TODO: 检查 get_lora_tokenizer 函数
    def get_tokenizer_for_seq(self, sequence: Sequence):
        return self.tokenizer.get_lora_tokenizer(sequence.lora_request)

    def _init_workers_sp(self, model, distributed_init_method: str):
        # 延迟导入 Worker，以避免在 Worker 中设置 CUDA_VISIBLE_DEVICES
        # 之前导入 torch.cuda/xformers
        from .worker import Worker  # pylint: disable=import-outside-toplevel

        rank = int(os.getenv("RANK"))

        self.worker = Worker(
            model,
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            rank,
            distributed_init_method,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
        )

        # NOTE(shengguangming): torch.distributed.init_process_group 会在 init_model() 内部被调用
        self.worker.init_model()
        self.worker.load_model()

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache_sp(self) -> None:
        """分析内存使用情况并初始化 KV cache。"""
        # 获取 GPU 和 CPU 上可分配的最大块数。
        num_blocks = self.worker.profile_num_available_blocks(
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
            cache_dtype=self.cache_config.cache_dtype,
        )

        # NOTE(shengguangming): 现在我们不再使用共享的集中式控制器，
        # 而是每个进程拥有自己的调度器
        num_gpu_blocks = num_blocks[0]
        num_cpu_blocks = num_blocks[1]

        # FIXME(woosuk): 改为 debug 日志。
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(f"The model's max seq len ({self.model_config.max_model_len}) "
                             "is larger than the maximum number of tokens that can be "
                             f"stored in KV cache ({max_seq_len}). Try increasing "
                             "`gpu_memory_utilization` or decreasing `max_model_len` when "
                             "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # 初始化缓存。
        self.worker.init_cache_engine(cache_config=self.cache_config)
        self.worker.warm_up_model()

    def init_cache_engine(self):
        self.worker.init_cache_engine(cache_config=self.cache_config)

    def free_cache_engine(self):
        self.worker.free_cache_engine()

    @classmethod
    def from_engine_args(cls, model, tokenizer, engine_args: EngineArgs) -> "LLMEngine":
        """根据引擎参数创建 LLM 引擎。"""
        # 创建引擎配置。
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # 初始化集群。
        distributed_init_method, placement_group = initialize_cluster(parallel_config)
        # 创建 LLM 引擎。
        engine = cls(model,
                     tokenizer,
                     *engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        prefix_pos: Optional[int] = None,
    ) -> None:
        """向引擎的请求池中添加一个请求。

        请求会被加入请求池，并在调用 `engine.step()` 时由调度器处理。
        具体的调度策略由调度器决定。

        Args:
            request_id: 请求的唯一 ID。
            prompt: prompt 字符串。如果提供了 prompt_token_ids，可以为
                None。
            sampling_params: 文本生成的采样参数。
            prompt_token_ids: prompt 的 token ID。若为 None，则使用
                tokenizer 将 prompt 转换为 token ID。
            arrival_time: 请求的到达时间。若为 None，则使用当前的
                单调时钟时间。
            prefix_pos: 若不为 None，则将给定位置作为每个 prompt 的
                前缀位置。我们会缓存前缀的 KV cache，并在后续具有
                相同前缀的请求中复用它。这是一个实验性功能，未来可能
                被自动前缀缓存（automatic prefix caching）取代。

        Details:
            - 若 arrival_time 为 None，则设置为当前时间。
            - 若 prompt_token_ids 为 None，则设置为编码后的 prompt。
            - 创建 `best_of` 个 :class:`~vllm.Sequence` 对象。
            - 从 :class:`~vllm.Sequence` 列表创建
              :class:`~vllm.SequenceGroup` 对象。
            - 将 :class:`~vllm.SequenceGroup` 对象添加到调度器。

        Example:
            >>> # 初始化引擎
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> # 设置请求参数
            >>> example_prompt = "Who is the president of the United States?"
            >>> sampling_params = SamplingParams(temperature=0.0)
            >>> request_id = 0
            >>>
            >>> # 将请求添加到引擎
            >>> engine.add_request(
            >>>    str(request_id),
            >>>    example_prompt,
            >>>    SamplingParams(temperature=0.0))
            >>> # 继续处理请求
            >>> ...
        """
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.monotonic()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # 创建序列。
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size, lora_request)

        # 检查输入是否指定了前缀
        prefix = self.scheduler.prefix_pool.add_or_get_prefix(prompt_token_ids[:prefix_pos], lora_request.lora_int_id if
                                                              lora_request else 0) if prefix_pos is not None else None

        # 创建序列组。
        seq_group = SequenceGroup(request_id, [seq], sampling_params, arrival_time, lora_request, prefix)

        # 将序列组添加到调度器。
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """中止具有给定 ID 的请求。

        Args:
            request_id: 要中止的请求 ID（可多个）。

        Details:
            - 参见 :class:`~vllm.core.scheduler.Scheduler` 类中的
              :meth:`~vllm.core.scheduler.Scheduler.abort_seq_group`。

        Example:
            >>> # 初始化引擎并添加一个带 request_id 的请求
            >>> request_id = str(0)
            >>> # 中止该请求
            >>> engine.abort_request(request_id)
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """获取模型配置。"""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """获取未完成请求的数量。"""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """如果存在未完成的请求则返回 True。"""
        return self.scheduler.has_unfinished_seqs()

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = (current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(current_worst_seq).eos_token_id))
        if early_stopping is False:
            highest_attainable_score = (best_running_seq.get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(best_running_seq).eos_token_id))
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # 若 length_penalty > 0.0，beam search 会偏好更长的
                # 序列。此时最高可达分数的计算基于最长的可能序列长度。
                max_possible_length = max(best_running_seq.get_prompt_len() + sampling_params.max_tokens,
                                          self.scheduler_config.max_model_len)
                highest_attainable_score = (best_running_seq.get_beam_search_score(
                    length_penalty=length_penalty,
                    eos_token_id=self.get_tokenizer_for_seq(best_running_seq).eos_token_id,
                    seq_len=max_possible_length))
            else:
                # 否则，beam search 会偏好更短的序列。最高可达分数的
                # 计算基于当前序列长度。
                highest_attainable_score = (best_running_seq.get_beam_search_score(
                    length_penalty=length_penalty,
                    eos_token_id=self.get_tokenizer_for_seq(best_running_seq).eos_token_id))

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup, outputs: SequenceGroupOutput) -> None:

        # 处理 prompt 的 logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            seq_group.prompt_logprobs = prompt_logprobs

        # 处理采样结果
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict = {parent_seq.seq_id: [] for parent_seq in parent_seqs}
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # 列表元素为 (子序列, 父序列)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # 为每个父序列处理其子采样结果
        for parent in parent_seqs:
            child_samples: List[SequenceOutput] = parent_child_dict[parent.seq_id]
            if len(child_samples) == 0:
                # 该父序列没有子采样结果。由于它在后续迭代中
                # 不会再被使用，将其从序列组中移除。
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue
            # 如果有多个子采样结果，则对父序列进行分叉（fork）。
            for child_sample in child_samples[:-1]:
                new_child_seq_id = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token, child_sample.logprobs)
                child_seqs.append((child, parent))
            # 对最后一个子采样结果继续使用父序列。
            # 这里复用父序列以减少冗余的内存拷贝，尤其是使用
            # 非 beam search 采样方法时。
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token, last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            # self._decode_sequence(seq, seq_group.sampling_params)
            self._check_stop(seq, seq_group.sampling_params)

        # 非 beam search 的情况
        if not seq_group.sampling_params.use_beam_search:
            # 对于新建的子序列，将其添加到序列组中，
            # 如果尚未完成，则在块管理器（block manager）中分叉它们。
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        self.scheduler.fork_seq(parent, seq)

            # 在块管理器中释放已完成和被选中的父序列的内存，
            # 并将其保留在序列组中作为候选输出。
            # NOTE: 我们需要在释放旧序列之前先分叉新序列。
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_seq(seq)
            return

        # beam search 的情况
        # 选择要保留在序列组中的子序列。
        selected_child_seqs = []
        unselected_child_seqs = []
        beam_width = seq_group.sampling_params.best_of
        length_penalty = seq_group.sampling_params.length_penalty

        # 选择得分最高的新完成序列，以替换现有的已完成序列。
        # 元组为 (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False) for seq in existing_finished_seqs]
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs if seq.is_finished()]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # 按得分对已完成的序列排序。
        all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(x[0]).eos_token_id),
                               reverse=True)
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # 新生成的子序列已完成且得分较高，
                # 因此将其加入序列组。
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # 新生成的子序列已完成但得分较低，因此不将其加入
                # 序列组。此外，如果该序列是某个父序列的延续，
                # 还需要从序列组中移除该父序列。
                unselected_child_seqs.append((seq, parent))
            else:
                # 已有的完成序列得分较低，因此将其从序列组中移除。
                seq_group.remove(seq.seq_id)

        # 从运行中的序列里选出前 beam_width 个序列，
        # 供下一次迭代继续 beam search。
        running_child_seqs = [(seq, parent) for seq, parent in child_seqs if not seq.is_finished()]
        # 按得分对运行中的序列排序。
        running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(x[0]).eos_token_id),
                                reverse=True)

        # 检查是否可以停止 beam search。
        if len(running_child_seqs) == 0:
            # 没有运行中的序列，停止 beam search。
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # 已完成的序列不足，继续 beam search。
            stop_beam_search = False
        else:
            # 检查提前停止的判定条件
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(seq_group.sampling_params.early_stopping,
                                                                      seq_group.sampling_params, best_running_seq,
                                                                      current_worst_seq)

        if stop_beam_search:
            # 停止 beam search，并将所有运行中的序列从序列组中移除。
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # 继续 beam search，选出前 beam_width 个序列
            # 以继续搜索。
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # 剩余的运行序列在下次迭代中不会再被使用。同样，
            # 如果这些序列是某个父序列的延续，则需要从序列组中
            # 移除对应的父序列。
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # 对于新建的子序列，将其添加到序列组中，
        # 如果尚未完成，则在块管理器中分叉它们。
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)

        # 在块管理器中释放已完成和被选中的父序列的内存，
        # 并将其保留在序列组中作为候选输出。
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)

        # 将未被选中的父序列从序列组中移除，
        # 并在块管理器中释放其内存。
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # 如果父序列未被选中进入下一次迭代，则将其移除
                seq_group.remove(seq.seq_id)
                self.scheduler.free_seq(seq)

    def _process_model_outputs(self, output: SamplerOutput, scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        # 用模型输出更新已调度的序列组。
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_sequence_group_outputs(seq_group, outputs)

        # 释放已完成的序列组。
        self.scheduler.free_finished_seq_groups()

        # 创建输出。
        request_outputs: List[RequestOutput] = []
        for seq_group in scheduled_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        for seq_group in scheduler_outputs.ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        # 更新前缀状态，现在所有未计算的前缀都已计算完成。
        for seq_group in scheduled_seq_groups:
            if (seq_group.prefix is not None and seq_group.prefix.allocated and not seq_group.prefix.computed):
                seq_group.prefix.computed = True

        # 记录统计信息。
        if self.log_stats:
            self.stat_logger.log(self._get_stats(scheduler_outputs))

        return request_outputs

    def step(self) -> List[RequestOutput]:
        """执行一次解码迭代，返回新生成的结果。

        该函数执行引擎的一次解码迭代。它首先调度下一次迭代要执行的
        序列以及要换入/换出/拷贝的 token 块，然后执行模型并根据模型
        输出更新调度器。最后，对序列进行解码并返回新生成的结果。
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        if not scheduler_outputs.is_empty():
            output = self.worker.execute_model(
                        seq_group_metadata_list=seq_group_metadata_list, # TODO: 检查此输入
                        blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                        blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                        blocks_to_copy=scheduler_outputs.blocks_to_copy,)
        else:
            return [RequestOutput.from_seq_group(seq_group) for seq_group in scheduler_outputs.ignored_seq_groups]

        return self._process_model_outputs(output, scheduler_outputs)

    def do_log_stats(self) -> None:
        """无活动请求时强制记录日志。"""
        if self.log_stats:
            self.stat_logger.log(self._get_stats(scheduler_outputs=None))

    def _get_stats(self, scheduler_outputs: Optional[SchedulerOutputs]) -> Stats:
        """获取要记录到 Prometheus 的统计数据。"""
        now = time.monotonic()

        # KV cache 使用率（百分比）。
        num_total_gpu = self.cache_config.num_gpu_blocks
        num_free_gpu = self.scheduler.block_manager.get_num_free_gpu_blocks()
        gpu_cache_usage = 1.0 - (num_free_gpu / num_total_gpu)

        num_total_cpu = self.cache_config.num_cpu_blocks
        cpu_cache_usage = 0.
        if num_total_cpu > 0:
            num_free_cpu = self.scheduler.block_manager.get_num_free_cpu_blocks()
            cpu_cache_usage = 1.0 - (num_free_cpu / num_total_cpu)

        # 调度器状态
        num_running = len(self.scheduler.running)
        num_swapped = len(self.scheduler.swapped)
        num_waiting = len(self.scheduler.waiting)

        # 如果有调度器输出，则为迭代统计数据。
        num_prompt_tokens = 0
        num_generation_tokens = 0
        time_to_first_tokens = []
        time_per_output_tokens = []
        time_e2e_requests = []
        if scheduler_outputs is not None:
            prompt_run = scheduler_outputs.prompt_run

            # token 数量。
            if prompt_run:
                num_prompt_tokens = scheduler_outputs.num_batched_tokens
            else:
                num_generation_tokens = scheduler_outputs.num_batched_tokens

            # 延迟计时。
            time_last_iters = []
            for seq_group in scheduler_outputs.scheduled_seq_groups:
                # 距上一个 token 的时间。（注意：会更新 seq_group.last_token_time）
                time_last_iters.append(seq_group.get_last_latency(now))
                # 所有已完成请求自到达以来的时间。
                if seq_group.is_finished():
                    time_e2e_requests.append(now - seq_group.arrival_time)

            time_to_first_tokens = time_last_iters if prompt_run else []
            time_per_output_tokens = [] if prompt_run else time_last_iters

        return Stats(
            now=now,
            num_running=num_running,
            num_swapped=num_swapped,
            num_waiting=num_waiting,
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=num_generation_tokens,
            time_to_first_tokens=time_to_first_tokens,
            time_per_output_tokens=time_per_output_tokens,
            time_e2e_requests=time_e2e_requests,
        )

    # TODO: 我们可能不需要解码
    def _decode_sequence(self, seq: Sequence, prms: SamplingParams) -> None:
        """对序列的新 token 进行解码。"""
        (new_tokens, new_output_text, prefix_offset, read_offset) = detokenize_incrementally(
            self.get_tokenizer_for_seq(seq),
            all_input_ids=seq.get_token_ids(),
            prev_tokens=seq.tokens,
            prefix_offset=seq.prefix_offset,
            read_offset=seq.read_offset,
            skip_special_tokens=prms.skip_special_tokens,
            spaces_between_special_tokens=prms.spaces_between_special_tokens,
        )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text

    def _check_stop(self, seq: Sequence, sampling_params: SamplingParams) -> None:
        """停止已完成的序列。"""
        # for stop_str in sampling_params.stop:
        #     if seq.output_text.endswith(stop_str):
        #         self._finalize_sequence(seq, sampling_params, stop_str)
        #         seq.status = SequenceStatus.FINISHED_STOPPED
        #         return
        # if seq.get_last_token_id() in sampling_params.stop_token_ids:
        #     stop_str = self.get_tokenizer_for_seq(seq).convert_ids_to_tokens(seq.get_last_token_id())
        #     self._finalize_sequence(seq, sampling_params, stop_str)
        #     seq.status = SequenceStatus.FINISHED_STOPPED
        #     return

        # 检查序列是否已达到 max_model_len。
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # 检查序列是否已达到 max_tokens。
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # 检查序列是否已生成 EOS token。
        if ((not sampling_params.ignore_eos) and
                seq.get_last_token_id() == self.get_tokenizer_for_seq(seq).eos_token_id):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    def _finalize_sequence(self, seq: Sequence, sampling_params: SamplingParams, stop_string: str) -> None:
        if not sampling_params.include_stop_str_in_output and stop_string:
            # 截断输出文本，使停止字符串不包含在输出中。
            seq.output_text = seq.output_text[:-len(stop_string)]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.worker.remove_lora(lora_id)

    def list_loras(self) -> List[int]:
        return self.worker.list_loras()

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor]) -> None:
        self.worker.sync_model_weights(actor_weights=actor_weights)

    def offload_model_weights(self) -> None:
        self.worker.offload_model_weights()


def initialize_cluster(
    parallel_config: ParallelConfig,
    engine_use_ray: bool = False,
    ray_address: Optional[str] = None,
) -> Tuple[str, Optional[None]]:
    """初始化分布式集群（可能使用 Ray）。

    Args:
        parallel_config: 并行执行的配置。
        engine_use_ray: 异步引擎是否使用 Ray。
        ray_address: Ray 集群的地址。若为 None，则使用默认的
            Ray 集群地址。

    Returns:
        由 (`distributed_init_method`, `placement_group`) 组成的元组。
        `distributed_init_method` 是用于初始化分布式后端的地址。
        `placement_group` 包含每个分布式 worker 的资源规格。
    """

    # 在本地初始化集群。
    port = get_open_port()
    # 我们需要设置分布式初始化方法，以确保分布式 megatron 代码
    # （例如获取 world size）能够正常工作。
    distributed_init_method = f"tcp://localhost:{port}"
    return distributed_init_method, None


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
