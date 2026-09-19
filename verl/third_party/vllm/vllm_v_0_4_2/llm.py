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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import PretrainedConfig
import torch.nn as nn
from .arg_utils import EngineArgs
from .llm_engine_sp import LLMEngine
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from verl.workers.rollout.tokenizer import HybridEngineBaseTokenizer


class LLM:
    """根据给定的 prompt 和采样参数生成文本的 LLM。

    这个类包含一个 tokenizer、一个语言模型（可能分布在
    多个 GPU 上），以及为中间状态（即 KV cache）分配的
    GPU 内存空间。给定一批 prompt 和采样参数，这个类会使用
    智能的批处理机制和高效的内存管理，从模型生成文本。

    NOTE: 这个类旨在用于离线推理。在线推理请改用
    `AsyncLLMEngine` 类。
    NOTE: 完整的参数列表请参见 `EngineArgs`。

    参数:
        model: HuggingFace Transformers 模型实例。
        tokenizer: HuggingFace Transformers tokenizer 实例。
        tokenizer_mode: Tokenizer 模式。"auto" 会使用 fast tokenizer
            （如果可用），"slow" 则始终使用 slow tokenizer。
        trust_remote_code: 下载模型和 tokenizer 时信任远程代码
            （例如来自 HuggingFace 的代码）。
        tensor_parallel_size: 使用张量并行进行分布式执行时
            所用的 GPU 数量。
        dtype: 模型权重和激活值的数据类型。目前，
            我们支持 `float32`、`float16` 和 `bfloat16`。如果为 `auto`，
            我们使用模型配置文件中指定的 `torch_dtype` 属性。
            但是，如果配置中的 `torch_dtype` 为 `float32`，我们将
            改用 `float16`。
        quantization: 用于量化模型权重的方法。目前，
            我们支持 "awq"。如果为 None，我们假设模型权重
            未被量化，并使用 `dtype` 来确定权重的数据类型。
        revision: 要使用的具体模型版本。可以是分支名、
            标签名或 commit id。
        tokenizer_revision: 要使用的具体 tokenizer 版本。可以是
            分支名、标签名或 commit id。
        seed: 用于初始化采样随机数生成器的种子。
        gpu_memory_utilization: 为模型权重、激活值和 KV cache 预留的
            GPU 内存比例（0 到 1 之间）。更高的
            值会增大 KV cache 容量，从而提高模型的
            吞吐量。但如果值过高，可能导致显存
            不足（OOM）错误。
        swap_space: 每个 GPU 用作交换空间的 CPU 内存大小（GiB）。
            当请求的 `best_of` 采样参数大于 1 时，
            可用于临时存储请求的状态。如果所有
            请求的 `best_of=1`，可以安全地将此值设为 0。
            否则，过小的值可能导致显存不足（OOM）错误。
        enforce_eager: 是否强制使用 eager 执行。如果为 True，我们将
            禁用 CUDA graph，并始终以 eager 模式执行模型。
            如果为 False，我们将混合使用 CUDA graph 和 eager 执行。
        max_context_len_to_capture: CUDA graph 覆盖的最大上下文长度。
            当序列的上下文长度超过此值时，我们回退到
            eager 模式。
        disable_custom_all_reduce: 参见 ParallelConfig
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict], # 模型本身或其参数字典
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer],
        model_hf_config: PretrainedConfig,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = None,
        disable_custom_all_reduce: bool = False,
        load_format = 'auto',
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            load_format=load_format,
            **kwargs,
        )
        tokenizer_cls = (PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer)
        if not isinstance(tokenizer, tokenizer_cls):
            raise ValueError(
                f"Unexpected tokenizer type: {type(tokenizer)}. Must be"
                "one of the following: PreTrainedTokenizer, PreTrainedTokenizerFast, verl.workers.rollout.HybridEngineBaseTokenizer"
            )
        self.llm_engine = LLMEngine.from_engine_args(model, tokenizer, engine_args)
        self.request_counter = Counter()

    def init_cache_engine(self):
        self.llm_engine.init_cache_engine()

    def free_cache_engine(self):
        self.llm_engine.free_cache_engine()

    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> List[RequestOutput]:
        """为输入的 prompt 生成补全结果。

        NOTE: 这个类会考虑内存限制，自动对给定的 prompt 进行
        批处理。为了获得最佳性能，请将所有 prompt
        放入一个列表中并传给此方法。

        参数:
            prompts: 用于生成补全结果的 prompt 列表。
            sampling_params: 文本生成所用的采样参数。如果
                为 None，我们使用默认的采样参数。
                当它是单个值时，会应用于每个 prompt。
                当它是列表时，列表长度必须与
                prompt 数量相同，并与 prompt 一一配对。
            prompt_token_ids: prompt 的 token ID 列表。如果为 None，我们
                使用 tokenizer 将 prompt 转换为 token ID。
            use_tqdm: 是否使用 tqdm 显示进度条。
            lora_request: 用于生成的 LoRA 请求（如果有）。
            multi_modal_data: 多模态数据。

        返回:
            一个 `RequestOutput` 对象列表，其中包含生成的
            补全结果，顺序与输入的 prompt 相同。
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if self.llm_engine.model_config.skip_tokenizer_init \
            and prompts is not None:
            raise ValueError("prompts must be None if skip_tokenizer_init "
                             "is True")
        if isinstance(prompts, str):
            # 将单个 prompt 转换为列表。
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")

        if prompts is not None:
            num_requests = len(prompts)
        else:
            assert prompt_token_ids is not None
            num_requests = len(prompt_token_ids)

        if sampling_params is None:
            # 使用默认的采样参数。
            sampling_params = SamplingParams()

        elif isinstance(sampling_params, list) and len(sampling_params) != num_requests:
            raise ValueError("The lengths of prompts and sampling_params "
                             "must be the same.")
        if multi_modal_data:
            multi_modal_data.data = multi_modal_data.data.to(torch.float16)

        # 将请求添加到引擎中。
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[i]
            if not isinstance(token_ids, list):
                # NOTE(shengguangming): 将 rollout 输入转换为 List[str]
                token_ids = self._pre_process_inputs(token_ids)
            self._add_request(
                prompt,
                sampling_params[i] if isinstance(sampling_params, list) else sampling_params,
                token_ids,
                lora_request=lora_request,
                # 在保持 batch 维度的同时获取第 i 张图像。
                multi_modal_data=MultiModalData(type=multi_modal_data.type, data=multi_modal_data.data[i].unsqueeze(0))
                if multi_modal_data else None,
            )
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids,
                                    lora_request=lora_request,
                                    multi_modal_data=multi_modal_data)

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # 初始化 tqdm。
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts", dynamic_ncols=True)
        # 运行引擎。
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # 按 request ID 对输出排序。
        # 这是必要的，因为有些请求可能比它前面的请求
        # 更早结束。
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        # TODO(shengguangming): 也许可以 hack 自回归逻辑，而不只是应用后处理，以获得更好的性能
        return self._post_process_outputs(outputs)

    # NOTE(shengguangming): 为 verl 添加
    # TODO(sgm): 我们可以通过让 dataloader 产出不带 padding 的 List[int] 来优化它。
    def _pre_process_inputs(self, prompt_token_ids: torch.Tensor) -> List[int]:
        # 移除 prompt token_id 中的左侧 padding
        pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
        non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
        token_ids = prompt_token_ids[non_pad_index:].tolist()
        return token_ids

    # NOTE(shengguangming): 为 verl 添加
    def _post_process_outputs(self, request_outputs: List[RequestOutput]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_token_ids = []
        logprobs = []
        for request_output in request_outputs:  # List[RequestOutput]
            outputs = request_output.outputs
            for output in outputs:  # List[CompletionOutput]，通常 len == 1
                output_token_ids.append(torch.tensor(output.token_ids))
                # TODO(shengguangming): 可以通过重写 Sampler._get_logprobs() 的 logits 来优化
                logprobs_dicts = output.logprobs
                if logprobs_dicts is not None:
                    logprob = []
                    for logprobs_dict, id in zip(logprobs_dicts, output.token_ids):
                        logprob.append(logprobs_dict[id].logprob)
                    logprobs.append(torch.tensor(logprob))

        pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
        output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=pad_token_id)
        if len(logprobs) > 0:
            logprobs = pad_sequence(logprobs, batch_first=True, padding_value=pad_token_id)
        return output_token_ids, logprobs

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        self.llm_engine.sync_model_weights(actor_weights=actor_weights, load_format=load_format)

    def offload_model_weights(self) -> None:
        self.llm_engine.offload_model_weights()
