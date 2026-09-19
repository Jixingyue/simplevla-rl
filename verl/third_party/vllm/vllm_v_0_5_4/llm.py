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
# 改编自 https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

from contextlib import contextmanager
from typing import ClassVar, List, Optional, Sequence, Union, cast, overload, Dict, Tuple

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import PretrainedConfig
import torch.nn as nn
from .arg_utils import EngineArgs
from .llm_engine_sp import LLMEngine
from vllm import LLM
from vllm.inputs import (PromptInputs, TextPrompt, TokensPrompt, parse_and_batch_prompt)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (GuidedDecodingRequest, get_local_guided_decoding_logits_processor)
from vllm.model_executor.guided_decoding.guided_fields import LLMGuidedOptions
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_cached_tokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, deprecate_kwargs
import torch
from torch.nn.utils.rnn import pad_sequence
from verl.workers.rollout.tokenizer import HybridEngineBaseTokenizer


class LLM(LLM):
    """用于根据给定的 prompt 和采样参数生成文本的大语言模型（LLM）。

    该类包含一个 tokenizer、一个语言模型（可能通过张量并行分布在
    多个 GPU 上），以及为中间状态（即 KV cache）分配的 GPU 显存空间。
    给定一批 prompt 和采样参数，该类使用智能批处理机制和高效的内存
    管理从模型生成文本。

    NOTE: 该类用于离线推理。若需在线 serving，请改用 `AsyncLLMEngine` 类。
    NOTE: 完整的参数列表请参见 `EngineArgs`。

    Args:
        model: HuggingFace Transformers 模型实例。
        tokenizer: HuggingFace Transformers tokenizer 实例。
        tokenizer_mode: tokenizer 模式。"auto" 会在可用时使用快速
            tokenizer，"slow" 则始终使用慢速 tokenizer。
        trust_remote_code: 下载模型和 tokenizer 时是否信任远程代码
            （例如来自 HuggingFace 的代码）。
        tensor_parallel_size: 使用张量并行进行分布式执行时所用的
            GPU 数量。
        dtype: 模型权重和激活值的数据类型。目前支持 `float32`、
            `float16` 和 `bfloat16`。若为 `auto`，则使用模型配置文件中
            指定的 `torch_dtype` 属性；但如果配置中的 `torch_dtype` 为
            `float32`，我们将改用 `float16`。
        quantization: 模型权重的量化方法。目前支持 "awq"。若为 None，
            则假定模型权重未量化，并使用 `dtype` 决定权重的数据类型。
        revision: 要使用的具体模型版本。可以是分支名、标签名或
            commit id。
        tokenizer_revision: 要使用的具体 tokenizer 版本。可以是分支名、
            标签名或 commit id。
        seed: 用于初始化采样随机数生成器的种子。
        gpu_memory_utilization: 为模型权重、激活值和 KV cache 预留的
            GPU 显存比例（0 到 1 之间）。该值越高，KV cache 越大，
            模型吞吐量越高。但若该值过高，可能导致显存溢出（OOM）
            错误。
        swap_space: 每个 GPU 用作交换空间的 CPU 内存大小（GiB）。
            当请求的 `best_of` 采样参数大于 1 时，可用于临时存放请求
            状态。若所有请求的 `best_of=1`，可安全地将其设为 0；
            否则，过小的值可能导致显存溢出（OOM）错误。
        enforce_eager: 是否强制 eager 执行。若为 True，将禁用
            CUDA graph，始终以 eager 模式执行模型；若为 False，将
            混合使用 CUDA graph 和 eager 执行。
        max_context_len_to_capture: CUDA graph 覆盖的最大上下文长度。
            当序列的上下文长度超过该值时，回退到 eager 模式。
        disable_custom_all_reduce: 参见 ParallelConfig
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict], # 模型本身或其参数字典
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer],
        model_hf_config: PretrainedConfig,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        skip_tokenizer_init: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
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
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            load_format=load_format,
            skip_tokenizer_init=skip_tokenizer_init,
            **kwargs,
        )
        tokenizer_cls = (PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer)
        if not isinstance(tokenizer, tokenizer_cls):
            raise ValueError(
                f"Unexpected tokenizer type: {type(tokenizer)}. Must be"
                "one of the following: PreTrainedTokenizer, PreTrainedTokenizerFast, verl.workers.rollout.HybridEngineBaseTokenizer"
            )
        self.llm_engine = LLMEngine.from_engine_args(model, tokenizer, engine_args)  # TODO: 检查 usagecontext
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

    def _run_engine(self, *, use_tqdm: bool) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # 初始化 tqdm。
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )
        # 运行引擎。
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # 仅为 RequestOutput 计算 token 数
                            total_in_toks += len(output.prompt_token_ids)
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = (f"est. speed input: {in_spd:.2f} toks/s, "
                                            f"output: {out_spd:.2f} toks/s")
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # 按 request ID 对输出排序。
        # 这是必要的，因为某些请求可能比它之前的请求更早完成。
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return self._post_process_outputs(outputs)

    # # NOTE(shengguangming): 为 verl 添加
    # # TODO(sgm): 可以通过让 dataloader 产出不带 padding 的 List[int] 来优化它。
    # def _pre_process_inputs(self, prompt_token_ids: torch.Tensor) -> List[int]:
    #     # 移除 prompt token_id 中的左侧 padding
    #     pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    #     non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    #     token_ids = prompt_token_ids[non_pad_index:].tolist()
    #     return token_ids

    # NOTE(shengguangming): 为 verl 添加
    def _post_process_outputs(self, request_outputs: List[RequestOutput]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_token_ids = []
        logprobs = []
        for request_output in request_outputs:  # List[RequestOutput]
            outputs = request_output.outputs
            for output in outputs:  # List[CompletionOutput], usually len == 1
                output_token_ids.append(torch.tensor(output.token_ids))
                # TODO(shengguangming): 可通过重写 Sampler._get_logprobs() 的 logits 来优化
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
