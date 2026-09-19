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
# 改编自 https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader

from typing import Dict

import torch.nn as nn
from torch.distributed._tensor import DTensor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import is_pp_missing_parameter


def gemma_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        for param_name, shard_name, shard_id in stacked_params_mapping:
            if shard_name not in name:
                continue
            stacked_name = name.replace(shard_name, param_name)
            # 跳过 GPTQ 模型的额外 bias 加载。
            if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[stacked_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # lm_head 在 vllm 中不使用，因为它与 embed_token 绑定（tied）。
            # 为避免报错，跳过加载 lm_head.weight。
            if "lm_head.weight" in name:
                continue
            # 跳过 GPTQ 模型的额外 bias 加载。
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def gptbigcode_dtensor_load_weights(actor_weights: Dict, vllm_model: nn.Module):
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "lm_head.weight" in name:
            continue
        if ".attn.bias" in name:
            # 跳过 attention mask。
            # NOTE: "c_attn.bias" 不应被跳过。
            continue
        local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def starcoder2_dtensor_load_weights(actor_weights: Dict, vllm_model: nn.Module):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]

    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def llama_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            # 使用 ColossalAI 训练的模型可能在 checkpoint 中包含这些张量。
            # 跳过它们。
            continue
        # 使用 tie_word_embeddings 时，可以跳过 lm_head.weight。
        # 若模型经过量化、LoRA、微调等处理，该权重可能会不必要地出现在文件中。
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # 跳过 GPTQ 模型的额外 bias 加载。
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # 跳过 GPTQ 模型的额外 bias 加载。
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight)


def qwen2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # 跳过 GPTQ 模型的额外 bias 加载。
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # 跳过 GPTQ 模型的额外 bias 加载。
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def qwen2vl_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # 跳过 GPTQ 模型的额外 bias 加载。
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # 跳过 GPTQ 模型的额外 bias 加载。
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


from vllm.model_executor.layers.fused_moe import FusedMoE


def deepseekv2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # 权重、fp8 权重缩放因子、fp8 激活值缩放因子的参数
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=vllm_model.config.n_routed_experts,
    )

    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            # 跳过非堆叠层和 experts（experts 在下方处理）。
            if weight_name not in name:
                continue
            # checkpoint 中存在 mlp.experts[0].gate_proj。
            # 由于我们在下方通过 expert_params_mapping 处理 experts，
            # 需要在更新 name 之前在这里跳过，否则 name 会被更新为
            # mlp.experts[0].gate_up_proj，随后在下方 expert_params_mapping
            # 中又被更新为 mlp.experts[0].gate_gate_up_proj，导致加载失败。
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            # 跳过 GPTQ 模型的额外 bias 加载。
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, vllm_model):
                continue

            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, vllm_model):
                    continue

                param = params_dict[name]
                local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    local_loaded_weight.to(dtype=param.dtype),
                    weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
            else:
                # 跳过 GPTQ 模型的额外 bias 加载。
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, vllm_model):
                    continue

                param = params_dict[name]
                local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def gpt2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    pass


def redistribute_dtensor(param_name: str, loaded_weights: DTensor, parallelize_plan: Dict = None):
    param_name = _process_parameter_names(name=param_name)
    if parallelize_plan is not None:
        assert (
            param_name
            in parallelize_plan.keys()), f"param name: {param_name} not in parallelize_plan :{parallelize_plan.keys()}"
        placement = parallelize_plan[param_name]
        local_loaded_weights = loaded_weights.redistribute(device_mesh=loaded_weights.device_mesh,
                                                           placements=placement).to_local()
    else:
        local_loaded_weights = loaded_weights.full_tensor()
    return local_loaded_weights


def _process_parameter_names(name):
    # 如果字符串末尾存在 '.weight'，则移除
    if name.endswith(".weight"):
        name = name[:-7]

    # 移除 'model.layers.x.' 或 'model.' 前缀
    if "model.layers" in name:
        parts = name.split(".")
        # 重建不含 'model.layers.x.' 的字符串
        name = ".".join(parts[3:])  # parts[0] 是 'model'，parts[1] 是 'layers'，parts[2] 是 'x'
    elif name.startswith("model."):
        name = name[6:]  # 移除 'model.'

    return name


__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__ = {
    "GPT2LMHeadModel": gpt2_dtensor_weight_loader,
    "LlamaForCausalLM": llama_dtensor_weight_loader,
    "LLaMAForCausalLM": llama_dtensor_weight_loader,
    "MistralForCausalLM": llama_dtensor_weight_loader,  # 在 vLLM 中 mistral 与 llama 相同
    "InternLMForCausalLM": llama_dtensor_weight_loader,
    "AquilaModel": llama_dtensor_weight_loader,
    "AquilaForCausalLM": llama_dtensor_weight_loader,
    "Phi3ForCausalLM": llama_dtensor_weight_loader,
    "GemmaForCausalLM": gemma_dtensor_weight_loader,
    "Gemma2ForCausalLM": gemma_dtensor_weight_loader,
    "GPTBigCodeForCausalLM": gptbigcode_dtensor_load_weights,
    "Starcoder2ForCausalLM": starcoder2_dtensor_load_weights,
    "Qwen2ForCausalLM": qwen2_dtensor_weight_loader,
    "DeepseekV2ForCausalLM": deepseekv2_dtensor_weight_loader,
    "Qwen2VLForConditionalGeneration": qwen2vl_dtensor_weight_loader,
}


# actor 模型即 .state_dict()
# 加载 dtensor 权重
def load_dtensor_weights(actor_weights: Dict, vllm_model: nn.Module):
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    weight_loader(actor_weights, vllm_model)
    # NOTE(sgm) 为降低峰值内存占用，初始化后我们将 vllm 模型卸载到 CPU，
    # 在第一次迭代同步模型权重后需要用到它。
    vllm_model = vllm_model.cuda()


def _get_model_weight_loader(arch: str):
    if arch in __MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__:
        return __MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__.keys()}")


# NOTE(sgm): 我们在每个 vllm 子模块中使用按参数的 weight loader
def update_dtensor_weight_loader():
    pass
