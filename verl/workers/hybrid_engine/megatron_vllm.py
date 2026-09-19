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
本文件实现了一个 Megatron 风格的混合引擎（Hybrid Engine），将 actor 的权重共享给推理引擎。
"""

import torch
import torch.distributed as dist

from torch import nn

from megatron.core import parallel_state as mpu
from megatron.core import DistributedDataParallel as LocalDDP
from megatron.core.transformer.module import Float16Module
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from verl.utils.megatron_utils import get_model, unwrap_model
from verl.utils.memory_buffer import (
    build_memory_buffer,
    build_memory_reference_from_module,
    get_weight_buffer_meta_from_module,
)


class AllGatherPPModel:

    def __init__(self, model_provider) -> None:

        self._pp_group = mpu.get_pipeline_model_parallel_group()
        self._pp_rank = mpu.get_pipeline_model_parallel_rank()
        self._pp_size = mpu.get_pipeline_model_parallel_world_size()
        self._vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        self._model_chunk_size = self._vpp_size or 1

        # 每个元素保存当前 pp 阶段的 model_chunks 列表
        self._pp_models = [None] * self.pp_size

        rank_list = list(range(self.pp_size))
        # 让当前 rank 最后初始化
        rank_list[self.pp_rank], rank_list[-1] = rank_list[-1], rank_list[self.pp_rank]
        self._this_rank_models = None

        # 存储每个 pp 阶段的参数
        self.memory_buffers = [None] * self.pp_size
        for cur_pp_rank in rank_list:
            print(
                f'create pp model', f'torch allocated {torch.cuda.memory_allocated() / 1e9:.4f} GB, '
                f'reserved {torch.cuda.memory_reserved() / 1e9:.4f} GB')
            # 由于最后初始化的 rank 就是当前 pp rank，因此初始化完成后 pp rank 仍然正确
            mpu.set_pipeline_model_parallel_rank(cur_pp_rank)
            if cur_pp_rank != self.pp_rank:
                models = get_model(model_provider, wrap_with_ddp=False)
                models = nn.ModuleList(models)
                assert len(models) == self._model_chunk_size, f"{len(models)} != {self._model_chunk_size}"
                self.pp_models[cur_pp_rank] = models
            else:
                # 对于常规模型，我们用 DDP 对其进行包装
                models = get_model(model_provider)
                assert len(models) == self._model_chunk_size, f"{len(models)} != {self._model_chunk_size}"
                self._this_rank_models = nn.ModuleList(models)
                self.pp_models[cur_pp_rank] = nn.ModuleList(unwrap_model(models, (torchDDP, LocalDDP)))

            self._build_param_buffer(cur_pp_rank)
            self._build_param_references(cur_pp_rank, maintain_weight=cur_pp_rank == self.pp_rank)

            # TODO: 绑定到内存缓冲区之后，可以在这里加载 checkpoint
            if cur_pp_rank != self.pp_rank:
                for model in self.pp_models[cur_pp_rank]:
                    model.eval()
                self._offload_params_to_cpu(cur_pp_rank)

    def _build_param_buffer(self, pp_rank):
        """在每个 pp rank 上构建参数缓冲区"""
        model = self.pp_models[pp_rank]
        weight_buffer_meta = get_weight_buffer_meta_from_module(model)
        self.memory_buffers[pp_rank] = build_memory_buffer(weight_buffer_meta)

    def _build_param_references(self, pp_rank, maintain_weight=False):
        model = self.pp_models[pp_rank]
        build_memory_reference_from_module(model, self.memory_buffers[pp_rank], maintain_weight=maintain_weight)

    def _load_params_to_cuda(self, pp_rank, to_empty=False):
        assert pp_rank != self.pp_rank, f"unexpected to load current pp rank [{pp_rank}] back to cuda"
        for buffer in self.memory_buffers[pp_rank].values():
            if not to_empty:
                buffer.data = buffer.data.to(torch.cuda.current_device(), non_blocking=True)
            else:
                buffer.data = torch.empty_like(buffer.data, device='cuda')
        # 加载到 CUDA 后重建引用
        self._build_param_references(pp_rank)

    def _offload_params_to_cpu(self, pp_rank, to_empty=False):
        assert pp_rank != self.pp_rank, f"unexpected to offload current pp rank [{pp_rank}] to cpu"
        for buffer in self.memory_buffers[pp_rank].values():
            if not to_empty:
                # 将整个内存缓冲区 offload 到 CPU
                buffer.data = buffer.data.to('cpu', non_blocking=True)
            else:
                buffer.data = torch.empty_like(buffer.data, device='cpu')
        self._build_param_references(pp_rank)

    def load_params_to_cuda(self, to_empty=False):
        """将所有模型参数加载到 cuda"""
        for cur_pp_rank in range(self.pp_size):
            if cur_pp_rank != self.pp_rank:
                self._load_params_to_cuda(cur_pp_rank, to_empty=to_empty)

    def allgather_params(self):
        """对所有 pp rank 的参数执行 allgather。返回句柄列表"""
        for cur_pp_rank in range(self.pp_size):
            global_src = dist.get_global_rank(group=self.pp_group, group_rank=cur_pp_rank)

            # NOTE(sgm): 异步操作可能导致 memory_buffer/pp_models 发生内存泄漏
            for memory_buffer in self.memory_buffers[cur_pp_rank].values():
                dist.broadcast(tensor=memory_buffer.data, src=global_src, group=self.pp_group, async_op=False)

    def forward(self, *inputs, **kwargs):
        try:
            prev_output = None
            for cur_chunk_rank in range(self._model_chunk_size):
                if self._vpp_size:
                    mpu.set_virtual_pipeline_model_parallel_rank(cur_chunk_rank)

                for cur_pp_rank in range(self.pp_size):
                    mpu.set_pipeline_model_parallel_rank(cur_pp_rank)
                    self.pp_models[cur_pp_rank][cur_chunk_rank].set_input_tensor(prev_output)
                    ret = self.pp_models[cur_pp_rank][cur_chunk_rank](*inputs, **kwargs)
                    self.pp_models[cur_pp_rank][cur_chunk_rank].set_input_tensor(None)
                    prev_output = ret
        finally:
            if self._vpp_size:
                mpu.set_virtual_pipeline_model_parallel_rank(0)
            mpu.set_pipeline_model_parallel_rank(self.pp_rank)
        return ret

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def eval(self):
        for model in self.pp_models[self.pp_rank]:
            model.eval()

    def train(self):
        for model in self.pp_models[self.pp_rank]:
            model.train()

    def offload_params_to_cpu(self, to_empty=False):
        """将不属于当前 pp rank 的模型参数 offload 到 cpu"""
        for cur_pp_rank in range(self.pp_size):
            if cur_pp_rank != self.pp_rank:
                self._offload_params_to_cpu(cur_pp_rank, to_empty=to_empty)

    def get_all_params(self):
        """获取所有 pp rank 上模型的全部参数

        Returns:
            params: List[List[Dict[str, Tensor]]]: 所有 pp 中参数组成的列表，其中每个元素是
                一个 dict 列表，对应每个 model chunk 的张量字典

        """
        params = []
        for pp_rank in range(self.pp_size):
            params.append([])
            for model_chunk_idx in range(len(self.pp_models[pp_rank])):
                params[pp_rank].append({})
                pp_model = self.pp_models[pp_rank][model_chunk_idx]
                pp_model = unwrap_model(pp_model, ((torchDDP, LocalDDP, Float16Module)))  # 不使用 Float16Module
                for name, param in pp_model.named_parameters():
                    # NOTE(gh) 变通方案: 推理时不应获取 lora 参数
                    if 'lora' in name:
                        continue
                    params[pp_rank][model_chunk_idx][name] = param

        return params

    def update_this_rank_models(self, new_models):
        self._this_rank_models = new_models
        self._pp_models[self.pp_rank] = unwrap_model(new_models, (torchDDP, LocalDDP))

    @property
    def this_rank_models(self):
        return self._this_rank_models

    @property
    def pp_size(self):
        return self._pp_size

    @property
    def pp_rank(self):
        return self._pp_rank

    @property
    def pp_group(self):
        return self._pp_group

    @property
    def pp_models(self):
        return self._pp_models


"""
Megatron 混合引擎：
- 训练期间，只有当前 pp 阶段持有参数
- 推理之前，将当前 pp rank 的参数广播给所有其他 pp rank（所有 pp rank 持有全部参数）
- 将参数绑定到推理引擎
- 在 tp 维度上做推理，pp 被视作额外的 dp
- 推理结束后，释放所有不属于当前 pp rank 的参数
"""

from .base import BaseShardingManager

import torch
from torch import nn
import torch.distributed
from torch.distributed import new_group

from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
import verl.utils.megatron.tensor_parallel as tp_utils
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.third_party.vllm import LLM
from verl.utils.model import normalize_pp_vpp_params
# 微数据并行组。微数据并行组是额外的 dp 组，源自将训练 tp 拆分为 infer_tp 和 micro_tp。
# 默认情况下，我们采用 micro_dp - tp 的顺序
_MICRO_DATA_PARALLEL_GROUP = None


class MegatronVLLMShardingManager(BaseShardingManager):

    def __init__(self, module: AllGatherPPModel, inference_engine: LLM, model_config, layer_name_mapping):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.layer_name_mapping = layer_name_mapping

        # 为 vllm 推理初始化 micro_dp 组
        global _MICRO_DATA_PARALLEL_GROUP
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        train_tensor_parallel_size = mpu.get_tensor_model_parallel_world_size()
        infer_tensor_parallel_size = vllm_ps.get_tensor_model_parallel_world_size()

        # TODO(sgm): 对于 FSDP -> vLLM 的情况，这一点可能并不成立
        assert infer_tensor_parallel_size <= train_tensor_parallel_size, \
            'Not implemented for infer_tp > train_tp'
        assert train_tensor_parallel_size % infer_tensor_parallel_size == 0

        micro_dp_size = train_tensor_parallel_size // infer_tensor_parallel_size
        num_micro_dp_groups = world_size // micro_dp_size
        assert _MICRO_DATA_PARALLEL_GROUP is None, ("micro data parallel group is already initialized")
        for i in range(num_micro_dp_groups):
            ranks = range(i * micro_dp_size, (i + 1) * micro_dp_size)
            group = new_group(ranks=ranks)
            if rank in ranks:
                _MICRO_DATA_PARALLEL_GROUP = group

    def default_tp_concat_fn(self, name, param, infer_params, model_config):
        """
        name: 参数名
        param: 训练参数
        infer_params (List[torch.Tensor]): 从 micro_dp_group all-gather 得到的参数列表
        model_config: huggingface 的 model_config
        TODO(zhangchi.usc1992): 当前实现是针对特定场景临时写的。我们可以把这个函数移到模型
        定义中，使其与具体模型无关。如果模型没有实现该函数，
        我们可以抛出错误，强制用户禁用 TP HybridEngine。
        """

        if self.layer_name_mapping.get("qkv_layer_name") in name:
            # 如果张量是 qkv，则对 tp 上的每个参数按 q、k、v 拆分
            # 再分别对 q、k、v 进行拼接。
            q_lst = []
            k_lst = []
            v_lst = []
            assert model_config.num_attention_heads % model_config.num_key_value_heads == 0
            num_q_per_kv = model_config.num_attention_heads // model_config.num_key_value_heads
            assert infer_params[0].shape[0] % (num_q_per_kv + 2) == 0
            kv_size_per_tp = infer_params[0].shape[0] // (num_q_per_kv + 2)
            split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]
            for infer_param in infer_params:
                q, k, v = infer_param.split(split_size)
                q_lst.append(q)
                k_lst.append(k)
                v_lst.append(v)
            q = torch.cat(q_lst, dim=0)
            k = torch.cat(k_lst, dim=0)
            v = torch.cat(v_lst, dim=0)

            infer_params = torch.cat((q, k, v), dim=0)

        elif self.layer_name_mapping.get("gate_proj_layer_name") in name:
            # 如果张量是 gate 和 proj
            gate_lst = []
            up_lst = []
            for infer_param in infer_params:
                gate, up = infer_param.chunk(2)
                gate_lst.append(gate)
                up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)
            infer_params = torch.cat((gate, up), dim=0)

        else:
            # 拼接张量
            infer_params = torch.cat(infer_params, dim=tp_utils.get_tensor_parallel_partition_dim(param))

        return infer_params

    def _post_process_params(self, params):
        """
        对每个参数，如果它是被 tp 拆分过的参数，则从 micro_dp 组执行 all-gather。
        """
        # 这里的参数是训练 tp 格式。我们遍历参数并执行 all-gather
        # TODO(zhangchi.usc1992) 可以考虑把非 tp 权重复制到另一个推理缓冲区。
        # 这样，原始 memory_buffers 中的所有参数都可以被 offload。
        micro_dp_size = get_micro_data_parallel_world_size()
        micro_dp_group = get_micro_data_parallel_group()

        if micro_dp_size <= 1:
            return

        origin_params = {}
        for name in params.keys():
            param = params[name]
            if tp_utils.is_tensor_parallel_param(param):
                # 分配一个大小合适的新张量
                infer_params = [torch.empty_like(param) for _ in range(micro_dp_size)]
                torch.distributed.all_gather(infer_params, param, group=micro_dp_group)
                infer_params = self.default_tp_concat_fn(name, param, infer_params, self.model_config)
                # 替换回原始参数
                params[name] = infer_params
            origin_params[name] = param

        return origin_params

    def __enter__(self):
        # 为不属于当前 pp rank 的参数创建新的 cuda 空间
        self.module.load_params_to_cuda()
        # 将参数从当前 pp rank 广播给其他 rank
        self.module.allgather_params()
        # 获取 pp/vpp 中参数名到参数的映射
        params = self.module.get_all_params()

        # 将参数绑定到推理引擎
        self.params = normalize_pp_vpp_params(params=params,
                                              num_hidden_layers=self.model_config.num_hidden_layers,
                                              layer_name='layers')
        self.origin_params = self._post_process_params(self.params)
        self.inference_engine.sync_model_weights(self.params, load_format='megatron')

    def __exit__(self, exc_type, exc_value, traceback):
        # offload 不属于当前 pp rank 的参数
        self.module.offload_params_to_cpu()

        # FIXME(sgm): 最佳做法是删除 cuda 张量
        # 重新绑定模型权重，可以是任意的 cpu 张量
        if get_micro_data_parallel_world_size() > 1:
            for name in self.params.keys():
                self.params[name] = self.origin_params[name]

        # self.inference_engine.sync_model_weights(params)
        self.inference_engine.offload_model_weights()

        self.module.train()

        # 每次计算后添加空缓存清理
        torch.cuda.empty_cache()

    def preprocess_data(self, data: DataProto) -> DataProto:
        # 每个训练 tp 上的 prompts 是相同的。我们为每个推理 tp 选取对应分片
        micro_dp_size = get_micro_data_parallel_world_size()
        micro_dp_rank = get_micro_data_parallel_rank()

        # 从 tp=0 广播到其他 tp rank
        broadcast_dict_tensor(data.batch,
                              src=mpu.get_tensor_model_parallel_src_rank(),
                              group=mpu.get_tensor_model_parallel_group())

        if micro_dp_size > 1:
            local_prompts = data.chunk(chunks=micro_dp_size)
            data = local_prompts[micro_dp_rank]

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        meta_info = data.meta_info
        # 在 micro-dp 组内对 batch 执行 all gather
        micro_dp_size = get_micro_data_parallel_world_size()
        if micro_dp_size > 1:
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                size=get_micro_data_parallel_world_size(),
                                                group=get_micro_data_parallel_group(),
                                                dim=0)

        # 在 pp 组内对 batch 执行 all gather
        if meta_info.get('allgather_pp_output', True):
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                size=mpu.get_pipeline_model_parallel_world_size(),
                                                group=mpu.get_pipeline_model_parallel_group(),
                                                dim=0)
        return data


"""
微数据并行组
"""


def get_micro_data_parallel_group():
    assert _MICRO_DATA_PARALLEL_GROUP is not None
    return _MICRO_DATA_PARALLEL_GROUP


def get_micro_data_parallel_world_size():
    return torch.distributed.get_world_size(group=get_micro_data_parallel_group())


def get_micro_data_parallel_rank():
    return torch.distributed.get_rank(group=get_micro_data_parallel_group())
