# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""模型并行和数据并行分组。"""

import torch
import torch.distributed

import vllm.model_executor.parallel_utils.parallel_state as ps
"""
本版本与 Megatron 强耦合，用于实现 HybridEngine 以及 vllm 与 Megatron 之间的权重共享。
- 我们假设在调用该函数之前 Megatron 的 tp+dp+pp 世界已经建立。

"""

# 当前 rank 所属的张量模型并行组。
_TENSOR_MODEL_PARALLEL_GROUP = None

# Micro 数据并行组。Micro data parallel group 是一个额外的 dp 组，源自将训练 tp
# 切分为 infer_tp 和 micro_tp。默认情况下，我们使用 micro_dp - tp 的顺序
_MICRO_DATA_PARALLEL_GROUP = None


def initialize_model_parallel_from_megatron(
        tensor_model_parallel_size=None  # 为向后兼容我们将它设为 None，即设置 infer_tp = train_tp
) -> None:
    from megatron.core import parallel_state as mpu
    from megatron.distributed import new_group
    # 获取 world size 和 rank，并确保一些一致性。
    assert torch.distributed.is_initialized()

    if tensor_model_parallel_size is None:
        tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
    else:
        assert isinstance(tensor_model_parallel_size, int)

    # 构建张量模型并行组。
    assert ps._TENSOR_MODEL_PARALLEL_GROUP is None, ("tensor model parallel group is already initialized")

    assert tensor_model_parallel_size <= mpu.get_tensor_model_parallel_world_size(
    ), 'Not implemented for infer_tp > train_tp'

    global _TENSOR_MODEL_PARALLEL_GROUP
    global _MICRO_DATA_PARALLEL_GROUP

    assert mpu.get_tensor_model_parallel_world_size() % tensor_model_parallel_size == 0

    micro_dp_size = mpu.get_tensor_model_parallel_world_size() // tensor_model_parallel_size

    world_size: int = torch.distributed.get_world_size()

    num_micro_dp_groups = world_size // micro_dp_size

    rank = torch.distributed.get_rank()

    # 构建 micro dp 组。
    assert _MICRO_DATA_PARALLEL_GROUP is None, ("micro data parallel group is already initialized")
    for i in range(num_micro_dp_groups):
        ranks = range(i * micro_dp_size, (i + 1) * micro_dp_size)
        group = new_group(rank=rank, ranks=ranks, group_type='micro_dp')
        if rank in ranks:
            _MICRO_DATA_PARALLEL_GROUP = group

    if tensor_model_parallel_size == mpu.get_tensor_model_parallel_world_size():
        # 使用与 Megatron 相同的 tp 组
        ps._TENSOR_MODEL_PARALLEL_GROUP = mpu.get_tensor_model_parallel_group()

        _TENSOR_MODEL_PARALLEL_GROUP = mpu.get_tensor_model_parallel_group()
        # 没有 _MICRO_DATA_PARALLEL_GROUP
    else:
        # 初始化一个 micro_dp 组和一个 tp 组
        # 假设训练 tp=4，推理 tp=2，那么权重划分方式为
        # 训练时 [1], [2], [3], [4]，推理时 [1,2], [1,2], [3,4], [3,4]

        # 构建推理 tp 组
        train_tp = mpu.get_tensor_model_parallel_world_size()
        num_tensor_model_parallel_groups_per_train_tp = train_tp // tensor_model_parallel_size
        num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
        assert _TENSOR_MODEL_PARALLEL_GROUP is None, ("tensor model parallel group is already initialized")
        for i in range(num_tensor_model_parallel_groups // num_tensor_model_parallel_groups_per_train_tp):
            start = train_tp * i
            end = train_tp * (i + 1)
            for j in range(num_tensor_model_parallel_groups_per_train_tp):
                ranks = list(range(start, end, num_tensor_model_parallel_groups_per_train_tp))
                for i in range(len(ranks)):
                    ranks[i] += j
                # group = torch.distributed.new_group(ranks)
                group = new_group(rank=rank, ranks=ranks, group_type='infer_tp')
                if rank in ranks:
                    _TENSOR_MODEL_PARALLEL_GROUP = group
                    ps._TENSOR_MODEL_PARALLEL_GROUP = _TENSOR_MODEL_PARALLEL_GROUP
    # 构建流水线模型并行组。
    # global _PIPELINE_MODEL_PARALLEL_GROUP
    # global _PIPELINE_GLOBAL_RANKS
    # assert ps._PIPELINE_MODEL_PARALLEL_GROUP is None, ("pipeline model parallel group is already initialized")

    # ps._PIPELINE_MODEL_PARALLEL_GROUP = mpu.get_pipeline_model_parallel_group()
    # ps._PIPELINE_GLOBAL_RANKS = mpu.get_pipeline_model_parallel_ranks()


"""
张量模型并行工具函数
"""


def get_tensor_model_parallel_group():
    """获取调用者 rank 所属的张量模型并行组。"""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, ("tensor model parallel group is not initialized")
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """返回张量模型并行组的 world size。"""
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_rank():
    """返回我在张量模型并行组中的 rank。"""
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_src_rank():
    """计算张量模型并行组中第一个本地 rank
    对应的全局 rank。"""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


"""
Micro 数据并行组
"""


def get_micro_data_parallel_group():
    assert _MICRO_DATA_PARALLEL_GROUP is not None
    return _MICRO_DATA_PARALLEL_GROUP


def get_micro_data_parallel_world_size():
    return torch.distributed.get_world_size(group=get_micro_data_parallel_group())


def get_micro_data_parallel_rank():
    return torch.distributed.get_rank(group=get_micro_data_parallel_group())
