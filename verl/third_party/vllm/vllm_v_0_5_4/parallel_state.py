# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# 改编自
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""模型并行和数据并行分组。"""
import os
import torch
import torch.distributed
from typing import Optional

import vllm.distributed.parallel_state as ps
from vllm.distributed.parallel_state import get_pp_group, get_world_group, init_distributed_environment, init_model_parallel_group

import vllm.envs as envs
from vllm.logger import init_logger

from torch.distributed.device_mesh import init_device_mesh

logger = init_logger(__name__)
"""
该版本与 Megatron 深度耦合，以实现 HybridEngine 以及 vllm 与 Megatron 之间的权重共享。
- 我们假定在调用此函数之前，Megatron 的 tp+dp+pp 并行世界已经建立。

"""

# 用于 DTensor 的 device mesh
_DEVICE_MESH = None

# 当前 rank 所属的张量模型并行组。
_TP = None
# 当前 rank 所属的流水线模型并行组。
_PP = None


# 该方法用于在使用 HybridEngine 时初始化 ParallelGroup
def initialize_parallel_state(
    distributed_init_method: str = "env://",
    backend: str = "nccl",
    tensor_model_parallel_size: int = 1,
    num_tp_per_train_tp: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    # torch.distributed.all_reduce 在到达同步点之前不会释放输入张量。
    # 这会导致内存占用随着 all_reduce 调用次数的增加而增长。
    # 该环境变量禁用了这一行为。
    # 相关 issue：
    # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    # NOTE(sgm): 为 verl 修改，环境变量将由 TORCHRUN 设置。
    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # 使用 TORCHRUN 设置的 world_size
    world_size = int(os.getenv("WORLD_SIZE", "-1"))
    assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
    init_distributed_environment(world_size, rank, distributed_init_method, local_rank, backend)
    if torch.distributed.get_world_size() > 1:
        # NOTE: 使用 infer tp 和 micro dp 构建一个独立的推理组
        initialize_model_parallel_for_vllm(tensor_model_parallel_size=tensor_model_parallel_size,
                                           num_tensor_model_parallel_groups_per_train_tp=num_tp_per_train_tp)
    else:
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, backend)


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """辅助函数：若模型并行组尚未初始化则进行初始化；
    若已初始化，则确保张量并行和流水线并行的大小与期望值一致。
    """
    # 获取 _DEVICE_WORLD_GROUP 的 backend
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, backend)
        return

    assert (get_tensor_model_parallel_world_size() == tensor_model_parallel_size), (
        "tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}")
    pp_world_size = get_pp_group().world_size
    assert (pp_world_size == pipeline_model_parallel_size), (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{pp_world_size=} vs. "
        f"{pipeline_model_parallel_size=}")


# TODO(sgm): 与 v0.5.4 不同，现在不支持 pp
def model_parallel_is_initialized():
    """检查张量并行组和流水线并行组是否已初始化。"""
    return (ps._TP is not None)
    # and _PIPELINE_MODEL_PARALLEL_GROUP is not None)


def initialize_model_parallel_for_vllm(tensor_model_parallel_size: int,
                                       num_tensor_model_parallel_groups_per_train_tp: int = 1,
                                       pipeline_model_parallel_size: int = 1) -> None:
    from torch.distributed import new_group
    # 获取 world size 和 rank。确保一些一致性。
    assert torch.distributed.is_initialized()

    assert isinstance(tensor_model_parallel_size, int)

    # assert num_tensor_model_parallel_groups_per_train_tp == 1 and not different_tp_group
    # assert num_tensor_model_parallel_groups_per_train_tp > 1 and different_tp_group

    # 构建张量模型并行组。
    assert ps._TP is None, ("tensor model parallel group is already initialized")

    global _TP

    world_size: int = torch.distributed.get_world_size()

    rank = torch.distributed.get_rank()

    backend = torch.distributed.get_backend()

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size

    if num_tensor_model_parallel_groups_per_train_tp == 1:
        # if tensor_model_parallel_size == train_tensor_parallel_size:
        # 使用与 Megatron/vllm 相同的 tp 组
        assert _TP is None, ("tensor model parallel group is already initialized")
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            group_ranks.append(ranks)
        _TP = init_model_parallel_group(
            group_ranks=group_ranks,
            local_rank=get_world_group().local_rank,
            backend=backend,
            use_custom_allreduce=False,  # TODO: 检查为什么在 Ray trainer 中设为 True 不起作用
            use_message_queue_broadcaster=True)
        ps._TP = _TP
        # _MICRO_DATA_PARALLEL_GROUP 已移至 hybrid engine
    else:
        # 初始化一个 micro_dp 组和一个 tp 组
        # 假设训练 tp=4，推理 tp=2，那么权重的划分方式为：
        # 训练时 [1], [2], [3], [4]，推理时 [1,2], [1,2], [3,4], [3,4]

        # 构建推理 tp 组
        # train_tp = train_tensor_parallel_size
        train_tp = num_tensor_model_parallel_groups_per_train_tp * tensor_model_parallel_size
        # num_tensor_model_parallel_groups_per_train_tp = train_tp // tensor_model_parallel_size
        assert _TP is None, ("tensor model parallel group is already initialized")
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups // num_tensor_model_parallel_groups_per_train_tp):
            start = train_tp * i
            end = train_tp * (i + 1)
            for j in range(num_tensor_model_parallel_groups_per_train_tp):
                ranks = list(range(start, end, num_tensor_model_parallel_groups_per_train_tp))
                for i in range(len(ranks)):
                    ranks[i] += j
                group_ranks.append(ranks)
        _TP = init_model_parallel_group(
            group_ranks=group_ranks,
            local_rank=get_world_group().local_rank,
            backend=backend,
            use_custom_allreduce=False,  # TODO: 检查为什么在 Ray trainer 中设为 True 不起作用
            use_message_queue_broadcaster=True)
        ps._TP = _TP

    # 构建流水线模型并行组。
    # global _PIPELINE_MODEL_PARALLEL_GROUP
    # global _PIPELINE_GLOBAL_RANKS
    # assert ps._PIPELINE_MODEL_PARALLEL_GROUP is None, ("pipeline model parallel group is already initialized")

    # ps._PIPELINE_MODEL_PARALLEL_GROUP = mpu.get_pipeline_model_parallel_group()
    # ps._PIPELINE_GLOBAL_RANKS = mpu.get_pipeline_model_parallel_ranks()

    # TODO: 使用 device mesh 初始化（目前不支持 hybrid engine）
    # 构建流水线模型并行组。
    num_pipeline_model_parallel_groups: int = (world_size // pipeline_model_parallel_size)
    global _PP
    assert _PP is None, ("pipeline model parallel group is already initialized")
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # 流水线并行不需要 custom allreduce
    _PP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_custom_allreduce=False)
    ps._PP = _PP  # 为 verl 添加


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    NOTE: 该方法是对开源版本的一个改动，去掉了 world_size = tp * pp 的断言。

    初始化模型并行组。

    Arguments:
        tensor_model_parallel_size: 用于张量模型并行的 GPU 数量。
        pipeline_model_parallel_size: 用于流水线模型并行的 GPU 数量。

    假设我们总共有 8 个 GPU，记为 g0 ... g7，使用 2 个 GPU 做
    模型张量并行，4 个 GPU 做模型流水线并行。本函数将创建
    4 个张量模型并行组和 2 个流水线模型并行组：
        4 个张量模型并行组：
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 个流水线模型并行组：
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    注意，出于效率考虑，调用方应确保相邻的 rank 位于同一台
    DGX 机器上。例如，如果我们使用 2 台各 16 卡的 DGX-1 机器，
    rank 0 到 7 属于第一台机器，rank 8 到 15 属于第二台机器。
    """
    # 获取 world size 和 rank。确保一些一致性。
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(ps.get_world_group().device_group)

    # NOTE(sgm) 我们不断言 world_size == tp * pp
    # DP 不是由 vllm 管理的，而是由 veRL WorkerGroup 管理
    # if (world_size !=
    #         tensor_model_parallel_size * pipeline_model_parallel_size):
    #     raise RuntimeError(
    #         f"world_size ({world_size}) is not equal to "
    #         f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
    #         f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

    num_tensor_model_parallel_groups: int = (world_size // tensor_model_parallel_size)
    rank = torch.distributed.get_rank()
    global _TP
    assert _TP is None, ("tensor model parallel group is already initialized")
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
        group_ranks.append(ranks)

    # message queue broadcaster 仅在张量模型并行组中使用
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_custom_allreduce=False,  # TODO: 检查为什么在 Ray trainer 中设为 True 不起作用
        use_message_queue_broadcaster=True)
    ps._TP = _TP

    # TODO: 使用 device mesh 初始化（目前不支持 hybrid engine）
    # 构建流水线模型并行组。
    num_pipeline_model_parallel_groups: int = (world_size // pipeline_model_parallel_size)
    global _PP
    assert _PP is None, ("pipeline model parallel group is already initialized")
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # 流水线并行不需要 custom allreduce
    _PP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_custom_allreduce=False)
    ps._PP = _PP  # 为 verl 添加


"""
Device mesh 工具函数
"""


def get_device_mesh():
    assert _DEVICE_MESH is not None, ("device mesh is not initialized")
    return _DEVICE_MESH


"""
张量模型并行工具函数
"""


def get_tensor_model_parallel_group():
    """获取调用方 rank 所属的张量模型并行组。"""
    assert _TP is not None, ("tensor model parallel group is not initialized")
    return _TP.device_group


def get_tensor_model_parallel_world_size():
    """返回张量模型并行组的 world size。"""
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_rank():
    """返回调用方在张量模型并行组中的 rank。"""
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_src_rank():
    """计算张量模型并行组中第一个 local rank 对应的全局 rank。"""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size
