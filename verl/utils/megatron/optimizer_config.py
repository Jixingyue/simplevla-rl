# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class OptimizerConfig:
    """优化器配置。"""

    ##############
    # 通用
    ##############
    optimizer: str = 'adam'
    """要使用的优化器（Adam 或 SGD 之一）。"""

    lr: Optional[float] = None
    """初始学习率。根据衰减方式和初始预热，每次迭代的学习率会有所不同。
    """

    min_lr: Optional[float] = None
    """学习率的最小值。学习率调度器会将低于该阈值的值截断。"""

    decoupled_lr: Optional[float] = None
    """输入层和输出层单独使用的学习率。"""

    decoupled_min_lr: Optional[float] = None
    """输入层和输出层学习率的最小值。学习率调度器会将低于该阈值的值截断。
    """

    weight_decay: float = 0.01
    """L2 正则化的权重衰减系数。"""

    ##############
    # 精度
    ##############
    fp16: bool = False
    """若为 True，则使用 fp16 混合精度训练。默认为 False。"""

    bf16: bool = False
    """若为 True，则使用 bf16 混合精度训练。默认为 False。"""

    params_dtype: torch.dtype = torch.float32
    """初始化权重时使用的 dtype。默认为 torch.float32。"""

    ###############
    # 损失缩放
    ###############
    loss_scale: Optional[float] = None
    """静态损失缩放因子，取 2 的正整数次幂有助于提升 fp16 收敛性。若为 None，
       则使用动态损失缩放。
    """

    initial_loss_scale: float = 2**32
    """动态损失缩放的初始损失缩放因子。"""

    min_loss_scale: float = 1.0
    """动态损失缩放的最小损失缩放因子。"""

    loss_scale_window: float = 1000
    """动态缩放值上调/下调的窗口。"""

    hysteresis: int = 2
    """动态损失缩放的迟滞参数。"""

    ##############
    # 优化器
    ##############
    # Adam
    adam_beta1: float = 0.9
    """Adam 优化器中用于计算梯度及其平方滑动平均的第一个系数。
    """

    adam_beta2: float = 0.999
    """Adam 优化器中用于计算梯度及其平方滑动平均的第二个系数。
    """

    adam_eps: float = 1e-08
    """加到分母上的项，用于提升 Adam 优化器的数值稳定性。"""

    # SGD。
    sgd_momentum: float = 0.9
    """SGD 优化器的动量因子。"""

    #######################
    # 分布式优化器
    #######################
    use_distributed_optimizer: bool = False
    """将优化器状态分布到各个数据并行副本上。"""

    overlap_grad_reduce: bool = False
    """若为 True，在分布式优化器中将梯度 reduce-scatter 与反向计算重叠。"""

    overlap_param_gather: bool = False
    """若为 True，在分布式优化器中将参数 all-gather 与前向计算重叠。"""

    ################
    # 其他
    ################
    clip_grad: float = 1.0
    """基于全局 L2 范数的梯度裁剪。"""

    log_num_zeros_in_grad: bool = False
    """若为 True，则计算并记录梯度中零元素的数量。"""

    barrier_with_L1_time: bool = False
    """若为 True，则在进行级别 1 的计时测量时使用 barrier。"""

    timers: Callable = None
    """获取计时器的函数。"""
