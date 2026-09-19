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

from typing import Dict, Optional

import ray

from .base import RayWorkerGroup, RayResourcePool, RayClassWithInitArgs
from verl.single_controller.base.megatron.worker import DistRankInfo, DistGlobalInfo
from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup


# NOTE(sgm): 适用于开源的 megatron-core
class NVMegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    """
    MegatronWorkerGroup 会向每个 worker 查询其 megatron rank 信息并保存在 WorkerGroup 内部，
    以便 dispatcher 利用这些信息分发数据。
    """

    def __init__(self, resource_pool: RayResourcePool, ray_cls_with_init: RayClassWithInitArgs, **kwargs):
        super().__init__(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, **kwargs)
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name='get_megatron_rank_info')
        self._megatron_global_info: DistGlobalInfo = ray.get(
            self.execute_rank_zero_async(method_name='get_megatron_global_info'))


class MegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    """
    MegatronWorkerGroup 会向每个 worker 查询其 megatron rank 信息并保存在 WorkerGroup 内部，
    以便 dispatcher 利用这些信息分发数据。
    """

    def __init__(self,
                 resource_pool: RayResourcePool,
                 ray_cls_with_init: RayClassWithInitArgs,
                 default_megatron_kwargs: Dict = None,
                 **kwargs):
        super().__init__(resource_pool=resource_pool,
                         ray_cls_with_init=ray_cls_with_init,
                         default_megatron_kwargs=default_megatron_kwargs,
                         **kwargs)
        self.init_megatron(default_megatron_kwargs=default_megatron_kwargs)
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name='get_megatron_rank_info')
        self._megatron_global_info: DistGlobalInfo = ray.get(
            self.execute_rank_zero_async(method_name='get_megatron_global_info'))

    def init_megatron(self, default_megatron_kwargs: Optional[Dict] = None):
        # 在 super 之后，我们将调用每个 worker 的 init
        if not self._is_init_with_detached_workers:
            # 只有当 WorkerGroup 是从零创建时才调用 init_megatron
            self.execute_all_sync(method_name='init_megatron', default_megatron_kwargs=default_megatron_kwargs)
