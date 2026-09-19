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

import functools
import json
import os

import ray

# 兼容性处理
from verl.single_controller.base.decorator import *


def maybe_remote(main):
    """如果配置中指定了 VERL_DRIVER_NUM_GPUS 或 VERL_DRIVER_RESOURCES，则将 main 函数调度为 ray remote 任务。
       - VERL_DRIVER_NUM_GPUS: driver 任务的 GPU 数量。
       - VERL_DRIVER_RESOURCES: driver 任务的自定义资源，例如 {"verl_driver": 1.0}。

    若要向 ray 集群提交作业，可以在 runtime.yaml 中指定这两个环境变量。
    ```yaml
    working_dir: "."
    env_vars:
      VERL_DRIVER_NUM_GPUS: "1"
      VERL_DRIVER_RESOURCES: '{"verl_driver": 1.0}'
    ```

    ray job submit --runtime-env=runtime.yaml -- python3 test.py

    Args:
        main (Callable): 待调度的 main 函数。
    """

    num_gpus = 0
    resources = {}
    env_num_gpus = os.getenv("VERL_DRIVER_NUM_GPUS")
    if env_num_gpus:
        num_gpus = int(env_num_gpus)
    env_resources = os.getenv("VERL_DRIVER_RESOURCES")
    if env_resources:
        resources = json.loads(env_resources)
    print(f"verl driver num_gpus: {num_gpus}, resources={resources}")
    assert isinstance(resources, dict), f"resources must be dict, got {type(resources)}"

    @functools.wraps(main)
    def _main(*args, **kwargs):
        # 在本地运行 main 函数。
        if num_gpus == 0 and len(resources) == 0:
            return main(*args, **kwargs)

        # 作为 ray task 在远端运行 main 函数。
        f = ray.remote(num_gpus=num_gpus, resources=resources)(main)
        return ray.get(f.remote(*args, **kwargs))

    return _main
