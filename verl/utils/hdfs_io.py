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

import os
import shutil
import logging

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

_HDFS_PREFIX = "hdfs://"

_HDFS_BIN_PATH = shutil.which('hdfs')


def exists(path: str, **kwargs) -> bool:
    r"""功能类似于 os.path.exists()，但支持 hdfs。

    测试路径是否存在。对于失效的符号链接返回 False。

    Args:
        path (str): 要测试的路径

    Returns:
        bool: 如果路径存在则返回 True，否则返回 False
    """
    if _is_non_local(path):
        return _exists(path, **kwargs)
    return os.path.exists(path)


def _exists(file_path: str):
    """ 支持 hdfs 检查 file_path 是否存在 """
    if file_path.startswith("hdfs"):
        return _run_cmd(_hdfs_cmd(f"-test -e {file_path}")) == 0
    return os.path.exists(file_path)


def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    r"""功能类似于 os.makedirs()，但支持 hdfs。

    超级 mkdir；创建一个叶子目录及其所有中间目录。工作方式类似
    mkdir，区别在于任何中间路径段（不仅仅是最后一段）在不存在时
    都会被创建。如果目标目录已存在，当 exist_ok 为 False 时抛出 OSError，
    否则不抛出异常。该操作是递归的。

    Args:
        name (str): 要创建的目录
        mode (int): 文件权限位
        exist_ok (bool): 如果为 True，当目录已存在时不抛出异常
        kwargs: hdfs 的关键字参数

    """
    if _is_non_local(name):
        # TODO(haibin.lin):
        # - 处理 hdfs 的 OSError(?)
        # - 为 hdfs 支持 exist_ok(?)
        _mkdir(name, **kwargs)
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


def _mkdir(file_path: str) -> bool:
    """hdfs mkdir"""
    if file_path.startswith("hdfs"):
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))
    else:
        os.makedirs(file_path, exist_ok=True)
    return True


def copy(src: str, dst: str, **kwargs) -> bool:
    r"""对文件功能类似于 shutil.copy()，对目录功能类似于 shutil.copytree，并支持 hdfs。

    复制数据和权限位（"cp src dst"）。返回文件的目标路径。
    目标可以是一个目录。
    如果源和目标是同一个文件，将抛出 SameFileError。

    Arg:
        src (str): 源文件路径
        dst (str): 目标文件路径
        kwargs: hdfs copy 的关键字参数

    Returns:
        str: 目标文件路径

    """
    if _is_non_local(src) or _is_non_local(dst):
        # TODO(haibin.lin):
        # - 处理 hdfs 文件的 SameFileError(?)
        # - 返回 hdfs 文件的目标路径
        return _copy(src, dst)
    else:
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        else:
            return shutil.copy(src, dst, **kwargs)


def _copy(from_path: str, to_path: str, timeout: int = None) -> bool:
    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(_hdfs_cmd(f"-cp -f {from_path} {to_path}"), timeout=timeout)
        else:
            returncode = _run_cmd(_hdfs_cmd(f"-put -f {from_path} {to_path}"), timeout=timeout)
    else:
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(_hdfs_cmd(f"-get \
                {from_path} {to_path}"), timeout=timeout)
        else:
            try:
                shutil.copy(from_path, to_path)
                returncode = 0
            except shutil.SameFileError:
                returncode = 0
            except Exception as e:
                logger.warning(f"copy {from_path} {to_path} failed: {e}")
                returncode = -1
    return returncode == 0


def _run_cmd(cmd: str, timeout=None):
    return os.system(cmd)


def _hdfs_cmd(cmd: str) -> str:
    return f"{_HDFS_BIN_PATH} dfs {cmd}"


def _is_non_local(path: str):
    return path.startswith(_HDFS_PREFIX)
