#!/usr/bin/env python
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

# -*- coding: utf-8 -*-
"""与文件系统无关的 IO API"""
import os
import tempfile
import hashlib

from .hdfs_io import copy, makedirs, exists

__all__ = ["copy", "exists", "makedirs"]

_HDFS_PREFIX = "hdfs://"


def _is_non_local(path):
    return path.startswith(_HDFS_PREFIX)


def md5_encode(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def get_local_temp_path(hdfs_path: str, cache_dir: str) -> str:
    """返回一个本地临时路径，由 cache_dir 与 hdfs_path 的 basename 拼接而成

    Args:
        hdfs_path:
        cache_dir:

    Returns:

    """
    # 对 hdfs_path 做编码以避免目录冲突
    encoded_hdfs_path = md5_encode(hdfs_path)
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)
    os.makedirs(temp_dir, exist_ok=True)
    dst = os.path.join(temp_dir, os.path.basename(hdfs_path))
    return dst


def copy_local_path_from_hdfs(src: str, cache_dir=None, filelock='.file.lock', verbose=False) -> str:
    """如果 src 位于 hdfs 上，则将 src 从 hdfs 复制到本地，否则直接返回 src。
    如果 cache_dir 为 None，将使用系统的默认缓存目录。注意，如果多次调用中
    src 名称相同，可能会导致冲突

    Args:
        src (str): 一个 HDFS 路径或本地路径

    Returns:
        复制后的文件的本地路径
    """
    from filelock import FileLock

    assert src[-1] != '/', f'Make sure the last char in src is not / because it will cause error. Got {src}'

    if _is_non_local(src):
        # 从 hdfs 下载到本地
        if cache_dir is None:
            # 获取一个临时文件夹
            cache_dir = tempfile.gettempdir()
        os.makedirs(cache_dir, exist_ok=True)
        assert os.path.exists(cache_dir)
        local_path = get_local_temp_path(src, cache_dir)
        # 获取一个特定的锁
        filelock = md5_encode(src) + '.lock'
        lock_file = os.path.join(cache_dir, filelock)
        with FileLock(lock_file=lock_file):
            if not os.path.exists(local_path):
                if verbose:
                    print(f'Copy from {src} to {local_path}')
                copy(src, local_path)
        return local_path
    else:
        return src
