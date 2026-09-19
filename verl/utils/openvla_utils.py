"""用于评估 OpenVLA 或微调后 OpenVLA 策略的工具函数。"""

import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# 应用 json_numpy 补丁以支持序列化
json_numpy.patch()

# 配置 NumPy 打印选项
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    更新 checkpoint config.json 文件中的 AutoMap 配置。

    该函数会加载 checkpoint 目录中的 config.json 文件，并将
    AutoConfig 和 AutoModelForVision2Seq 字段改写为 OpenVLA 特定的类。

    Args:
        pretrained_checkpoint: checkpoint 目录的路径
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # 创建带时间戳的备份
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # 读取并更新配置
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # 写回更新后的配置
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    检查两个文件的内容是否完全相同。

    Args:
        path1: 第一个文件的路径
        path2: 第二个文件的路径

    Returns:
        bool: 如果文件相同则返回 True，否则返回 False
    """
    path1, path2 = Path(path1), Path(path2)

    # 先检查文件大小是否一致
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # 检查内容是否一致
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    处理当前目录与 checkpoint 之间的文件同步。

    如果文件已存在但内容不同，则创建备份，并将当前版本复制到 checkpoint。

    Args:
        curr_filepath: 当前文件版本的路径
        checkpoint_filepath: 文件在 checkpoint 中应有的路径
        file_type: 用于日志记录的文件类型描述
    """
    if os.path.exists(checkpoint_filepath):
        # 检查已有文件是否相同
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # 创建带时间戳的备份
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # 将当前版本复制到 checkpoint 目录
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # 如果 checkpoint 目录中不存在该文件，则复制过去
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    检查并同步当前代码与 checkpoint 之间的模型逻辑文件。

    处理 modeling_prismatic.py 和 configuration_prismatic.py 的当前版本
    与 checkpoint 版本之间的关系：
    - 如果 checkpoint 文件存在且不同：创建备份并复制当前版本
    - 如果 checkpoint 文件不存在：复制当前版本

    Args:
        pretrained_checkpoint: checkpoint 目录的路径
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # 查找当前文件
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # 检查并处理每个文件
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)







