"""
VLA 训练与评估的重要常量。

尝试根据用于启动训练或评估的 Python 命令自动识别应设置的正确常量。若无法确定，
则默认使用 LIBERO 仿真基准的常量。
"""
import os
import sys
from enum import Enum

# Llama 2 token 常量
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# 定义动作与本体感觉状态支持的归一化方式。
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # 归一化为均值 = 0，标准差 = 1
    BOUNDS = "bounds"               # 归一化到区间 = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # 将 [quantile_01, ..., quantile_99] 归一化 --> [-1, ..., 1]
    # fmt: on


# 为每个机器人平台定义常量
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

ALOHA_CONSTANTS_12chunk = {
    "NUM_ACTIONS_CHUNK": 12,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

ALOHA_CONSTANTS_8chunk = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

ALOHA_CONSTANTS_6chunk = {
    "NUM_ACTIONS_CHUNK": 6,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# 根据命令行参数检测机器人平台的函数
def detect_robot_platform():
    
    robot_env = os.environ.get('ROBOT_PLATFORM', '').upper()
    if robot_env:
        # 环境变量映射到平台
        env_mapping = {
            'LIBERO': 'LIBERO',
            'ALOHA': 'ALOHA',
            'ALOHA_12': 'ALOHA_12',
            'ALOHA_8': 'ALOHA_8',
            'ALOHA_6': 'ALOHA_6',
            'BRIDGE': 'BRIDGE',
        }
        if robot_env in env_mapping:
            print(f"Detected robot platform from environment: {env_mapping[robot_env]}")
            return env_mapping[robot_env]
    
    cmd_args = " ".join(sys.argv).lower()

    if "aloha_12chunk" in cmd_args:
        return "ALOHA_12"
    elif "aloha_8chunk" in cmd_args:
        return "ALOHA_8"
    elif "aloha_6chunk" in cmd_args:
        return "ALOHA_6"
    elif "libero" in cmd_args:
        return "ALOHA"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    else:
        # TODO (cjh, fix): 修改此处使其更健壮
        # 无法确定时默认使用 ALOHA
        return "ALOHA"


# 确定使用哪个机器人平台
ROBOT_PLATFORM = detect_robot_platform()
#ROBOT_PLATFORM = "ALOHA_12"

# 根据检测到的平台设置相应的常量
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA_12":
    constants = ALOHA_CONSTANTS_12chunk
elif ROBOT_PLATFORM == "ALOHA_8":
    constants = ALOHA_CONSTANTS_8chunk
elif ROBOT_PLATFORM == "ALOHA_6":
    constants = ALOHA_CONSTANTS_6chunk   
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS

# 将常量赋值给全局变量
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# 打印正在使用哪个机器人平台的常量（用于调试）
print(f"Using {ROBOT_PLATFORM} constants:",flush=True)
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}",flush=True)
# print(f"  ACTION_DIM = {ACTION_DIM}")
# print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
# print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
# print("If needed, manually set the correct constants in `/verl/utils/vla_utils/openvla_oft/constants.py`!")
