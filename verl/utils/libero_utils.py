"""用于在 LIBERO 仿真环境中评估策略的工具函数。"""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
try:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as e:
    print(f"Warning : can't import libero: {e}")
    
import random
# from experiments.robot.robot_utils import (
#     DATE,
#     DATE_TIME,
# )


def get_libero_env(task, model_family, resolution=256):
    """初始化并返回 LIBERO 环境以及任务描述。"""
    # from libero.libero import get_libero_path
    # from libero.libero.envs import OffScreenRenderEnv
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # 重要：即使使用固定的初始状态，随机种子似乎也会影响物体位置
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """获取哑动作/空操作动作，用于在机器人不执行任何操作时推进仿真。"""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    接收对应单张图像的 numpy 数组，返回调整尺寸后的 numpy 数组图像。

    NOTE (Moo Jin): 为了使输入图像的分布与训练时所见输入保持一致，我们采用
                    Octo dataloader 所使用的相同缩放方案，OpenVLA 训练时也使用该方案。
    """

    assert isinstance(resize_size, tuple)
    # 调整为模型期望的图像尺寸
    img = tf.image.encode_jpeg(img)  # 编码为 JPEG，与 RLDS dataset builder 的做法一致
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # 立即解码回来
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """从观测中提取图像并进行预处理。"""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # 重要：旋转 180 度以匹配训练时的预处理
    img = resize_image(img, resize_size)
    return img


def get_libero_wrist_image(obs, resize_size):
    """从观测中提取图像并进行预处理。"""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # 重要：旋转 180 度以匹配训练时的预处理
    img = resize_image(img, resize_size)
    return img

# def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
#     """保存一段 episode 的 MP4 回放。"""
#     rollout_dir = f"./rollouts/{DATE}"
#     os.makedirs(rollout_dir, exist_ok=True)
#     processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
#     mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
#     video_writer = imageio.get_writer(mp4_path, fps=30)
#     for img in rollout_images:
#         video_writer.append_data(img)
#     video_writer.close()
#     print(f"Saved rollout MP4 at path {mp4_path}")
#     if log_file is not None:
#         log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
#     return mp4_path


def quat2axisangle(quat):
    """
    复制自 robosuite：https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    将四元数转换为轴角（axis-angle）格式。
    返回一个按其弧度角缩放的单位向量方向。

    Args:
        quat (np.array): (x,y,z,w) 形式的 vec4 浮点角度

    Returns:
        np.array: (ax,ay,az) 轴角指数坐标
    """
    # 裁剪四元数
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # 这是（接近）零角度的旋转，立即返回
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def get_image_resize_size(cfg):
    """
    获取某一模型类别对应的图像缩放尺寸。
    如果 `resize_size` 是整数，则缩放后的图像为正方形。
    否则，图像为矩形。
    """
    if cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size






# def normalize_gripper_action(action, binarize=True):
#     """
#     将夹爪动作（动作向量的最后一维）从 [0,1] 变换到 [-1,+1]。
#     对于某些环境（非 Bridge）这是必要的，因为 dataset wrapper 会将夹爪动作标准化到 [0,1]。
#     注意，与其他动作维度不同，dataset wrapper 默认不会将夹爪动作
#     归一化到 [-1,+1]。

#     归一化公式：y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
#     """
#     # Just normalize the last action to [-1,+1].
#     orig_low, orig_high = 0.0, 1.0
#     action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

#     if binarize:
#         # 二值化为 -1 或 +1。
#         action[..., -1] = np.sign(action[..., -1])

#     return action

def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    将夹爪动作从 [0,1] 归一化到 [-1,+1] 范围。

    对于某些环境这是必要的，因为 dataset wrapper 会将夹爪动作
    标准化到 [0,1]。注意，与其他动作维度不同，夹爪动作默认
    不会被归一化到 [-1,+1]。

    归一化公式：y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: 最后一维为夹爪动作的动作数组
        binarize: 是否将夹爪动作二值化为 -1 或 +1

    Returns:
        np.ndarray: 夹爪动作已归一化的动作数组
    """
    # 创建副本以避免修改原始数组
    normalized_action = action.copy()

    # 将最后一个动作维度归一化到 [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # 二值化为 -1 或 +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


# def invert_gripper_action(action):
#     """
#     翻转夹爪动作（动作向量的最后一维）的符号。
#     对于某些 -1 = 张开、+1 = 闭合的环境这是必要的，因为
#     RLDS dataloader 对齐夹爪动作的方式是 0 = 闭合、1 = 张开。
#     """
#     action[..., -1] = action[..., -1] * -1.0
#     return action

def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    翻转夹爪动作（动作向量的最后一维）的符号。

    对于 -1 = 张开、+1 = 闭合的环境这是必要的，因为
    RLDS dataloader 对齐夹爪动作的方式是 0 = 闭合、1 = 张开。

    Args:
        action: 最后一维为夹爪动作的动作数组

    Returns:
        np.ndarray: 夹爪动作已翻转的动作数组
    """
    # 创建副本以避免修改原始数组
    inverted_action = action.copy()

    # 翻转夹爪动作
    inverted_action[..., -1] =inverted_action[..., -1] *  -1.0

    return inverted_action

def save_rollout_video(rollout_images, exp_name, task_name, step_idx, success ):
    """保存一个回合的 MP4 回放。"""
    rollout_dir = f"./rollouts/{exp_name}" 
    os.makedirs(rollout_dir, exist_ok=True)
    ran_id = random.randint(1, 10000)
    #processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/step={step_idx}--task={task_name}--success={success}--ran={ran_id}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path
