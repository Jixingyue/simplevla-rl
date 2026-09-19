"""训练/微调脚本的工具函数。"""

import torch
import os
from .constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    # 创建一个张量，标记 IGNORE_INDEX 所在的位置
    newline_positions = token_ids != IGNORE_INDEX

    # 计算累加和以识别换行之间的区域
    cumsum = torch.cumsum(newline_positions, dim=1)

    # 创建掩码
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

    # 只提取动作部分
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def get_next_actions_mask(token_ids):
    # 创建一个张量，标记 IGNORE_INDEX 所在的位置
    newline_positions = token_ids != IGNORE_INDEX

    # 计算累加和以识别换行之间的区域
    cumsum = torch.cumsum(newline_positions, dim=1)

    # 创建掩码
    mask = cumsum > ACTION_DIM

    # 只提取动作部分
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss

def find_checkpoint_file(pretrained_checkpoint, file_pattern) :
    """
    查找匹配给定模式的特定 checkpoint 文件。

    参数：
        pretrained_checkpoint: checkpoint 目录的路径
        file_pattern: 用于匹配文件名的字符串模式

    返回：
        str: 匹配到的 checkpoint 文件路径

    抛出：
        AssertionError: 如果没有文件或有多个文件匹配该模式
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path) :
    """
    从 checkpoint 加载某个组件的 state dict，并处理可能存在的 DDP 前缀。

    参数：
        checkpoint_path: checkpoint 文件的路径

    返回：
        Dict: 处理后用于加载的 state 字典
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # 如果组件是用 DDP 训练的，state dict 中的元素带有 "module." 前缀，必须去除
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict