import json
import pdb
import re
from typing import List, Dict, Any
import os
import argparse
import random
import yaml

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def extract_placeholders(instruction: str) -> List[str]:
    """从指令中提取所有形如 {X} 的占位符。"""
    placeholders = re.findall(r"{([^}]+)}", instruction)
    return placeholders


def filter_instructions(instructions: List[str], episode_params: Dict[str, str], rng: random.Random = None) -> List[str]:
    """
    过滤指令，仅保留所有占位符都与可用回合参数匹配的指令。
    不多不少。同时也接受不包含手臂占位符 {[a-z]} 的指令。
    
    参数：
        instructions：指令模板列表
        episode_params：回合参数字典
        rng：随机数生成器实例（如果为 None，则使用全局 random）
    """
    filtered_instructions = []
    # 创建副本以避免修改原始列表
    instructions_copy = instructions.copy()
    
    if rng is None:
        random.shuffle(instructions_copy)
    else:
        rng.shuffle(instructions_copy)

    for instruction in instructions_copy:
        placeholders = extract_placeholders(instruction)
        # 从 episode_params 键中移除 {} 以便比较
        stripped_episode_params = {key.strip("{}"): value for key, value in episode_params.items()}

        # 获取所有与手臂相关的参数（单个小写字母）
        arm_params = {key for key in stripped_episode_params.keys() if len(key) == 1 and "a" <= key <= "z"}
        non_arm_params = set(stripped_episode_params.keys()) - arm_params
        
        # 如果完全匹配，或者唯一缺失的参数是手臂参数，则接受
        if set(placeholders) == set(stripped_episode_params.keys()) or (
                # 特殊情况：如果唯一的差异是缺少手臂参数，则接受
                arm_params and set(placeholders).union(arm_params) == set(stripped_episode_params.keys()) and
                not arm_params.intersection(set(placeholders))):
            filtered_instructions.append(instruction)

    return filtered_instructions


def replace_placeholders(instruction: str, episode_params: Dict[str, str], rng: random.Random = None) -> str:
    """将指令中所有 {X} 占位符替换为 episode_params 中对应的值。
    对于手臂占位符 {[a-z]}，在值前面加 'the '，后面加 ' arm'。
    如果值是已存在 JSON 文件的路径，则随机选择一个 'description' 项并在前面加 'the'。
    如果值包含 '\' 或 '/' 但文件不存在，则打印粗体警告。
    
    参数：
        instruction：带占位符的指令模板
        episode_params：回合参数字典
        rng：随机数生成器实例（如果为 None，则使用全局 random）
    """
    # 从 episode_params 键中移除 {} 以便替换
    stripped_episode_params = {key.strip("{}"): value for key, value in episode_params.items()}

    for key, value in stripped_episode_params.items():
        placeholder = "{" + key + "}"
        # 检查值是否包含 '\' 或 '/'
        if "\\" in value or "/" in value:
            json_path = os.path.join(
                os.path.join(parent_directory, "../objects_description"),
                value + ".json",
            )
            if not os.path.exists(json_path):
                print(f"\033[1mERROR: '{json_path}' looks like a description file, but does not exist.\033[0m")
                exit()

        # 检查值是否是已存在 JSON 文件的路径
        json_path = os.path.join(os.path.join(parent_directory, "../objects_description"), value + ".json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                json_data = json.load(f)
            # 随机选择一个描述并在前面加 'the'
            descriptions = json_data.get("seen", [])
            if rng is None:
                description = random.choice(descriptions)
            else:
                description = rng.choice(descriptions)
            value = f"the {description}"
        # 检查键是否为单个小写字母（手臂占位符）
        elif len(key) == 1 and "a" <= key <= "z":
            value = f"the {value} arm"
        else:
            value = f"{value}"

        instruction = instruction.replace(placeholder, value)

    return instruction


def replace_placeholders_unseen(instruction: str, episode_params: Dict[str, str], rng: random.Random = None) -> str:
    """与 replace_placeholders 类似，但使用 JSON 文件中的 'unseen' 描述。
    对于手臂占位符 {[a-z]}，在值前面加 'the '，后面加 ' arm'。
    如果值是已存在 JSON 文件的路径，则随机选择一个 'unseen' 描述并在前面加 'the'。
    如果值包含 '\' 或 '/' 但文件不存在，则打印粗体警告。
    
    参数：
        instruction：带占位符的指令模板
        episode_params：回合参数字典
        rng：随机数生成器实例（如果为 None，则使用全局 random）
    """
    # 从 episode_params 键中移除 {} 以便替换
    stripped_episode_params = {key.strip("{}"): value for key, value in episode_params.items()}

    for key, value in stripped_episode_params.items():
        placeholder = "{" + key + "}"
        # 检查值是否包含 '\' 或 '/'
        if "\\" in value or "/" in value:
            json_path = os.path.join(
                os.path.join(parent_directory, "../objects_description"),
                value + ".json",
            )
            if not os.path.exists(json_path):
                print(f"\033[1mERROR: '{json_path}' looks like a description file, but does not exist.\033[0m")
                exit()

        # 检查值是否是已存在 JSON 文件的路径
        json_path = os.path.join(os.path.join(parent_directory, "../objects_description"), value + ".json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                json_data = json.load(f)
            # 随机选择一个 unseen 描述并在前面加 'the'
            if "unseen" in json_data and json_data["unseen"]:
                descriptions = json_data.get("unseen", [])
                if rng is None:
                    description = random.choice(descriptions)
                else:
                    description = rng.choice(descriptions)
                value = f"the {description}"
            else:
                # 如果 unseen 为空，则回退到 seen 描述
                descriptions = json_data.get("seen", [])
                if rng is None:
                    description = random.choice(descriptions)
                else:
                    description = rng.choice(descriptions)
                value = f"the {description}"
        # 检查键是否为单个小写字母（手臂占位符）
        elif len(key) == 1 and "a" <= key <= "z":
            value = f"the {value} arm"
        else:
            value = f"{value}"

        instruction = instruction.replace(placeholder, value)

    return instruction


def load_task_instructions(task_name: str) -> Dict[str, Any]:
    """从 JSON 文件加载任务指令。"""
    file_path = os.path.join(parent_directory, f"../task_instruction/{task_name}.json")
    with open(file_path, "r") as f:
        task_data = json.load(f)
    return task_data


def load_scene_info(task_name: str, setting: str, scene_info_path: str) -> Dict[str, Dict]:
    """从数据目录下的 JSON 文件加载场景信息。"""
    file_path = os.path.join(parent_directory, f"../../{scene_info_path}/{task_name}/{setting}/scene_info.json")
    try:
        with open(file_path, "r") as f:
            scene_data = json.load(f)
        return scene_data
    except FileNotFoundError:
        print(f"\033[1mERROR: Scene info file '{file_path}' not found.\033[0m")
        exit(1)
    except json.JSONDecodeError:
        print(f"\033[1mERROR: Scene info file '{file_path}' contains invalid JSON.\033[0m")
        exit(1)


def extract_episodes_from_scene_info(scene_info: Dict) -> List[Dict[str, str]]:
    """从 scene_info 中提取回合参数。"""
    episodes = []
    for episode_key, episode_data in scene_info.items():
        if "info" in episode_data:
            episodes.append(episode_data["info"])
        else:
            episodes.append(dict())
    return episodes


def save_episode_descriptions(task_name: str, setting: str, generated_descriptions: List[Dict]):
    """将生成的描述保存到输出文件。"""
    output_dir = os.path.join(parent_directory, f"../../data/{task_name}/{setting}/instructions")
    os.makedirs(output_dir, exist_ok=True)

    for episode_desc in generated_descriptions:
        episode_index = episode_desc["episode_index"]
        output_file = os.path.join(output_dir, f"episode{episode_index}.json")

        with open(output_file, "w") as f:
            json.dump(
                {
                    "seen": episode_desc.get("seen", []),
                    "unseen": episode_desc.get("unseen", []),
                },
                f,
                indent=2,
            )


def generate_episode_descriptions(task_name: str, episodes: List[Dict[str, str]], max_descriptions: int = 1000000, seed: int = None):
    """
    通过将指令中的占位符替换为参数值来为回合生成描述。
    对于每个回合，过滤出占位符匹配的指令，并通过将占位符替换为参数值
    生成最多 max_descriptions 条描述。
    现在同时也生成 unseen 描述。
    
    参数：
        task_name：任务名称（不含扩展名的 JSON 文件名）
        episodes：回合参数列表
        max_descriptions：每个回合的最大描述数量
        seed：用于可复现结果的随机种子。如果为 None，则使用全局随机状态。
    """
    # 如果提供了种子，则创建本地 Random 实例
    rng = random.Random(seed) if seed is not None else None
    
    # 加载任务指令
    task_data = load_task_instructions(task_name)
    seen_instructions = task_data.get("seen", [])
    unseen_instructions = task_data.get("unseen", [])

    # 存储每个回合生成的描述
    all_generated_descriptions = []

    # 处理每个回合
    for i, episode in enumerate(episodes):
        # 过滤出所有占位符都与回合参数匹配的指令
        filtered_seen_instructions = filter_instructions(seen_instructions, episode, rng)
        filtered_unseen_instructions = filter_instructions(unseen_instructions, episode, rng)

        if filtered_seen_instructions == [] and filtered_unseen_instructions == []:
            print(f"Episode {i}: No valid instructions found")
            continue

        # 通过替换占位符生成 seen 描述
        seen_episode_descriptions = []
        flag_seen = True
        while (len(seen_episode_descriptions) < max_descriptions and flag_seen and filtered_seen_instructions):
            for instruction in filtered_seen_instructions:
                if len(seen_episode_descriptions) >= max_descriptions:
                    flag_seen = False
                    break
                description = replace_placeholders(instruction, episode, rng)
                seen_episode_descriptions.append(description)

        # 通过替换占位符生成 unseen 描述
        unseen_episode_descriptions = []
        flag_unseen = True
        while (len(unseen_episode_descriptions) < max_descriptions and flag_unseen and filtered_unseen_instructions):
            for instruction in filtered_unseen_instructions:
                if len(unseen_episode_descriptions) >= max_descriptions:
                    flag_unseen = False
                    break
                description = replace_placeholders_unseen(instruction, episode, rng)
                unseen_episode_descriptions.append(description)

        all_generated_descriptions.append({
            "episode_index": i,
            "seen": seen_episode_descriptions,
            "unseen": unseen_episode_descriptions,
        })

    return all_generated_descriptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate episode descriptions by replacing placeholders")
    parser.add_argument(
        "task_name",
        type=str,
        help="Name of the task (JSON file name without extension)",
    )
    parser.add_argument(
        "setting",
        type=str,
        help="Setting name used to construct the data directory path",
    )
    parser.add_argument(
        "max_num",
        type=int,
        default=100,
        help="Maximum number of descriptions per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )

    args = parser.parse_args()
    setting_file = os.path.join(
        parent_directory, f"../../task_config/{args.setting}.yml"
    )
    with open(setting_file, "r", encoding="utf-8") as f:
        args_dict = yaml.load(f.read(), Loader=yaml.FullLoader)

    # 加载场景信息并提取回合参数
    scene_info = load_scene_info(args.task_name, args.setting, args_dict['save_path'])
    episodes = extract_episodes_from_scene_info(scene_info)

    # 使用种子生成描述
    results = generate_episode_descriptions(args.task_name, episodes, args.max_num, args.seed)

    # 将结果保存到输出文件
    save_episode_descriptions(args.task_name, args.setting, results)
    print("Successfully Saved Instructions")