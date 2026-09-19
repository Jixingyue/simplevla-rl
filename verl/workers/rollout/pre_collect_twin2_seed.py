import contextlib
import os
import torch
import torch.multiprocessing as mp
import sys
import importlib
import numpy as np
from PIL import Image
import random
import yaml
from pathlib import Path
import traceback
import time
from codetiming import Timer
from sapien.render import clear_cache as sapien_clear_cache
import warnings
import json
from datetime import datetime
from typing import List, Dict, Tuple

warnings.filterwarnings("ignore", message="Batch mode enable graph is only supported with num_graph_seeds==1")

def get_robotwin2_task(task_name, config):
    """参照 eval_policy.py 的方式获取 RoboTwin 2.0 任务"""
    # 将 robotwin2 路径添加到 sys.path
    robotwin2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'envs', 'robotwin2')
    if robotwin2_path not in sys.path:
        sys.path.append(robotwin2_path)
        
    robotwin2_utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'envs', 'robotwin2',"description","utils")
    if robotwin2_utils_path not in sys.path:
        sys.path.append(robotwin2_utils_path)
    
    # 从 robotwin2 导入必要的模块
    from envs import CONFIGS_PATH
    
    # 获取环境实例
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit(f"No Task: {task_name}")
    
    # 加载配置
    task_config = config.get('task_config', 'demo_randomized')
    config_file = os.path.join(robotwin2_path, f"task_config/{task_config}.yml")
    
    with open(config_file, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name
    args['task_config'] = task_config
    args['ckpt_setting'] = config.get('ckpt_setting', 'demo_randomized')
    
    # 加载本体（embodiment）配置
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file
    
    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args
    
    # 设置本体（embodiment）配置
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")
    
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    
    # 加载相机配置
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]
    
    # 设置评估模式
    args["eval_mode"] = True
    args["eval_video_log"] = False
    args["render_freq"] = 0
    
    args['instruction_type'] = config.get('instruction_type', 'unseen')
    
    return env_instance, args


def collect_success_seeds_worker(gpu_id: int, task_name: str, seed_ranges: List[Tuple[int, int]], 
                                target_per_worker: int, result_queue: mp.Queue, 
                                worker_id: int, num_workers: int):
    """
    在指定 GPU 上收集成功种子的 worker 函数

    Args:
        gpu_id: 要使用的 GPU 设备 ID
        task_name: 任务名称
        seed_ranges: 由 (start, end) 元组组成的种子范围列表
        target_per_worker: 该 worker 需要收集的成功种子目标数量
        result_queue: 用于存放结果的队列
        worker_id: 该 worker 的 ID
        num_workers: worker 总数
    """
    # 设置 GPU 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # 由于只能看到一个 GPU，因此始终是设备 0
    
    success_seeds = []
    setup_demo_fail_seeds = []
    play_once_fail_seeds = []
    not_success_seeds = []
    
    print(f"\n[Worker {worker_id}/{num_workers} - GPU {gpu_id}] Starting collection for task: {task_name}", flush=True)
    print(f"[Worker {worker_id}] Target: {target_per_worker} success seeds", flush=True)
    print(f"[Worker {worker_id}] Seed ranges: {seed_ranges}", flush=True)
    
    seeds_to_try = []
    for start, end in seed_ranges:
        seeds_to_try.extend(range(start, end + 1))
    
    for seed in seeds_to_try:
        if len(success_seeds) >= target_per_worker:
            print(f"\n[Worker {worker_id}] Successfully collected {target_per_worker} seeds!", flush=True)
            break
            
        print(f"\n[Worker {worker_id}, Task: {task_name}, Seed: {seed}] Starting evaluation...", flush=True)
        
        try:
            with Timer(name="get_robotwin2_task", logger=None) as process_timer:
                env, args = get_robotwin2_task(task_name, {})
            print(f"[Worker {worker_id}, Seed {seed}] get_robotwin2_task took: {process_timer.last:.3f} seconds", flush=True)
        except Exception as e:
            print(f"[Worker {worker_id}, Seed {seed}] Failed to get task: {e}", flush=True)
            continue
        
        with Timer(name="setup_demo", logger=None) as setup_demotimer:
            try:
                env.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **args)      
            except Exception as e:
                print(f"****[Worker {worker_id}, Seed {seed}] setup_demo fail: {e}***", flush=True) 
                setup_demo_fail_seeds.append(seed)
                env.close()
                continue
        print(f"[Worker {worker_id}, Seed {seed}] setup_demo took: {setup_demotimer.last:.3f} seconds", flush=True)      
                
        with Timer(name="play_once", logger=None) as play_once_demotimer:
            try:
                episode_info = env.play_once()
            except Exception as e:
                print(f"****[Worker {worker_id}, Seed {seed}] play_once fail: {e}***", flush=True) 
                play_once_fail_seeds.append(seed)
                env.close()
                continue
        print(f"[Worker {worker_id}, Seed {seed}] play_once took: {play_once_demotimer.last:.3f} seconds", flush=True)   
        
        env.close()
        if env.plan_success and env.check_success():
            success_seeds.append(seed)
            print(f"[Worker {worker_id}, Seed {seed}] SUCCESS! Total success: {len(success_seeds)}/{target_per_worker}", flush=True)
        else:
            not_success_seeds.append(seed)
            print(f"[Worker {worker_id}, Seed {seed}] Failed!", flush=True)
            
        total_processed = len(success_seeds) + len(not_success_seeds) + len(setup_demo_fail_seeds) + len(play_once_fail_seeds)
        if total_processed % 10 == 0:  # 每处理 10 个种子打印一次统计信息
            print(f"[Worker {worker_id}] Progress: {total_processed} seeds processed", flush=True)
            print(f"  - Success rate: {len(success_seeds)}/{total_processed} ({len(success_seeds)/total_processed*100:.1f}%)", flush=True)
    
    # 将结果放入队列
    result = {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "task_name": task_name,
        "success_seeds": success_seeds,
        "setup_demo_fail_seeds": setup_demo_fail_seeds,
        "play_once_fail_seeds": play_once_fail_seeds,
        "not_success_seeds": not_success_seeds,
        "seeds_tried": len(seeds_to_try)
    }
    result_queue.put(result)
    
    print(f"\n[Worker {worker_id}] Completed! Collected {len(success_seeds)} success seeds", flush=True)


def collect_success_seeds_for_task_parallel(task_name: str, seed_start: int, seed_end: int, 
                                           target_count: int, num_gpus: int = 8):
    """
    使用多个 GPU 并行收集单个任务的成功种子

    Args:
        task_name: 任务名称
        seed_start: 起始种子编号
        seed_end: 结束种子编号
        target_count: 需要收集的成功种子目标数量
        num_gpus: 使用的 GPU 数量

    Returns:
        dict: 所有 worker 的合并结果
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Starting parallel collection for task: {task_name}", flush=True)
    print(f"Target: {target_count} success seeds", flush=True)
    print(f"Seed range: {seed_start} - {seed_end}", flush=True)
    print(f"Using {num_gpus} GPUs", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 计算每个 worker 处理的种子数
    total_seeds = seed_end - seed_start + 1
    seeds_per_worker = total_seeds // num_gpus
    target_per_worker = (target_count + num_gpus - 1) // num_gpus  # 向上取整除法
    
    # 为每个 worker 创建种子范围
    worker_ranges = []
    for i in range(num_gpus):
        worker_start = seed_start + i * seeds_per_worker
        if i == num_gpus - 1:  # 最后一个 worker 处理剩余的种子
            worker_end = seed_end
        else:
            worker_end = worker_start + seeds_per_worker - 1
        worker_ranges.append([(worker_start, worker_end)])
    
    # 创建多进程上下文
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    
    # 启动 worker 进程
    processes = []
    for i in range(num_gpus):
        p = mp.Process(
            target=collect_success_seeds_worker,
            args=(i, task_name, worker_ranges[i], target_per_worker, 
                  result_queue, i, num_gpus)
        )
        p.start()
        processes.append(p)
        print(f"Started worker {i} on GPU {i} with seed range {worker_ranges[i]}", flush=True)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 收集所有 worker 的结果
    all_success_seeds = []
    all_setup_demo_fail_seeds = []
    all_play_once_fail_seeds = []
    all_not_success_seeds = []
    total_seeds_tried = 0
    
    while not result_queue.empty():
        result = result_queue.get()
        all_success_seeds.extend(result["success_seeds"])
        all_setup_demo_fail_seeds.extend(result["setup_demo_fail_seeds"])
        all_play_once_fail_seeds.extend(result["play_once_fail_seeds"])
        all_not_success_seeds.extend(result["not_success_seeds"])
        total_seeds_tried += result["seeds_tried"]
        
        print(f"\nWorker {result['worker_id']} results:", flush=True)
        print(f"  - Success seeds: {len(result['success_seeds'])}", flush=True)
        print(f"  - Seeds tried: {result['seeds_tried']}", flush=True)
    
    # 如果收集得更多，则裁剪到目标数量
    if len(all_success_seeds) > target_count:
        all_success_seeds = all_success_seeds[:target_count]
    
    # 最终汇总
    print(f"\n{'='*60}", flush=True)
    print(f"Task '{task_name}' parallel collection completed!", flush=True)
    print(f"Total success seeds collected: {len(all_success_seeds)}/{target_count}", flush=True)
    print(f"Total seeds tried: {total_seeds_tried}", flush=True)
    print(f"Overall success rate: {len(all_success_seeds)/total_seeds_tried*100:.1f}%", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    return {
        "task_name": task_name,
        "success_seeds": all_success_seeds,
        "setup_demo_fail_seeds": all_setup_demo_fail_seeds,
        "play_once_fail_seeds": all_play_once_fail_seeds,
        "not_success_seeds": all_not_success_seeds,
        "total_tried": total_seeds_tried
    }


def save_results(results, filepath, data_split):
    """将结果保存到 JSON 文件，包含合并逻辑

    Args:
        results: 包含待保存结果的字典
        filepath: JSON 文件路径
        data_split: 数据集划分类型（train/val/test）
    """
    # 检查文件是否存在
    if os.path.exists(filepath):
        # 加载已有数据
        with open(filepath, 'r') as f:
            existing_data = json.load(f)

        # 检查已存在的键并对覆盖发出警告
        for task_name in results:
            if task_name in existing_data:
                print(f"\n⚠️  WARNING: Task '{task_name}' already exists in {filepath}. Overwriting existing data!", flush=True)

        # 合并结果（相同键时新结果覆盖已有内容）
        existing_data.update(results)
        final_data = existing_data
    else:
        # 用结果创建新文件
        final_data = results
        print(f"\nCreating new file: {filepath}", flush=True)

    # 保存到文件
    with open(filepath, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"Results saved to: {filepath}", flush=True)
    print(f"Data split: {data_split}", flush=True)
    print(f"Total tasks in file: {len(final_data)}", flush=True)


def main(tasks=None, seed_start=100000, seed_end=100500, target_count=150, num_gpus=8, data_split="train"):
    """使用多个 GPU 为所有任务收集成功种子的主函数

    Args:
        tasks: 待处理的任务名称列表。如果为 None，则处理所有任务。
        seed_start: 起始种子编号
        seed_end: 结束种子编号
        target_count: 每个任务需要收集的成功种子目标数量
        num_gpus: 用于并行处理的 GPU 数量
        data_split: 数据集划分类型（train/val/test），默认为 "train"
    """
    # 默认任务列表
    all_tasks = ["handover_mic", "move_can_pot", "pick_dual_bottles", 
                 "place_phone_stand", "click_bell", "place_a2b_left", "place_a2b_right",
                 "lift_pot","put_bottles_dustbin","stack_blocks_two","stack_bowls_two",
                 "handover_block","place_empty_cup","shake_bottle","move_stapler_pad",
                 "place_container_plate","place_shoe","blocks_ranking_rgb","beat_block_hammer","place_mouse_pad","move_pillbottle_pad"]
    
    # 使用指定的任务或全部任务
    tasks_to_process = tasks if tasks is not None else all_tasks
    
    print(f"\nTasks to process: {tasks_to_process}", flush=True)
    print(f"Seed range: {seed_start} - {seed_end}", flush=True)
    print(f"Target count per task: {target_count}", flush=True)
    print(f"Number of GPUs: {num_gpus}", flush=True)
    print(f"Data split: {data_split}", flush=True)
    
    # 构建输出文件路径
    output_filepath = os.path.join(os.path.dirname(__file__), '..', '..', '..', f'robotwin2_{data_split}_seeds.json')
    print(f"Output file path: {output_filepath}", flush=True)
    
    # 结果存储
    all_results = {}
    
    # 处理每个任务
    for task_name in tasks_to_process:
        print(f"\n{'#'*80}", flush=True)
        print(f"PROCESSING TASK {tasks_to_process.index(task_name) + 1}/{len(tasks_to_process)}: {task_name}", flush=True)
        print(f"{'#'*80}", flush=True)
        
        try:
            results = collect_success_seeds_for_task_parallel(
                task_name=task_name,
                seed_start=seed_start,
                seed_end=seed_end,
                target_count=target_count,
                num_gpus=num_gpus
            )
            all_results[task_name] = results
            
            # 每个任务完成后用新的逻辑保存中间结果
            save_results(all_results, output_filepath, data_split)
            
        except Exception as e:
            print(f"\nERROR processing task '{task_name}': {e}", flush=True)
            traceback.print_exc()
            all_results[task_name] = {
                "task_name": task_name,
                "error": str(e),
                "success_seeds": [],
                "setup_demo_fail_seeds": [],
                "play_once_fail_seeds": [],
                "not_success_seeds": [],
                "total_tried": 0
            }
    
    # 打印最终汇总
    print_final_summary(all_results)
    
    return all_results


def print_final_summary(all_results):
    """打印所有任务的最终汇总"""
    print(f"\n{'='*80}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    
    for task_name, results in all_results.items():
        if "error" in results:
            print(f"\n{task_name}: ERROR - {results['error']}", flush=True)
        else:
            print(f"\n{task_name}:", flush=True)
            print(f"  - Success seeds: {len(results['success_seeds'])}", flush=True)
            print(f"  - Total seeds tried: {results['total_tried']}", flush=True)
            if results['total_tried'] > 0:
                success_rate = len(results['success_seeds']) / results['total_tried'] * 100
                print(f"  - Success rate: {success_rate:.1f}%", flush=True)
            
            # 打印前 10 个成功种子作为示例
            if len(results['success_seeds']) > 0:
                print(f"  - Sample success seeds: {results['success_seeds'][:10]}...", flush=True)
    
    print(f"\n{'='*80}", flush=True)


if __name__ == "__main__":
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Collect success seeds for RobotWin2 tasks using multiple GPUs')
    
    # 添加参数
    parser.add_argument('--tasks', nargs='+', 
                       choices=["handover_mic", "move_can_pot", "pick_dual_bottles", 
                               "place_phone_stand", "click_bell", "place_a2b_left", 
                               "place_a2b_right","lift_pot","put_bottles_dustbin",
                               "stack_blocks_two","stack_bowls_two","handover_block",
                               "place_empty_cup","shake_bottle","move_stapler_pad",
                               "place_container_plate","place_shoe","blocks_ranking_rgb","beat_block_hammer","place_mouse_pad","move_pillbottle_pad"],
                       help='Specify which tasks to collect seeds for. If not specified, all tasks will be processed.')
    
    parser.add_argument('--seed-start', type=int, default=100000,
                       help='Starting seed number (default: 100000)')
    
    parser.add_argument('--seed-end', type=int, default=105500,
                       help='Ending seed number (default: 105500)')
    
    parser.add_argument('--target-count', type=int, default=150,
                       help='Target number of success seeds to collect per task (default: 150)')
    
    parser.add_argument('--num-gpus', type=int, default=8,
                       help='Number of GPUs to use for parallel processing (default: 8)')
    
    parser.add_argument('--data-split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Data split type (train/val/test) (default: train)')
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查可用 GPU
    num_available_gpus = torch.cuda.device_count()
    if args.num_gpus > num_available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {num_available_gpus} available.", flush=True)
        print(f"Using {num_available_gpus} GPUs instead.", flush=True)
        args.num_gpus = num_available_gpus
    
    # 使用解析后的参数运行主收集流程
    all_results = main(
        tasks=args.tasks,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        target_count=args.target_count,
        num_gpus=args.num_gpus,
        data_split=args.data_split
    )