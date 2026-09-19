import asyncio
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import math

from .evaluation_utils.code_util import evaluate_code
from .evaluation_utils.math_util import evaluate_math
from tqdm.asyncio import tqdm


# def process_completion(completion, task, reference):
#     if task == "code":
#         return evaluate_code(completion, reference)
#     elif task == "math":
#         return evaluate_math(completion, str(reference))
#     else:
#         print('task')
#         raise NotImplementedError

# def compute_score(completions, references, tasks):
#     # 处理结果
#     scores = []
#     format = []
#     results = process_completion(completions, tasks, references)
#     for result, completion, reference, task in zip(results, completions, references, tasks):
#         if isinstance(result, Exception) or result is None:
#             # 处理失败或超时的任务
#             scores.append(0.0)
#             format.append(0.0)
#             continue

#         try:
#             # 根据任务类型处理结果
#             if task == 'code' and not result[0]: # 若任务是 code，则 reference 应为 json 字符串
#                 correct = 0
#                 total = min(
#                     len(json.loads(reference)['inputs'] if not isinstance(reference, dict) else reference['inputs']),
#                     10)
#                 for run in result[1]:
#                     if 'test_case' in run and 'res' in run['test_case'] and run['test_case']['res'] == '[True]':
#                         correct += 1
#                 scores.append(correct / total)
#                 format.append(correct / total)
#             elif task == 'code':
#                 scores.append(float(int(result[0])))
#                 format.append(float(int(result[0])))
#             else:
#                 scores.append(float(int(result[0])))
#                 format.append(float(int(result[1])))
            
#         except Exception as e:
#             print(f"Error processing result for row: {completion[:10]}, Error: {e}")
#             scores.append(0.0)
#             format.append(0.0)
            
#     return scores, format








# #############################################################
# def process_completion(completion, task, reference):
#     if task == "code":
#         return evaluate_code(completion, reference)
#     elif task == "math":
#         return evaluate_math(completion, str(reference))
#     else:
#         print('task')
#         raise NotImplementedError

# async def process_row_with_timeout(completion, reference, task, executor, timeout=30.0):
#     """
#     处理单行数据，带超时限制。
#     """
#     loop = asyncio.get_running_loop()
#     try:
#         result = await asyncio.wait_for(
#             loop.run_in_executor(
#                 executor,
#                 partial(process_completion, completion, task, reference)
#             ),
#             timeout=timeout
#         )
#         return result
#     except asyncio.TimeoutError:
#         print(f"Timeout occurred for completion: {completion[:10]}, reference: {reference[:10]}")
#         return None  # 超时行的默认返回值
#     except Exception as e:
#         print(f"Error processing completion: {completion[:10]}, Error: {e}")
#         return None  # 失败行的默认返回值

# async def parallel_evaluate_continual_async(completions, references, tasks, num_processes, task_timeout=30.0):
#     """
#     使用进程池并行评估各行数据，并处理超时。
#     """
#     scores = []
#     format = []
#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         # 为所有行创建任务
#         tasks_async = [
#             process_row_with_timeout(completion, reference, task, executor, timeout=task_timeout)
#             for completion, reference, task in zip(completions, references, tasks)
#         ]
#         # 使用 tqdm 跟踪进度
#         try:
#             results = await asyncio.gather(*tasks_async, return_exceptions=False)
#         except Exception as exc:
#             print('Error encountered in parallel evaluation! now shutting all evaluation workers down to prevent starvation')
#             for pid, proc in executor._processes.items():
#                 try:
#                     proc.kill()
#                 except Exception as kill_err:
#                     print('shut down failed: '+str(kill_err))

#     # 处理结果
#     for result, completion, reference, task in zip(results, completions, references, tasks):
#         if isinstance(result, Exception) or result is None:
#             # 处理失败或超时的任务
#             scores.append(0.0)
#             format.append(0.0)
#             continue

#         try:
#             # 根据任务类型处理结果
#             if task == 'code' and not result[0]: # 若任务是 code，则 reference 应为 json 字符串
#                 correct = 0
#                 total = min(
#                     len(json.loads(reference)['inputs'] if not isinstance(reference, dict) else reference['inputs']),
#                     10)
#                 for run in result[1]:
#                     if 'test_case' in run and 'res' in run['test_case'] and run['test_case']['res'] == '[True]':
#                         correct += 1
#                 scores.append(correct / total)
#                 format.append(correct / total)
#             elif task == 'code':
#                 scores.append(float(int(result[0])))
#                 format.append(float(int(result[0])))
#             else:
#                 scores.append(float(int(result[0])))
#                 format.append(float(int(result[1])))
            
#         except Exception as e:
#             print(f"Error processing result for row: {completion[:10]}, Error: {e}")
#             scores.append(0.0)
#             format.append(0.0)
            
#     return scores, format

# def compute_score(completions, references, tasks):
#     # 三个列表的长度应完全一致
#     # TODO: 将其改为完全异步，即主进程在计算得分的同时可以做其他事情（例如前向奖励模型）
#     assert len(completions) == len(references) == len(tasks)
#     try:
#         return asyncio.run(parallel_evaluate_continual_async(completions, references, tasks, num_processes=64))
#     except asyncio.TimeoutError as e:
#         print('Global timeout in reward computing! Setting all as 0.5.')
#         return [0.5 for _ in range(len(completions))], [0.5 for _ in range(len(completions))]
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         return [0.5 for _ in range(len(completions))], [0.5 for _ in range(len(completions))]


import asyncio
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import math

from .evaluation_utils.code_util import evaluate_code
from .evaluation_utils.math_util import evaluate_math
from .evaluation_utils.libero_util import evaluate_libero
from tqdm.asyncio import tqdm

def process_completion(completion, task, reference):
    if task == "code":
        return evaluate_code(completion, reference)
    elif task == "math":
        return evaluate_math(completion, str(reference))
    elif task == "libero":
        return evaluate_libero(completion)
    else:
        print('task')
        raise NotImplementedError

async def process_row_with_timeout(completion, reference, task, executor, timeout=30.0):
    """
    处理单行数据，带超时限制。
    """
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(process_completion, completion, task, reference)
            ),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion[:10]}, reference: {reference[:10]}")
        return None  # 超时行的默认返回值
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  # 失败行的默认返回值

async def parallel_evaluate_continual_async(completions, references, tasks, num_processes, task_timeout=600.0):
    """
    使用进程池并行评估各行数据，并处理超时。
    """
    scores = []
    format = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 为所有行创建任务
        tasks_async = [
            process_row_with_timeout(completion, reference, task, executor, timeout=task_timeout)
            for completion, reference, task in zip(completions, references, tasks)
        ]
        # 使用 tqdm 跟踪进度
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as exc:
            print('Error encountered in parallel evaluation! now shutting all evaluation workers down to prevent starvation')
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: '+str(kill_err))

    # 处理结果
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # 处理失败或超时的任务
            scores.append(0.0)
            format.append(0.0)
            continue

        try:
            # 根据任务类型处理结果
            if task == 'code' and not result[0]: # 若任务是 code，则 reference 应为 json 字符串
                correct = 0
                total = min(
                    len(json.loads(reference)['inputs'] if not isinstance(reference, dict) else reference['inputs']),
                    10)
                for run in result[1]:
                    if 'test_case' in run and 'res' in run['test_case'] and run['test_case']['res'] == '[True]':
                        correct += 1
                scores.append(correct / total)
                #format.append(correct / total)
                format.append(float(int(result[2])))
            elif task == 'code':
                scores.append(float(int(result[0])))
                format.append(float(int(result[2])))
            else:
                scores.append(float(int(result[0])))
                format.append(float(int(result[1])))
            
        except Exception as e:
            print(f"Error processing result for row: {completion[:10]}, Error: {e}")
            scores.append(0.0)
            format.append(0.0)
            
    return scores, format

def compute_score(completions, references, tasks):
    # 三个列表的长度应完全一致
    # TODO: 将其改为完全异步，即主进程在计算得分的同时可以做其他事情（例如前向奖励模型）
    assert len(completions) == len(references) == len(tasks)
    try:
        return asyncio.run(parallel_evaluate_continual_async(completions, references, tasks, num_processes=128))
    except asyncio.TimeoutError as e:
        print('Global timeout in reward computing! Setting all as 0.5.')
        return [0.5 for _ in range(len(completions))], [0.5 for _ in range(len(completions))]
    except Exception as e:
        print(f"Unexpected error: {e}")
        return [0.5 for _ in range(len(completions))], [0.5 for _ in range(len(completions))]