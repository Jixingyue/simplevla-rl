# 借鉴自: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py


import multiprocessing
from typing import Dict, Optional
from datasets import load_dataset
from .testing_util import run_test
import traceback
import os,sys

def _temp_run(sample, generation, debug, result,metadata_list,timeout):
    # 测试以静默方式运行。若进程被杀掉，则不会有任何输出
   
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata= run_test(in_outs=sample, test=generation, debug=debug,timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception as e:
            # print(e) # 某些 traceback 信息极其长。
            traceback.print_exc(10)
            result.append([-1 for i in range(len(sample['inputs']))])
            metadata_list.append({})

def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    """以全局超时机制检查代码生成的正确性。
    设置全局超时是为了捕捉 `run_test` 内部的超时机制无法处理的极端/罕见情况"""
    
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result,metadata_list,timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
        # p.terminate()
    if not result:
        # 认为所有测试均失败
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0], metadata_list