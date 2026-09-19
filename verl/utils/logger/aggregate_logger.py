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
"""
Ray 日志器，用于接收来自不同进程的日志信息。
"""
import numbers
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


def concat_dict_to_str(dict: Dict, step):
    output = [f'step:{step}']
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f'{k}:{v:.3f}')
    output_str = ' - '.join(output)
    return output_str


# class LocalLogger:

#     def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False):
#         self.print_to_console = print_to_console
#         if print_to_console:
#             print('Using LocalLogger is deprecated. The constructor API will change ')

#     def flush(self):
#         pass

#     def log(self, data, step):
#         if self.print_to_console:
#             print(concat_dict_to_str(data, step=step), flush=True)

class LocalLogger:
    """一个将数据写入本地文件并可选打印到控制台的日志器。"""
    
    def __init__(self, 
                 remote_logger=None, 
                 enable_wandb=False, 
                 print_to_console=False,
                 log_dir: str = "logs",
                 filename_prefix: str = "run"):
        """
        初始化 LocalLogger。
        
        参数:
            remote_logger: 旧参数（未使用）
            enable_wandb: 旧参数（未使用）
            print_to_console: 是否将日志打印到控制台
            log_dir: 日志文件存储目录
            filename_prefix: 日志文件名前缀
        """
        self.print_to_console = print_to_console
        if print_to_console:
            print('Using LocalLogger is deprecated. The constructor API will change')
        
        # 如果日志目录不存在则创建
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成带时间戳的唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.log"
        self.log_path = os.path.join(log_dir, filename)
        
        # 写入日志文件头部进行初始化
        with open(self.log_path, 'w') as f:
            f.write(f"# Log started at {datetime.now().isoformat()}\n")
            f.write("# Format: {timestamp}\t{step}\t{data_json}\n")
    
    def flush(self):
        """实现 flush 方法以保持兼容性。"""
        pass
    
    def log(self, data: Dict[str, Any], step: int) -> None:
        """
        将数据记录到文件，并可选输出到控制台。
        
        参数:
            data: 包含待记录指标/数据的字典
            step: 当前步数
        """
        # 控制台输出
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)
        
        # 文件输出
        timestamp = datetime.now().isoformat()
        data_str = json.dumps(data)
        log_line = f"{timestamp}\t{step}\t{data_str}\n"
        
        try:
            with open(self.log_path, 'a') as f:
                f.write(log_line)
        except IOError as e:
            if self.print_to_console:
                print(f"Error writing to log file: {e}")
    
    def get_log_path(self) -> str:
        """返回日志文件的路径。"""
        return self.log_path