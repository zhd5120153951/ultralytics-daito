import os
import re
import json
import subprocess
import psutil
import datetime
from redis import Redis
from config import (train_result_topic_name,
                    data_cfg,
                    pretrained_models,
                    train_result)


def file_save(fileDir: str, fileName: str, file):
    if not os.path.exists(fileDir):
        os.makedirs(fileDir, exist_ok=True)
    filePath = os.path.join(fileDir, fileName)
    try:
        with open(filePath, 'wb') as f:
            while True:
                chunk = file.read(1024)
                if not chunk:  # 不为空
                    break
                f.write(chunk)
    except Exception as ex:
        print(f"exception:{ex}")


def handle_pipe(pipe, epoch_pattern, epoch, log_file):
    if pipe is None:
        print("PIPE is null.......")
    else:
        for line in iter(pipe.readline, ''):
            ln = line.strip()

            match = epoch_pattern.search(ln)
            if match:
                current_epoch = int(match.group(1))
                total_epochs = int(match.group(2))
                epoch = f"Current Epoch:{current_epoch}/{total_epochs}"
                print(epoch)
                log_file.write(ln)


def find_pt(src_dir, target_name: str):
    """
    功能:遍历指定目录及其子目录,查找目标.pt文件

    参数:
      src_dir:需要遍历的根目录路径。
      target_name:要查找的.pt文件的名称,可以带或不带.pt后缀

    返回:
      如果找到目标文件,返回该文件的完整路径;如果未找到,则返回 None。
    """
    if not target_name.endswith('.pt'):
        target_name = ''.join([target_name, '.pt'])
    # 使用os.walk遍历目录及子目录
    for root, _, files in os.walk(src_dir):
        for file in files:
            # 检查文件是否和目标匹配
            if file.endswith(target_name):
                return os.path.join(root, file)
    return None  # 没有匹配到,返回None
