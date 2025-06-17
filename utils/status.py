'''
@FileName   :status.py
@Description:
@Date       :2025/06/16 15:47:08
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
from enum import Enum


class TrainStatusType(str, Enum):
    """训练状态枚举"""
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    KILLED = "KILLED"


class ExportStatusType(str, Enum):
    """导出状态枚举"""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
