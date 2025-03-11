"""
@FileName   :train.py
@Description:
@Date       :2025/03/05 14:58:08
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com.
"""

import redis

from ultralytics import YOLO

rds = redis.Redis("192.168.1.184", 6379, 1)


model_cfg = "pretrained_models/yolov8s.yaml"
model_path = "pretrained_models/yolov8s.pt"
train_params = {
    "data": "data_cfg/train_001_猫狗模型训练.yaml",
    "project": "train_results",
    "name": "train_001_猫狗模型训练",
    "task": "detect",
    "epochs": 30,
    "batch": 4,
    "imgsz": 640,
    "device": "0",
    "workers": 0,
    "exist_ok": True,
    "resume": False,
}
model = YOLO(model_cfg).load(rds, model_path)
model.train(**train_params)
