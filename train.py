# 此文件由程序自动生成.
# 请勿手动修改.
import os
import shutil
from ultralytics import YOLO
from redis import Redis
model_cfg = 'pretrained_models/yolov5nu.yaml'
model_path = 'pretrained_models/yolov5nu.pt'
train_params={'data': 'data_cfg/train_003_猫狗模型训练.yaml', 'project': 'train_results', 'name': 'train_003_猫狗模型训练', 'task': 'detect', 'epochs': 30, 'batch': 8, 'imgsz': 640, 'device': '0', 'workers': 0, 'exist_ok': True, 'resume': False}
rds=Redis(host='192.168.1.184',port=6379,db=1)
model = YOLO(model_cfg).load(rds,model_path)
model.train(**train_params)
model_trained_path = 'train_results/train_003_猫狗模型训练/weights/best.pt'
if not os.path.exists('export_results'):
    os.makedirs('export_results')
model_rename = 'train_003_猫狗模型训练'
model_rename = ''.join([model_rename,'.pt'])
model_rename_path = '/'.join(['export_results',model_rename])
shutil.copy(model_trained_path,model_rename_path)
