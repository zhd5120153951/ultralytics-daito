"""
@FileName   :config.py
@Description:
@Date       :2025/02/10 11:30:55
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :LongRuan@copyright
@version    :1.0
@Email      :2462491568@qq.com.
"""

# Redis连接相关配置
redis_ip = "192.168.1.134"
redis_port = 6379
redis_pwd = ""
redis_db = 6  # redis数据库索引
# 训练服务器IP
IP = "192.168.1.184"
# 数据集配置
minio_endpoint = "192.168.1.134:9000"
minio_access_key = "minioadmin"
minio_secret_key = "minioadmin"
# 训练的Redis消息流相关配置
train_action_opt_topic_name = "AI_TRAIN_TASK_ACTION_OPT"  # 训练任务下发的消息流
train_action_result_topic_name = "AI_TRAIN_TASK_ACTION_RESULT"  # 训练任务返回的消息流
# 导出的Redis消息流相关配置
export_action_opt_topic_name = "AI_EXPORT_TASK_ACTION_OPT"  # 导出任务下发的消息流
export_action_result_topic_name = "AI_EXPORT_TASK_ACTION_RESULT"  # 导出任务返回的消息流
# 增强的Redis消息流相关配置
enhance_action_opt_topic_name = "AI_ENHANCE_TASK_ACTION_OPT"  # 增强任务下发的消息流
enhance_action_result_topic_name = "AI_ENHANCE_TASK_ACTION_RESULT"  # 增强任务返回的消息流
# 支持训练的网络类型
support_net_type = [
    # "yolov3",#性能太差
    "yolov5n", "yolov5s", "yolov5m", "yolov5l",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l",
    # "yolov9",#仅有det+seg
    # "yolov10n", "yolov10s", "yolov10m", "yolov10l",#推理不兼容其他版本
    "yolov11n", "yolov11s", "yolov11m", "yolov11l",
    "yolov12n", "yolov12s", "yolov12m", "yolov12l"
]
# 支持导出的模型格式
support_export_type = ["paddle", "onnx", "torchscript", "tensorrt", "rknn"]
# 数据集路径配置目录
data_cfg = "data_cfg"
# 预训练模型目录
pretrained_models = "pretrained_models"
# 增强结果目录
enhance_result = "enhance_results"
# 增强图像下载保存目录
local_data_dir = "origin_data"
# 是否保存增强结果
save_enhance_result = True
# 训练结果目录
train_result = "train_results"
# 导出结果目录
export_result = "export_results"
# 训练导出的minio目录
minio_bucket_prefix = "train"
minio_data_prefix = "datasets"
minio_train_prefix = "train_result_package"
minio_export_prefix = "export_result_package"
# 数据增强的minio目录
minio_enhance_bucket_prefix = "train"  # 数据增强上传到train桶
minio_enhance_data_prefix = "enhance_result_package"  # 增强结果上传到train桶
# 日志保存目录
logs = "logs"
# 异步上传文件的线程池最大工作线程数
max_workers = 10
# 是否开启数据增强服务
enhance_service_enable = True
