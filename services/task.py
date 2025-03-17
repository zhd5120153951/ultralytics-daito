'''
@FileName   :task.py
@Description:
@Date       :2025/03/12 13:31:22
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import json
import shutil
import time
import datetime
import asyncio
from minio.error import S3Error
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from utils import find_pt
from config import (data_cfg,
                    pretrained_models,
                    train_result,
                    export_result,
                    support_net_type,
                    support_export_type,
                    train_result_topic_name,
                    export_result_topic_name)


class TrainTask:
    """_summary_
    训练任务类:负责单个训练任务的执行和管理
    """

    def __init__(self, rds, task_id, task_name, net_type, train_type, model_id, model_type, parameters, labels, log):
        self.rds = rds
        self.task_id = task_id
        self.task_name = task_name
        self.net_type = net_type
        self.train_type = train_type
        self.model_id = model_id
        self.model_type = model_type
        self.parameters = parameters
        self.labels = labels
        self.log = log
        self.process_key = f"{task_id}_{task_name}"  # 进程名:任务id_任务名称

    def regnix_format(self):
        re_list = []
        r1 = r"(\d+)/"
        re_list.append(f"{r1}{self.paramters['epochs']}")
        r1 = r"^      Epoch"
        re_list.append(f"{r1}")
        r1 = r"^                 Class"
        re_list.append(f"{r1}")
        r1 = r"^                   all"
        re_list.append(f"{r1}")
        for i in range(len(self.labels)):
            re_list.append(f'^                   {self.labels[i]}')
        re_format = "|".join(re_list)
        print(f"re_format:{re_format}")
        return re_format

    def start_train_task(self):
        """_summary_
        启动训练任务
        """
        try:
            self.rds.hset(f'{self.process_key}_train_status_info', mapping={
                "status": 1, "context": "训练中"})
            train_params = {
                "data": f"{data_cfg}/{self.process_key}.yaml",
                "project": train_result,
                "name": self.process_key,
                "task": self.model_type,  # 模型类型是det,cls,seg,obb,pose
            }  # 训练参数--根据传递参数生成的部分
            for key, value in self.parameters.items():
                train_params[key] = value  # 训练参数--传递参数固定部分
            if self.net_type in support_net_type:
                if self.train_type == "Init":  # 初始化
                    if self.net_type.startswith("yolov5"):
                        model_cfg = f"{pretrained_models}/{self.net_type}u.yaml"
                        model_path = f"{pretrained_models}/{self.net_type}u.pt"
                    else:
                        model_cfg = f"{pretrained_models}/{self.net_type}.yaml"
                        model_path = f"{pretrained_models}/{self.net_type}.pt"
                else:  # 迭代--modelId是上一次训练的taskId_taskName
                    iter_pre_model = find_pt(export_result, self.model_id)
                    if not iter_pre_model:
                        raise FileNotFoundError(
                            f"找不到迭代训练所需的模型文件:{self.model_id}")
                    if self.net_type.startswith("yolov5"):
                        model_cfg = f"{pretrained_models}/{self.net_type}u.yaml"
                        model_path = iter_pre_model
                    else:
                        model_cfg = f"{pretrained_models}/{self.net_type}.yaml"
                        model_path = iter_pre_model
            else:
                error_msg = f"不支持的网络类型:{self.net_type},目前支持的网络类型有:{support_net_type}"
                self.log.logger.error(error_msg)
                return
            # 加载模型
            self.log.logger.info(f"开始加载模型:{model_cfg},权重路径:{model_path}")
            model = YOLO(model_cfg).load(
                self.rds,
                self.task_id,
                self.task_name,
                model_path)
            # 开始训练
            self.log.logger.info(f"开始训练模型,训练参数:{train_params}")
            start_time = time.time()
            model.train(**train_params)
            # 训练完成，复制模型文件
            model_trained_path = f"{train_result}/{self.process_key}/weights/best.pt"
            model_rename_path = f"{export_result}/{self.process_key}.pt"
            if not os.path.exists(model_trained_path):
                raise FileNotFoundError(f"训练完成后找不到模型文件:{model_trained_path}")
            shutil.copy(model_trained_path, model_rename_path)
            success_msg = f"训练任务:{self.process_key}完成,耗时:{(time.time()-start_time):.3f}s"
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            task_result = {
                "result": "train_success",
                "taskId": self.task_id,
                "taskName": self.task_name,
                "exeMsg": success_msg,
                "exeTime": create_time
            }
            if hasattr(self, "rds"):
                self.rds.xadd(train_result_topic_name, {
                    "result": json.dumps(task_result).encode()}, maxlen=100)
            self.log.logger.info(success_msg)
            # 训练完成,马上删除状态流,开始上传minio
            self.rds.hdel(f"{self.process_key}_train_status_info",
                          ("status", "context"))
        except Exception as ex:
            error_msg = f"训练任务:{self.process_key}失败,异常信息:{ex}"
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            task_result = {
                "result": "train_error",
                "taskId": self.task_id,
                "taskName": self.task_name,
                "exeMsg": error_msg,
                "exeTime": create_time
            }
            if hasattr(self, "rds"):
                self.rds.xadd(train_result_topic_name, {
                    "result": json.dumps(task_result).encode()}, maxlen=100)
            self.log.logger.error(error_msg)
            # 训练失败,也要删除,并且还是上传(文件不全),但可以辅助查看训练失败的原因
            self.rds.hdel(
                f"{self.process_key}_train_status_info", ("status", "context"))

    def stop_train_task(self):
        """_summary_
        停止训练任务
        """
        pass

    def get_train_task_status(self):
        pass

    def get_train_task_result(self):
        pass

    def get_train_task_log(self):
        pass


class ExportTask:
    """_summary_
    导出任务类:负责单个导出任务的执行和管理--导出+上传(训练是仅训练,导出由uploadMinio完成)
    """

    def __init__(self, task_id, task_name, model_id, export_type, minio_client, bucket, minio_prefix, max_workers, log, rds=None):
        self.task_id = task_id
        self.task_name = task_name
        self.model_id = model_id
        self.export_type = export_type
        self.minio_client = minio_client
        self.bucket = bucket
        self.log = log
        self.rds = rds  # 添加Redis客户端，用于状态更新和消息发送
        self.process_key = f"{task_id}_{task_name}"  # 进程名:任务id_任务名称
        self.local_folder = f"{export_result}/{model_id}_paddle_model"
        self.remote_prefix = f"{minio_prefix}/{task_id}_{task_name}".rstrip(
            '/')
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _get_all_files(self):
        """_summary_
        遍历本地文件夹及其子目录,返回一个包含所有文件完整路径和相对路径的列表
        return: [(full_path, relative_path), ...]
        """
        files_list = []
        for root, _, files in os.walk(self.local_folder):
            for file in files:
                full_path = os.path.join(root, file)
                # 计算相对于local_folder的相对路径
                rel_path = os.path.relpath(full_path, self.local_folder)
                files_list.append((full_path, rel_path))
        return files_list

    async def upload_file(self, full_path: str, rel_path: str):
        """_summary_

        Args:
            full_path (str): 本地文件的完整路径
            rel_path (str): 文件相对于本地根目录的相对路径
        """
        remote_key = f"{self.remote_prefix}/{rel_path}".replace("\\", "/")
        loop = asyncio.get_event_loop()
        self.log.logger.info(f"开始上传文件:{full_path}到minio:{remote_key}")
        try:
            await loop.run_in_executor(self.executor, self.minio_client.fput_object, self.bucket, remote_key, full_path)
            self.log.logger.info(f"文件:{full_path}上传成功")
        except S3Error as s3e:
            self.log.logger.error(f"文件:{full_path}上传失败:{s3e}")

    async def upload_all_files(self):  # 异步函数
        """异步任务:上传所有文件到minio"""
        tasks = []
        files = self._get_all_files()
        for full_path, rel_path in files:
            tasks.append(self.upload_file(full_path, rel_path))
        await asyncio.gather(*tasks)  # 并发执行所有上传任务,此处会阻塞直到所有任务完成

    def start_export_task(self):
        """_summary_
        启动导出任务
        return True--成功,False--失败
3        """
        # 启动任务
        try:
            model_path = f"{export_result}/{self.model_id}.pt"
            imgsz = [640, 640]  # 这里导出模型输入尺寸可以由web给出,通用的是640,其他的1280
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"找不到要导出的模型文件:{model_path}")
            if self.export_type in support_export_type:
                self.log.logger.info(
                    f"开始导出模型:{model_path},格式:{self.export_type}")
                model = YOLO(model_path)
                start_time = time.time()
                model.export(format=self.export_type,
                             imgsz=imgsz,
                             opset=12,
                             device="0")
                # 上传导出结果
                asyncio.run(self.upload_all_files())
                success_msg = f"导出上传任务:{self.process_key}完成,耗时:{(time.time()-start_time):.3f}s"
                self.log.logger.info(success_msg)
                create_time = datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S.%f')[:-3]
                task_result = {
                    "result": "export_success",
                    "taskId": self.task_id,
                    "taskName": self.task_name,
                    "exeMsg": success_msg,
                    "exeTime": create_time
                }
                if hasattr(self, "rds"):
                    self.rds.xadd(export_result_topic_name, {
                                  "result": json.dumps(task_result).encode()}, maxlen=100)
                return True

            else:
                error_msg = f"不支持的导出类型:{self.export_type},目前支持的导出类型有:{support_export_type}"
                self.log.logger.error(error_msg)
                raise "不支持的导出类型..."
        except Exception as ex:
            error_msg = f"导出上传任务:{self.process_key}失败,异常信息:{ex}"
            self.log.logger.error(error_msg)
            # 上传失败信息到结果消息流
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            task_result = {
                "result": "export_error",
                "taskId": self.task_id,
                "taskName": self.task_name,
                "exeMsg": error_msg,
                "exeTime": create_time
            }
            if hasattr(self, 'rds'):
                self.rds.xadd(export_result_topic_name, {
                    'result': json.dumps(task_result).encode()}, maxlen=100)
            # 确保进程能够正常退出
            return False

    def stop_export_task(self):
        """_summary_
        停止导出任务--单进程且导出时长很短,此功能暂时不添加
        """
        pass

    def get_export_task_status(self):
        pass

    def get_export_task_result(self):
        pass

    def get_export_task_log(self):
        pass
