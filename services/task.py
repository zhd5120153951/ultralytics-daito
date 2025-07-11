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
import cv2
import asyncio
import imgaug as ia
import imgaug.augmenters as iaa
from minio.error import S3Error
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from utils import find_pt, TrainStatusType, ExportStatusType, EnhanceStatusType
from config import (data_cfg,
                    pretrained_models,
                    train_result,
                    export_result,
                    enhance_result,
                    save_enhance_result,
                    support_net_type,
                    support_export_type,
                    train_action_result_topic_name,
                    export_action_result_topic_name,
                    enhance_action_result_topic_name)

# 训练任务类


class TrainTask:
    """_summary_
    训练任务类:负责单个训练任务的执行和管理
    """

    def __init__(self, rds, task_id, net_type, train_type, model_id, model_type, parameters, labels, log):
        self.rds = rds
        self.task_id = task_id
        self.net_type = net_type
        self.train_type = train_type
        self.model_id = model_id
        self.model_type = model_type
        self.parameters = parameters
        self.labels = labels
        self.log = log

    def start_train_task(self):
        """_summary_
        启动训练任务
        """
        task_result = {}
        try:
            # 更新训练状态为训练中
            status_data = {
                "status": "1",
                "context": "训练中",
                "taskId": self.task_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            self.rds.hset(f'{self.task_id}_train_status_info',
                          mapping=status_data)
            self.log.logger.info(f"已更新任务{self.task_id}状态为训练中:{status_data}")
            train_params = {
                "data": f"{data_cfg}/{self.task_id}.yaml",
                "project": train_result,
                "name": self.task_id,
                "task": self.model_type,  # 模型类型是det,cls,seg,obb,pose
                "exist_ok": True
            }  # 训练参数--根据传递参数生成的部分
            for key, value in self.parameters.items():
                train_params[key] = value  # 训练参数--传递参数固定部分
            if self.net_type in support_net_type:
                if self.train_type == "INIT":  # 初始化
                    if self.net_type.startswith("yolov5"):
                        model_cfg = f"{pretrained_models}/{self.net_type}u.yaml"
                        model_path = f"{pretrained_models}/{self.net_type}u.pt"
                    else:
                        model_cfg = f"{pretrained_models}/{self.net_type}.yaml"
                        model_path = f"{pretrained_models}/{self.net_type}.pt"
                else:  # 迭代--modelId是上一次训练的taskId
                    iter_pre_model = find_pt(export_result, self.model_id)
                    if not iter_pre_model:
                        task_result = {
                            "taskId": self.task_id,
                            "status": TrainStatusType.FAILED,
                            "message": f"找不到迭代训练所需的模型文件:{self.model_id}"
                        }
                        self.log.logger.error(
                            f"找不到迭代训练所需的模型文件:{self.model_id}")
                        self.rds.xadd(train_action_result_topic_name, {
                                      "trainResult": json.dumps(task_result).encode()}, maxlen=100)
                        # 更新状态为训练失败
                        status_data = {
                            "status": "-1",
                            "context": "训练失败",
                            "taskId": self.task_id,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        }
                        self.rds.hset(
                            f'{self.task_id}_train_status_info', mapping=status_data)
                        self.log.logger.info(
                            f"已更新任务{self.task_id}状态为训练失败:{status_data}")
                        return
                    if self.net_type.startswith("yolov5"):
                        model_cfg = f"{pretrained_models}/{self.net_type}u.yaml"
                        model_path = iter_pre_model
                    else:
                        model_cfg = f"{pretrained_models}/{self.net_type}.yaml"
                        model_path = iter_pre_model
            else:
                error_msg = f"不支持的网络类型:{self.net_type},目前支持的网络类型有:{support_net_type}"
                self.log.logger.error(error_msg)
                task_result = {
                    "taskId": self.task_id,
                    "status": TrainStatusType.FAILED,
                    "message": error_msg
                }
                self.rds.xadd(train_action_result_topic_name, {
                    "trainResult": json.dumps(task_result).encode()}, maxlen=100)
                # 更新状态为训练失败
                status_data = {
                    "status": "-1",
                    "context": "训练失败",
                    "taskId": self.task_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                }
                self.rds.hset(
                    f'{self.task_id}_train_status_info', mapping=status_data)
                self.log.logger.info(
                    f"已更新任务{self.task_id}状态为训练失败:{status_data}")
                return
            # 加载模型
            self.log.logger.info(f"开始加载模型:{model_cfg},权重路径:{model_path}")
            model = YOLO(model_cfg).load(self.rds, self.task_id, model_path)
            # 开始训练
            self.log.logger.info(f"开始训练模型,训练参数:{train_params}")
            task_result = {
                "taskId": self.task_id,
                "status": TrainStatusType.RUNNING,
                "message": "开始训练"
            }
            self.rds.xadd(train_action_result_topic_name, {
                "trainResult": json.dumps(task_result).encode()}, maxlen=100)
            start_time = time.time()
            model.train(**train_params)
            # 训练完成，复制模型文件
            model_trained_path = f"{train_result}/{self.task_id}/weights/best.pt"
            model_rename_path = f"{export_result}/{self.task_id}.pt"
            if not os.path.exists(model_trained_path):
                task_result = {
                    "taskId": self.task_id,
                    "status": TrainStatusType.FAILED,
                    "message": f"训练完成后找不到模型文件:{model_trained_path}"
                }
                self.log.logger.error(f"训练完成后找不到模型文件:{model_trained_path}")
                self.rds.xadd(train_action_result_topic_name, {
                    "trainResult": json.dumps(task_result).encode()}, maxlen=100)
                # 更新状态为训练失败
                status_data = {
                    "status": "-1",
                    "context": "训练失败",
                    "taskId": self.task_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                }
                self.rds.hset(
                    f'{self.task_id}_train_status_info', mapping=status_data)
                self.log.logger.info(
                    f"已更新任务{self.task_id}状态为训练失败:{status_data}")
                return
            # 保存一份训练好的pt模型--导出用
            shutil.copy(model_trained_path, model_rename_path)
            success_msg = f"训练任务:{self.task_id}完成,耗时:{(time.time()-start_time):.3f}s"
            task_result = {
                "taskId": self.task_id,
                "status": TrainStatusType.SUCCESS,
                "message": success_msg
            }
            if hasattr(self, "rds"):
                self.rds.xadd(train_action_result_topic_name, {
                    "trainResult": json.dumps(task_result).encode()}, maxlen=100)
            self.log.logger.info(success_msg)
            # 训练完成,更新状态,开始上传minio
            status_data = {
                "status": "2",
                "context": "训练完成",
                "taskId": self.task_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            self.rds.hset(f'{self.task_id}_train_status_info',
                          mapping=status_data)
            self.log.logger.info(f"已更新任务{self.task_id}状态为训练完成:{status_data}")
        except Exception as ex:
            error_msg = f"训练任务:{self.task_id}失败,异常信息:{ex}"
            task_result = {
                "taskId": self.task_id,
                "status": TrainStatusType.FAILED,
                "message": error_msg
            }
            if hasattr(self, "rds"):
                self.rds.xadd(train_action_result_topic_name, {
                    "trainResult": json.dumps(task_result).encode()}, maxlen=100)
            self.log.logger.error(error_msg)
            # 训练异常
            status_data = {
                "status": "-1",
                "context": "训练失败",
                "taskId": self.task_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            self.rds.hset(f'{self.task_id}_train_status_info',
                          mapping=status_data)
            self.log.logger.info(f"已更新任务{self.task_id}状态为训练失败:{status_data}")

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

# 导出任务类


class ExportTask:
    """_summary_
    导出任务类:负责单个导出任务的执行和管理--导出+上传(训练是仅训练,导出由uploadMinio完成)
    """

    def __init__(self, task_id, model_id, export_type, minio_client, bucket, minio_prefix, max_workers, log, rds=None):
        self.task_id = task_id
        self.model_id = model_id
        self.export_type = export_type
        self.minio_client = minio_client
        self.bucket = bucket
        self.log = log
        self.rds = rds  # 添加Redis客户端，用于状态更新和消息发送
        self.local_folder = f"{export_result}/{model_id}_paddle_model"  # 本地目录
        self.remote_prefix = f"{minio_prefix}/{task_id}".rstrip(
            '/')  # minio目录
        self.export_executor = ThreadPoolExecutor(max_workers=max_workers)

    def _get_all_files_exported(self):
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

    async def upload_file_exported(self, full_path: str, rel_path: str):
        """_summary_

        Args:
            full_path (str): 本地文件的完整路径
            rel_path (str): 文件相对于本地根目录的相对路径
        """
        remote_key = f"{self.remote_prefix}/{rel_path}".replace("\\", "/")
        loop = asyncio.get_event_loop()
        self.log.logger.info(f"开始上传文件:{full_path}到minio:{remote_key}")
        try:
            await loop.run_in_executor(self.export_executor, self.minio_client.fput_object, self.bucket, remote_key, full_path)
            self.log.logger.info(f"文件:{full_path}上传成功")
        except S3Error as s3e:
            self.log.logger.error(f"文件:{full_path}上传失败:{s3e}")

    async def upload_all_files_exported(self):  # 异步函数
        """异步任务:上传所有文件到minio"""
        tasks = []
        files = self._get_all_files_exported()
        for full_path, rel_path in files:
            tasks.append(self.upload_file_exported(full_path, rel_path))
        await asyncio.gather(*tasks)  # 并发执行所有上传任务,此处会阻塞直到所有任务完成

    def _cleanup_executors(self):
        """清理线程池资源"""
        try:
            # 关闭数据增强线程池
            if hasattr(self, 'export_executor') and self.export_executor:
                self.export_executor.shutdown(wait=True)
                self.log.logger.info(f"任务{self.task_id}的上传导出线程池已关闭")
        except Exception as ex:
            self.log.logger.error(f"清理任务{self.task_id}线程池资源异常,错误:{ex}")

    def start_export_task(self):
        """_summary_
        启动导出任务
        return True--成功,False--失败
        """
        task_result = {}
        # 启动任务
        try:
            model_path = f"{export_result}/{self.model_id}.pt"
            imgsz = [640, 640]  # 这里导出模型输入尺寸可以由web给出,通用的是640,其他的1280
            # 检查模型文件是否存在--不存在(导出失败--结束)
            if not os.path.exists(model_path):
                error_msg = f"找不到要导出的模型文件:{model_path}"
                task_result = {
                    "taskId": self.task_id,
                    "status": ExportStatusType.FAILED,
                    "message": error_msg
                }
                self.log.logger.error(error_msg)
                if hasattr(self, "rds"):
                    self.rds.xadd(export_action_result_topic_name, {
                                  "exportResult": json.dumps(task_result).encode()}, maxlen=100)
                return
            # 开始导出
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
                asyncio.run(self.upload_all_files_exported())
                success_msg = f"导出任务+上传任务:{self.task_id}完成,耗时:{(time.time()-start_time):.3f}s"
                self.log.logger.info(success_msg)
                task_result = {
                    "taskId": self.task_id,
                    "status": ExportStatusType.SUCCESS,
                    "message": success_msg
                }
                if hasattr(self, "rds"):
                    self.rds.xadd(export_action_result_topic_name, {
                                  "exportResult": json.dumps(task_result).encode()}, maxlen=100)
                # 关闭导出上传线程池
                self._cleanup_executors()
            else:
                error_msg = f"不支持的导出类型:{self.export_type},目前支持的导出类型有:{support_export_type}"
                self.log.logger.error(error_msg)
                task_result = {
                    "taskId": self.task_id,
                    "status": ExportStatusType.FAILED,
                    "message": error_msg
                }
                if hasattr(self, "rds"):
                    self.rds.xadd(export_action_result_topic_name, {
                        "exportResult": json.dumps(task_result).encode()}, maxlen=100)
                # 关闭导出上传线程池
                self._cleanup_executors()
        except Exception as ex:
            error_msg = f"导出上传任务:{self.task_id}失败,异常信息:{ex}"
            self.log.logger.error(error_msg)
            task_result = {
                "taskId": self.task_id,
                "status": ExportStatusType.FAILED,
                "message": error_msg
            }
            if hasattr(self, 'rds'):
                self.rds.xadd(export_action_result_topic_name, {
                    'exportResult': json.dumps(task_result).encode()}, maxlen=100)
            # 关闭导出上传线程池
            self._cleanup_executors()
            # 确保进程能够正常退出

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

# 增强任务类


class EnhanceTask:
    """_summary_
    增强任务类:负责单个增强任务的执行和管理
    """

    def __init__(self, task_id, minio_client, bucket, local_prefix, minio_prefix, data, algoType, max_workers, log, rds=None):
        self.task_id = task_id
        self.minio_client = minio_client
        self.bucket = bucket  # 桶:enhance
        self.local_prefix = local_prefix  # 本地:origin_data
        self.minio_prefix = minio_prefix  # minio:enhance_data
        self.data = data  # list
        self.algoType = algoType
        self.log = log
        self.rds = rds  # 添加Redis客户端，用于状态更新和消息发送
        # 本地存放结果目录enhance_results/taskId
        self.enhance_dir = os.path.join(enhance_result, task_id)
        # 数据增强线程池
        self.enhance_executor = ThreadPoolExecutor(max_workers=max_workers)
        # 初始化imgaug随机种子
        ia.seed(42)

    def _get_all_files_enhanced(self):
        """_summary_
        遍历本地文件夹及其子目录,返回一个包含所有文件完整路径和相对路径的列表
        return: [(full_path, relative_path), ...]
        """
        files_list = []
        for root, _, files in os.walk(self.enhance_dir):
            for file in files:
                full_path = os.path.join(root, file)
                # 计算相对于local_folder的相对路径
                rel_path = os.path.relpath(full_path, self.enhance_dir)
                files_list.append((full_path, rel_path))
        return files_list

    async def upload_file_enhanced(self, full_path: str, rel_path: str, retry_count: int = 3):
        """_summary_

        Args:
            full_path (str): 本地文件的完整路径
            rel_path (str): 文件相对于本地根目录的相对路径
            retry_count (int): 重试次数，默认3次
        """
        remote_key = f"{self.minio_prefix}/{self.task_id}/{rel_path}".replace(
            "\\", "/")
        loop = asyncio.get_event_loop()

        for attempt in range(retry_count):
            try:
                await loop.run_in_executor(self.enhance_executor, self.minio_client.fput_object, self.bucket, remote_key, full_path)
                return True
            except S3Error as s3e:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    self.log.logger.error(
                        f"文件:{full_path}上传失败，已重试{retry_count}次:{s3e}")
                    return False
            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 2
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    self.log.logger.error(
                        f"文件:{full_path}上传异常，已重试{retry_count}次:{e}")
                    return False
        return False

    async def upload_all_files_enhanced(self):  # 异步函数
        """异步任务:上传所有文件到minio"""
        files = self._get_all_files_enhanced()
        total_files = len(files)

        if total_files == 0:
            self.log.logger.info("没有增强文件需要上传")
            return

        self.log.logger.info(f"开始上传{total_files}个增强文件到MinIO")

        # 分批上传，每批最多100个文件
        batch_size = min(100, max(10, total_files // 4))  # 动态调整批次大小
        success_count = 0
        failed_count = 0

        for i in range(0, total_files, batch_size):
            batch_files = files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_files + batch_size - 1) // batch_size

            self.log.logger.info(
                f"正在处理第{batch_num}/{total_batches}批增强文件，包含{len(batch_files)}个文件")

            # 创建当前批次的上传任务
            tasks = []
            for full_path, rel_path in batch_files:
                tasks.append(self.upload_file_enhanced(full_path, rel_path))

            # 并发执行当前批次的上传任务
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 统计结果
            batch_success = sum(1 for result in results if result is True)
            batch_failed = len(results) - batch_success
            success_count += batch_success
            failed_count += batch_failed

            self.log.logger.info(
                f"第{batch_num}批增强文件完成: 成功{batch_success}个,失败{batch_failed}个")

            # 批次间短暂休息，避免资源过度占用
            if i + batch_size < total_files:
                await asyncio.sleep(0.5)

        self.log.logger.info(
            f"增强文件上传完成: 总计{total_files}个，成功{success_count}个，失败{failed_count}个")

        if failed_count > 0:
            self.log.logger.warning(
                f"有{failed_count}个增强文件上传失败，请检查网络连接和MinIO服务状态")

        return success_count, failed_count

    def _get_image_files_from_origin(self):
        """获取所有下载的图像文件"""
        image_files = []
        supported_formats = ('.jpg', '.jpeg', '.png')

        for data_name in self.data:
            origin_path = os.path.join(self.local_prefix, data_name)
            if os.path.exists(origin_path):
                for root, dirs, files in os.walk(origin_path):
                    for file in files:
                        if file.lower().endswith(supported_formats):
                            full_path = os.path.join(root, file)
                            # 计算相对于origin_path的相对路径
                            rel_path = os.path.relpath(full_path, origin_path)
                            image_files.append(
                                (full_path, rel_path, data_name))
        return image_files

    def _create_augmentation_pipeline(self, algo_type: dict):
        """根据algoType创建imgaug增强管道"""
        augmenters = []
        try:
            ######## 几何变换#######
            # 1.翻转
            flip = algo_type.get('flip', False)
            if flip:  # 默认不翻转
                # 水平翻转
                augmenters.append(iaa.Fliplr(flip.get('flip_x', 0)))
                # 垂直翻转
                augmenters.append(iaa.Flipud(flip.get('flip_y', 0)))

            # 2.旋转
            rotation = algo_type.get('rotation', False)
            if rotation:  # 默认不旋转
                angle = rotation.get('rotation_angle', 0)
                # 顺时针>0,逆时针<0,[-360,360]
                augmenters.append(iaa.Rotate(angle))

            # 3.平移
            translate = algo_type.get('translate', False)
            if translate:  # 默认不平移
                x_percent = translate.get('translate_x_percent', 0)
                y_percent = translate.get('translate_y_percent', 0)
                # 水平,[-1,1]
                augmenters.append(iaa.TranslateX(
                    percent=(-x_percent, x_percent)))
                # 垂直,[-1,1]
                augmenters.append(iaa.TranslateY(
                    percent=(-y_percent, y_percent)))
            # 4.裁剪
            crop = algo_type.get('crop', False)
            if crop:
                percent = crop.get('crop_percent', 0)
                width = crop.get('crop_width', 1024)
                height = crop.get('crop_height', 1024)
                # augmenters.append(iaa.Crop(percent=(0, percent)))#范围代表随机取值
                # 百分比裁剪:[0,1]
                augmenters.append(iaa.Crop(percent=percent))
                # 中心裁剪:width,height
                augmenters.append(iaa.CenterCropToFixedSize(
                    width=width, height=height))

            # 5.缩放
            scale = algo_type.get('scale', False)
            if scale:
                scale_percent = scale.get('scale_percent', 1)
                # 百分比缩放:>1放大,<1缩小
                augmenters.append(iaa.Affine(scale=scale_percent))

            # 6.透视变换
            perspective = algo_type.get('perspective', False)
            if perspective:
                # 透视变换参数
                perspective_range = perspective.get(
                    'perspective_range', 0)
                # 透视变换
                augmenters.append(iaa.PerspectiveTransform(
                    scale=(-perspective_range, perspective_range)))
            ######### 颜色变换##########
            # 7.亮度
            brightness = algo_type.get('brightness', False)
            if brightness:
                factor = brightness.get('brightness_factor', 0)
                # >0增亮,<0变暗
                augmenters.append(iaa.AddToBrightness(factor))
            # 8.对比度
            contrast = algo_type.get('contrast', False)
            if contrast:
                factor = contrast.get('contrast_factor', 1)
                # >1增加对比度,<1减少对比度
                augmenters.append(iaa.LinearContrast(factor))
            # 9.饱和度
            saturation = algo_type.get('saturation', False)
            if saturation:
                factor = saturation.get('saturation_factor', 1)
                # >1增加饱和度,<1减少饱和度
                augmenters.append(iaa.MultiplySaturation(factor))
            # 10.色调
            hue = algo_type.get('hue', False)
            if hue:
                factor = hue.get('hue_factor', 0)
                # [-180,180]
                augmenters.append(iaa.AddToHue(factor))
            # 11.灰度
            gray = algo_type.get('gray', False)
            if gray:
                factor = gray.get('gray_factor', 0)
                # >0灰度化,<0反灰度化
                augmenters.append(iaa.Grayscale(factor))
            # 12.透明度
            alpha = algo_type.get('alpha', False)
            if alpha:
                factor = alpha.get('alpha_factor', 0)
                # >0增加透明度,<0减少透明度
                augmenters.append(iaa.AddToAlpha(factor))
            ########## 质量模拟###########
            # 12.高斯模糊
            gaussian_blur = algo_type.get('gaussian_blur', False)
            if gaussian_blur:
                sigma = gaussian_blur.get('blur_sigma', 0)
                augmenters.append(iaa.GaussianBlur(sigma=sigma))
            # 13.运动模糊
            motion_blur = algo_type.get('motion_blur', False)
            if motion_blur:
                k = motion_blur.get('blur_k', 0)
                angle = motion_blur.get('blur_angle', 0)
                augmenters.append(iaa.MotionBlur(k=k, angle=angle))
            # 14.锐化
            sharpen = algo_type.get('sharpen', False)
            if sharpen:
                # 锐化系数[0,1]
                alpha = sharpen.get('sharpen_alpha', 0)
                # 锐化系数[0,1]
                lightness = sharpen.get('sharpen_lightness', 0)
                augmenters.append(iaa.Sharpen(
                    alpha=alpha, lightness=lightness))
            # 15.高斯噪声
            gaussian_noise = algo_type.get('gaussian_noise', False)
            if gaussian_noise:
                scale = gaussian_noise.get('noise_scale', 0)
                # [0,1]
                augmenters.append(iaa.AdditiveGaussianNoise(scale=scale*255))
            # 16.椒盐噪声
            pepper_noise = algo_type.get('pepper_noise', False)
            if pepper_noise:
                # 椒盐噪声比例[0,1]
                pepper_percent = pepper_noise.get('pepper_percent', 0)
                augmenters.append(iaa.SaltAndPepper(pepper_percent))
            # 17.泊松噪声
            poisson_noise = algo_type.get('poisson_noise', False)
            if poisson_noise:
                # 泊松噪声比例[0,1]
                poisson_percent = poisson_noise.get('poisson_percent', 0)
                augmenters.append(iaa.AdditivePoissonNoise(poisson_percent))
            ########### 高级变换############
            # 18.弹性变换
            elastic_transform = algo_type.get('elastic_transform', False)
            if elastic_transform:
                alpha = elastic_transform.get('elastic_alpha', 50)
                sigma = elastic_transform.get('elastic_sigma', 5)
                augmenters.append(iaa.ElasticTransformation(
                    alpha=alpha, sigma=sigma))
            # 19.直方图均衡化
            hist_eq = algo_type.get('hist_eq', False)
            if hist_eq:
                augmenters.append(iaa.AllChannelsHistogramEqualization())
            # 20.通道抖动
            channel_shuffle = algo_type.get('channel_shuffle', False)
            if channel_shuffle:
                augmenters.append(iaa.ChannelShuffle())
            # 21.遮挡
            cutout = algo_type.get('cutout', False)
            if cutout:
                # 遮挡大小
                cutout_size = cutout.get('cutout_size', 0)
                # 遮挡数量
                cutout_count = cutout.get('cutout_count', 0)
                augmenters.append(iaa.Cutout(
                    size=cutout_size,
                    nb_iterations=cutout_count,
                    squared=False
                ))
        except Exception as ex:
            self.log.logger.error(f"算法管道创建异常,错误:{ex}")
            return

        # 如果没有指定任何增强，使用默认增强
        if not augmenters:
            augmenters = [
                iaa.Rotate((-15, 15)),
                iaa.Fliplr(0.5),
                iaa.Multiply((0.8, 1.2)),
                iaa.GaussianBlur(sigma=(0, 1.0))
            ]

        return iaa.Sequential(augmenters, random_order=True)

    async def _enhance_single_image(self, idx, total_count, image_info, augmentation_pipeline):
        """增强单张图像"""
        full_path, rel_path, data_name = image_info

        try:
            # 读取图像
            image = cv2.imread(full_path)
            if image is None:
                self.log.logger.error(f"无法读取图像:{full_path}")
                return False

            output_dir = os.path.join(self.enhance_dir, data_name)

            # 获取原始文件名和扩展名
            filename = os.path.basename(full_path)
            name, ext = os.path.splitext(filename)

            # # 保存原始图像
            # original_output_path = os.path.join(output_dir, filename)
            # cv2.imwrite(original_output_path, image)
            # self.log.logger.info(f"保存原始图像: {original_output_path}")

            # 生成增强图像
            try:
                # 应用增强
                augmented_image = augmentation_pipeline(image=image)

                # 生成增强图像文件名
                enhanced_filename = f"{name}{ext}"
                enhanced_output_path = os.path.join(output_dir,
                                                    enhanced_filename)

                # 保存增强图像
                cv2.imwrite(enhanced_output_path, augmented_image)
                # self.log.logger.info(f"保存增强图像:{enhanced_output_path}")
                # 发送一条进度消息
                if hasattr(self, 'rds'):
                    task_result = {
                        "taskId": self.task_id,
                        "status": EnhanceStatusType.RUNNING,
                        # "message": f"数据增强任务{self.task_id}运行中,当前处理第{idx}张图像!"
                        "progress": f"{idx+1}/{total_count}"
                    }
                    self.rds.xadd(enhance_action_result_topic_name, {
                        'enhanceResult': json.dumps(task_result).encode()}, maxlen=100)

            except Exception as e:
                self.log.logger.error(
                    f"增强图像{full_path}失败:{str(e)}")

            return True

        except Exception as e:
            self.log.logger.error(f"处理图像{full_path}失败:{str(e)}")
            return False

    async def _enhance_all_images(self, image_files: list, augmentation_pipeline: list):
        """异步增强所有图像"""

        tasks = []
        total_count = len(image_files)  # 总得图像数
        for idx, image_info in enumerate(image_files):  # 传个索引进去,方便统计进度
            task = self._enhance_single_image(
                idx, total_count, image_info, augmentation_pipeline)
            tasks.append(task)

        # 并发执行所有增强任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for result in results if result is True)
        # total_count = len(results)

        self.log.logger.info(
            f"图像增强任务:{self.task_id}完成,成功{success_count}/{total_count}")
        return success_count, total_count

    def _delete_local_files(self):
        """sumary_line
        清理增强后的图像        
        """
        try:
            if os.path.exists(self.enhance_dir):  # 删除到taskId目录
                shutil.rmtree(self.enhance_dir)
                os.makedirs(self.enhance_dir, exist_ok=True)
                self.log.logger.info(f"已删除任务{self.task_id}生成的增强图像!")
        except Exception as ex:
            self.log.logger.error(f"删除任务{self.task_id}生成的图像异常,错误:{ex}")

    def _cleanup_executors(self):
        """清理线程池资源"""
        try:
            # 关闭数据增强线程池
            if hasattr(self, 'enhance_executor') and self.enhance_executor:
                self.enhance_executor.shutdown(wait=True)
                self.log.logger.info(f"任务{self.task_id}的数据增强线程池已关闭")
        except Exception as ex:
            self.log.logger.error(f"清理任务{self.task_id}线程池资源异常,错误:{ex}")

    def start_enhance_task(self):
        """_summary_
        启动增强任务
        """
        task_result = {}
        start_time = time.time()
        try:
            # 上传增强开始状态
            if hasattr(self, 'rds'):
                task_result = {
                    "taskId": self.task_id,
                    "status": EnhanceStatusType.RUNNING,
                    "message": f"数据增强任务{self.task_id}运行中"
                }
                self.rds.xadd(enhance_action_result_topic_name, {
                    'enhanceResult': json.dumps(task_result).encode()}, maxlen=100)

            # 1. 获取下载到本地的图像数据信息
            image_files = self._get_image_files_from_origin()
            if not image_files:
                error_msg = f"在{self.local_prefix}目录中未找到图像文件,数据列表:{self.data}"
                self.log.logger.error(error_msg)
                if hasattr(self, 'rds'):
                    task_result = {
                        "taskId": self.task_id,
                        "status": EnhanceStatusType.FAILED,
                        "message": error_msg
                    }
                    self.rds.xadd(enhance_action_result_topic_name, {
                        'enhanceResult': json.dumps(task_result).encode()}, maxlen=100)
                return

            # 2. 创建增强算法管道
            augmentation_pipeline = self._create_augmentation_pipeline(
                self.algoType)

            # 3. 异步增强处理所有图像
            for data_path in [os.path.join(self.enhance_dir, item) for item in self.data]:
                if not os.path.exists(data_path):
                    os.makedirs(data_path, exist_ok=True)

            success_count, total_count = asyncio.run(
                self._enhance_all_images(image_files, augmentation_pipeline)
            )

            if success_count == 0:
                error_msg = f"所有图像增强失败"
                self.log.logger.error(error_msg)
                if hasattr(self, 'rds'):
                    task_result = {
                        "taskId": self.task_id,
                        "status": EnhanceStatusType.FAILED,
                        "message": error_msg
                    }
                    self.rds.xadd(enhance_action_result_topic_name, {
                        'enhanceResult': json.dumps(task_result).encode()}, maxlen=100)
                return

            # 4. 上传增强结果到MinIO--已优化：分批上传、重试机制、专用线程池
            upload_success, upload_failed = asyncio.run(
                self.upload_all_files_enhanced())

            if upload_failed > 0:
                self.log.logger.warning(
                    f"部分文件上传失败:成功{upload_success}个,失败{upload_failed}个!")

            # 5. 完成任务
            duration = time.time() - start_time
            success_msg = f"增强任务{self.task_id}完成,处理{success_count}/{total_count}张图像,耗时{duration:.2f}秒!"
            self.log.logger.info(success_msg)

            if hasattr(self, 'rds'):
                task_result = {
                    "taskId": self.task_id,
                    "status": EnhanceStatusType.SUCCESS,
                    "message": success_msg
                }
                self.rds.xadd(enhance_action_result_topic_name, {
                    'enhanceResult': json.dumps(task_result).encode()}, maxlen=100)
            # 6.删除本地增强图像
            if not save_enhance_result:
                self._delete_local_files()

            # 7.清理线程池资源
            self._cleanup_executors()
            return
        except Exception as ex:
            duration = time.time() - start_time
            error_msg = f"增强任务{self.task_id}失败:{str(ex)},耗时{duration:.2f}秒!"
            self.log.logger.error(error_msg)

            if hasattr(self, 'rds'):
                task_result = {
                    "taskId": self.task_id,
                    "status": EnhanceStatusType.FAILED,
                    "message": error_msg
                }
                self.rds.xadd(enhance_action_result_topic_name, {
                    'enhanceResult': json.dumps(task_result).encode()}, maxlen=100)

            # 清理线程池资源
            self._cleanup_executors()
            return

    def stop_enhance_task(self):
        """_summary_
        停止导出任务--单进程且导出时长很短,此功能暂时不添加
        """
        pass

    def get_enhance_task_status(self):
        pass

    def get_enhance_task_result(self):
        pass

    def get_enhance_task_log(self):
        pass
