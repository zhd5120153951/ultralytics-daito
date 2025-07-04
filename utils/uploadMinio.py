'''
@FileName   :uploadMinio.py
@Description:
@Date       :2025/02/21 16:31:54
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import time
import asyncio
from threading import Thread
from minio import Minio
from redis import Redis
from minio.error import S3Error
from concurrent.futures import ThreadPoolExecutor

from .logModels import Logger


class uploadMinio(Thread):
    '''
    1.通用的文件上传类
    2.训练时采用这个类上传训练结果,导出时因为耗时短,所以直接在导出服务中上传
    '''

    def __init__(self, rds: Redis, minio_client: Minio, bucket: str, result: str, taskId: str, minio_prefix: str, uploadType: str, modelId, log: Logger, max_workers: int = 10):
        """
        初始化uploadMinio类
        :param minio_client:    已初始化的Minio客户端
        :param bucket:          上传目标桶名称
        :param result:          本地训练的目录
        :param taskId:          训练任务的id
        :param minio_prefix:    minio的prefix
        :param uploadType:      minio上传类型,训练结果上传,导出结果上传
        :param modelId:         导出时采用到参数,用于找到导出的模型
        :param log:             日志
        :param max_workers:     线程池最大工作线程数
        """
        super().__init__()
        self.rds = rds
        self.process_key = taskId
        self.train_task_status_topic_name = f"{taskId}_train_status_info"
        self.max_retry_count = 30  # 最大重试次数
        self.cur_retry_count = 0  # 当前重试次数
        self.minio_client = minio_client
        self.bucket = bucket
        # **********************
        # 之前的思想是判断训练，导出过程的流存不存在？不存在--上传，存在--等待
        # 现在的思想是判断训练，导出进程存不存在？不存在--上传，存在--等待
        if uploadType == 'train':
            # self.rds_name = f'{taskId}_train_progress'  # 训练时的消息流名
            # train--本地待上传的文件夹路径
            self.local_folder = f'{result}/{taskId}'
        else:
            # self.rds_name = f'{taskId}_export_progress'  # 导出时的消息流名
            # export--本地待上传的文件夹路径
            self.local_folder = f'{result}/{modelId}_paddle_model'
        # **********************
        # 去除末尾的'/',方便后续路径拼接
        # 桶内目标目录前缀(上传后所有文件存放在该目录下)
        self.remote_prefix = f'{minio_prefix}/{taskId}'.rstrip('/')
        self.log = log
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _get_all_files(self):
        """
        遍历本地文件夹及其子目录,返回一个包含所有文件完整路径和相对路径的列表
        :return: [(full_path, relative_path), ...]
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
        """
        异步上传单个文件到Minio桶中
        :param full_path: 本地文件的完整路径
        :param rel_path: 文件相对于本地根目录的相对路径
        """
        # 构造远程对象名:remote_prefix+'/'+rel_path
        remote_key = f'{self.remote_prefix}/{rel_path}'
        # 把系统路径符转换为'/'--minio统一使用/
        remote_key = remote_key.replace('\\', '/')
        loop = asyncio.get_event_loop()
        self.log.logger.info(f'开始上传:{full_path}->{remote_key}')
        try:
            # 在线程池中调用同步函数fput_object()上传文件
            await loop.run_in_executor(self.executor, self.minio_client.fput_object, self.bucket, remote_key, full_path)
            self.log.logger.info(f'上传成功:{full_path}')
        except S3Error as s3e:
            self.log.logger.error(f'上传文件{full_path}出错,错误:{s3e}')

    async def upload_all_files(self):
        """
        异步任务:并发上传所有文件
        """
        tasks = []
        files = self._get_all_files()
        for full_path, rel_path in files:
            tasks.append(self.upload_file(full_path, rel_path))
        # 并发执行所有上传任务
        await asyncio.gather(*tasks)

    def _decode_redis_hash(self, redis_hash):
        """
        解码Redis hash数据，处理字节类型的键值对
        :param redis_hash: Redis hgetall返回的字典
        :return: 解码后的字典
        """
        if not redis_hash:
            return {}

        decoded_hash = {}
        for key, value in redis_hash.items():
            # 处理键的解码
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            # 处理值的解码
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            decoded_hash[key] = value
        return decoded_hash

    def _validate_status_info(self, status_info):
        """
        验证状态信息的完整性
        :param status_info: 状态信息字典
        :return: (is_valid, status, context)
        """
        if not status_info:
            return False, None, "状态信息为空"

        if 'status' not in status_info:
            return False, None, "缺少status字段"

        status = str(status_info['status']).strip()
        context = status_info.get('context', '未知状态')

        # 验证status值的有效性
        valid_statuses = ['-1', '0', '1', '2']
        if status not in valid_statuses:
            return False, status, f"无效的状态值: {status}"

        return True, status, context

    def run(self):
        """
        重写Thread的run方法,启动asyncio事件循环执行所有上传任务
        """
        time.sleep(5)
        self.log.logger.info(f'开始监控训练任务{self.process_key}的训练状态!')

        while True:
            try:
                # 检查重试次数
                if self.cur_retry_count >= self.max_retry_count:
                    self.log.logger.error(
                        f'任务{self.process_key}超过最大重试次数({self.max_retry_count}),停止监控,上传失败!')
                    break

                # 获取Redis训练状态的hash数据
                ret = self.rds.hgetall(self.train_task_status_topic_name)

                # 解码Redis数据
                decoded_ret = self._decode_redis_hash(ret)

                # 验证状态信息
                is_valid, status, context = self._validate_status_info(
                    decoded_ret)

                if not is_valid:
                    self.cur_retry_count += 1
                    self.log.logger.warning(
                        f'训练任务{self.process_key}状态信息无效:{context},重试次数:{self.cur_retry_count}/{self.max_retry_count}')
                    time.sleep(3)
                    continue

                # 重置重试计数器（成功获取到有效状态）
                self.cur_retry_count = 0

                # 处理不同的状态
                if status == '-1':
                    self.log.logger.info(
                        f'训练失败,任务{self.process_key}未上传,状态详情:{context}')
                    break
                elif status == '2':
                    self.log.logger.info(
                        f'训练完成,开始上传任务{self.process_key},状态详情:{context}')
                    try:
                        asyncio.run(self.upload_all_files())
                        self.log.logger.info(f'上传任务{self.process_key}成功完成!')
                    except Exception as ex:
                        self.log.logger.error(f'上传过程发生错误:{ex}')
                    break
                elif status == '0':
                    self.log.logger.info(
                        f'任务{self.process_key}数据集下载中,等待训练开始!')
                elif status == '1':
                    self.log.logger.info(f'任务{self.process_key}训练中,等待训练完成!')
                else:
                    self.log.logger.warning(
                        f'任务{self.process_key}未知状态:{status},继续等待!')
                time.sleep(3)

            except Exception as ex:
                self.cur_retry_count += 1
                self.log.logger.error(
                    f'监控任务{self.process_key}状态时发生异常:{ex},重试次数:{self.cur_retry_count}/{self.max_retry_count}')
                time.sleep(5)  # 异常时等待更长时间
                continue

        # 清理Redis状态信息
        try:
            self.rds.delete(self.train_task_status_topic_name)
            self.log.logger.info(f'已删除任务{self.process_key}的状态hash信息!')
        except Exception as ex:
            self.log.logger.error(f'删除任务{self.process_key}的状态hash信息时发生错误:{ex}')
