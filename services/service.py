'''
@FileName   :service.py
@Description:
@Date       :2025/03/11 11:29:02
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

from string import ascii_uppercase
import redis
import minio
import time
import datetime
import json
import multiprocessing as mp
from utils import Logger, Dataset, uploadMinio, TrainStatusType
from services.task import TrainTask, ExportTask
from config import (IP,
                    minio_endpoint,
                    minio_access_key,
                    minio_secret_key,
                    data_cfg,
                    train_result,
                    minio_bucket_prefix,
                    minio_data_prefix,
                    minio_train_prefix,
                    minio_export_prefix,
                    max_workers)
from utils.status import ExportStatusType


class BaseService:
    """服务基类，包含共用方法."""

    def __init__(self,
                 redis_ip,
                 redis_port,
                 redis_pwd,
                 redis_db,
                 logs):
        self.redis_pool = redis.ConnectionPool(host=redis_ip,
                                               port=redis_port,
                                               password=redis_pwd,
                                               db=redis_db,
                                               health_check_interval=30)
        self.rds = redis.StrictRedis(
            connection_pool=self.redis_pool, decode_responses=True)
        self.minio_client = minio.Minio(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False)
        self.logs = logs  # 日志目录

    def getAvailableGPUId(self, log: Logger):
        '''
        返回可用的 GPU ID，且返回显存占用最少的GPU ID
        '''
        import pynvml
        import numpy as np
        current_gpu_unit_use = []
        try:
            pynvml.nvmlInit()
            deviceCount = pynvml.nvmlDeviceGetCount()
        except Exception as ex:
            log.logger.error('GPU初始化异常,服务器没有检测到显卡!\terror:', ex)
            return str(-2), str(-2)
        for index in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            use = pynvml.nvmlDeviceGetUtilizationRates(handle)
            if use.memory < 80:
                current_gpu_unit_use.append(use.gpu)
        pynvml.nvmlShutdown()
        log.logger.info(f"当前服务器显卡GPU Usage:{current_gpu_unit_use}")
        if not current_gpu_unit_use:
            return str(-1), str(-1)
        else:
            GPU_ID = np.argmin(current_gpu_unit_use)
            return str(GPU_ID), ''.join([str(current_gpu_unit_use[GPU_ID]), '%'])

    def check_consumer_group_exists(self, stream_name, group_name):
        '''
        获取指定的stream的消费者组列表
        '''
        groups = self.rds.xinfo_groups(stream_name)
        for group in groups:
            if group['name'].decode('utf-8') == group_name:
                return True
        return False


class trainService(BaseService):
    def __init__(self,
                 redis_ip,
                 redis_port,
                 redis_pwd,
                 redis_db,
                 logs,
                 train_action_opt_topic_name,
                 train_action_result_topic_name):
        super().__init__(redis_ip,
                         redis_port,
                         redis_pwd,
                         redis_db,
                         logs)
        # ... 子类的初始化代码 ...
        self.train_action_opt_topic_name = train_action_opt_topic_name
        self.train_action_result_topic_name = train_action_result_topic_name
        self.active_procs = {}

    def add_proc(self, process_key: str, process: mp.Process, log: Logger):
        """_summary_
        添加进程
        """
        self.active_procs[process_key] = process
        log.logger.info(f"当前训练任务:{process_key}进程:{process.pid}添加成功.")

    def remove_proc(self, process_key: str, log: Logger):
        """_summary_
        移除进程
        """
        del self.active_procs[process_key]
        log.logger.info(f"当前训练任务:{process_key}进程移除成功.")

    def get_proc(self, process_key: str, log: Logger):
        """_summary_
        获取进程
        """
        proc = self.active_procs.get(process_key, None)
        if not proc:
            log.logger.info(f"当前训练任务:{process_key}进程不存在.")
            return None
        return proc

    def stop_proc(self, process_key: str, log: Logger):
        """_summary_
        停止进程
        """
        task_proc = self.get_proc(process_key, log)
        if not task_proc:
            log.logger.info(f"当前任务:{process_key}进程不存在,无需停止.")
            return False
        elif task_proc.is_alive():
            task_proc.terminate()
            task_proc.join(5)
            if task_proc.is_alive():
                task_proc.kill()
            log.logger.info(f"当前任务:{process_key}进程成功停止.")
            self.remove_proc(process_key, log)  # 停止后立刻移除taskId对应的进程对象
            return True
        log.logger.info(f"当前任务:{process_key}已训练完成,进程自己终止,无需停止.")
        return False

    def stop_all_procs(self):
        """_summary_
        停止所有进程
        """
        for proc_key in list(self.active_procs.keys()):
            self.remove_proc(proc_key)
            self.stop_proc(proc_key)
        print("All process have been stopped.")

    def lower_en(self, text: str):
        result = ""
        for char in text:
            if char in ascii_uppercase:
                result += char.lower()
            else:
                result += char
        return result

    def get_task_message(self, train_task_action_opt_msg: dict):
        """_summary_
        从消息队列中获取任务消息
        """
        action = train_task_action_opt_msg.get("action", None)
        taskId = train_task_action_opt_msg.get("taskId", None)  # 做为唯一进程名
        if train_task_action_opt_msg.get("trainType", None) == "INIT":
            modelId = None
        else:
            modelId = train_task_action_opt_msg.get("modelId", None)
        netType = train_task_action_opt_msg.get('netType', None)
        modelType = train_task_action_opt_msg.get("modelType", None)
        if modelType == "classification":
            modelType = "classify"
        elif modelType == "detection":
            modelType = "detect"
        elif modelType == "segmentation":
            modelType = "segment"
        prefix = train_task_action_opt_msg.get("datasets", [])  # list
        labels = train_task_action_opt_msg.get("labels", [])  # list
        ratio = train_task_action_opt_msg.get("ratio", 0)
        train_params = train_task_action_opt_msg.get("trainParams", {})
        train_params["workers"] = 0
        return (action, taskId, modelId, self.lower_en(netType), modelType, prefix, labels, ratio, train_params)

    def upload_train_result(self, taskId: str, status: TrainStatusType, message: str):
        '''
        具体上传怎么内容需要和平台协定
        '''
        data = {
            "taskId": taskId,
            "status": status,
            "message": message
        }
        message_id = self.rds.xadd(self.train_action_result_topic_name,
                                   {'trainResult': json.dumps(data).encode()}, maxlen=100)

    def upload_train_result_minio(self, action, taskId, taskName, create_time):
        '''
        上传训练最终结果:打包后的训练内容,往Minio发
        '''
        pass

    def start_train_task(self, taskId, taskName, netType, trainType, modelId, modelType, parameters, labels, log):
        pass

    def start_train_process(self, taskId: str, modelId: str, netType: str, modelType: str, datasets: list, labels: list, ratio: float, parameters: dict):
        """sumary_line
        taskId:每个任务进程的唯一标识
        """
        # 查到有同名任务--重复启动--记录日志后直接退出
        repeat_train_log = Logger(
            f'{self.logs}/repeat_train_log_{taskId}.txt', level='info')
        if self.get_proc(taskId, repeat_train_log):
            repeat_train_log.logger.info(f"训练任务:{taskId}已经启动运行中,无需重复启动！")
            self.upload_train_result(
                taskId, TrainStatusType.FAILED, "训练任务重复启动")
            return
        # 正常启动训练--数据集下载、模型训练、结果上传(异步)
        # 1.数据集并发下载
        downlaod_datasets_log = Logger(
            f"{self.logs}/download_datasets_log_{taskId}.txt", level="info")
        data_yaml_path = f"{data_cfg}/{taskId}.yaml"
        if not modelId:  # 初始化训练
            trainType = "INIT"
        else:  # 迭代训练
            trainType = "ITERATION"
        dataset = Dataset(self.minio_client,
                          minio_data_prefix,  # minio固定的数据集存放目录名
                          minio_bucket_prefix,  # minio训练的桶名
                          datasets,  # minio上数据集目录下的若干数据集名
                          labels,
                          ratio,
                          data_yaml_path,
                          downlaod_datasets_log)
        dataset.start()
        # 设置训练状态--0数据集下载中,1--训练中,2--上传minio中
        self.rds.hset(f'{taskId}_train_status_info', mapping={
            "status": 0, "context": "数据集下载中"})
        dataset.join()  # 这里阻塞,等待数据集的操作全部完成才能训练
        if not dataset.isFinished:
            self.upload_train_result(
                taskId, TrainStatusType.FAILED, "数据集下载失败")
            self.rds.hdel(f"{taskId}_train_status_info", "status", "context")
            return
        # 2.异步训练--classify,detect,obb,segment,pose
        start_train_log = Logger(
            f"{self.logs}/start_train_log_{taskId}.txt", level="info")
        # 每次启动任务都把已经停止的taskId:进程移除
        for k, v in self.active_procs.items():
            if not v.is_alive():
                self.remove_proc(k, start_train_log)
        train_task = TrainTask(
            self.rds,
            taskId,
            netType,
            trainType,
            modelId,
            modelType,
            parameters,
            labels,
            start_train_log)
        p_task = mp.Process(target=train_task.start_train_task)
        p_task.daemon = True
        p_task.start()
        time.sleep(3)
        if p_task.is_alive():
            # 保存当前任务对应的进程:taskId:一个训练进程
            self.add_proc(taskId, p_task, start_train_log)
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            start_train_log.logger.info(
                f"训练任务:{taskId}启动成功,进程id:{p_task.pid},时间:{create_time}")
        else:
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            start_train_log.logger.info(
                f"训练任务:{taskId}启动失败,时间:{create_time}")
            self.upload_train_result(
                taskId, TrainStatusType.FAILED, "训练任务进程启动失败")
            self.rds.hdel(f"{taskId}_train_status_info", "status", "context")
            return
        # 3.上传minio
        upload_train_result_log = Logger(
            f'{self.logs}/upload_train_result_log_{taskId}.txt', level='info')
        upload_minio = uploadMinio(self.rds,
                                   self.minio_client,
                                   minio_bucket_prefix,
                                   train_result,
                                   taskId,
                                   minio_train_prefix,
                                   'train',
                                   # modelId是导出上次的模型名字(taskId),对于训练时,此参数可任意给
                                   "",
                                   upload_train_result_log,
                                   max_workers)
        upload_minio.start()

    def stop_train_process(self, taskId: str):
        """sumary_line
            taskId:做为每个训练进城的唯一key
        """
        stop_train_log = Logger(
            f'{self.logs}/stop_train_log_{taskId}.txt', level='info')
        if not self.stop_proc(taskId, stop_train_log):  # 该任务不存在--无需终止
            stop_train_log.logger.info(f"训练任务{taskId}不存在,无需终止！")
        else:  # 该任务存在--需要终止
            # 同时训练的状态流应该删除
            self.rds.hdel(f"{taskId}_train_status_info", "status", "context")
            self.upload_train_result(
                taskId, TrainStatusType.KILLED, "训练任务被手动终止")
            stop_train_log.logger.info(f"已正常停止正在训练中的任务:{taskId}！")

    def run(self):
        # 消费平台下发的训练任务消息
        train_action_opt_log = Logger(
            f'{self.logs}/train_action_opt.txt', level='info')
        group_name = ''.join(['action_', IP])
        if not self.rds.exists(self.train_action_opt_topic_name) or not self.check_consumer_group_exists(self.train_action_opt_topic_name, group_name):
            self.rds.xgroup_create(
                name=self.train_action_opt_topic_name,
                groupname=group_name,
                id='$', mkstream=True)  # 创建消费者组--消费训练配置消息
        while True:
            try:
                # 检查GPU资源
                availableGPUId, gpu_unit_use = self.getAvailableGPUId(
                    train_action_opt_log)
                if availableGPUId == '-2':
                    train_action_opt_log.logger.info(
                        f'训练任务:{taskId}启动时未检测到可用的GPU,训练任务无法正常启动！')
                    # self.upload_train_result(
                    #     taskId, TrainStatusType.FAILED, "没检测到显卡,训练任务无法正常启动")
                    time.sleep(3)
                    continue
                elif availableGPUId == '-1':
                    train_action_opt_log.logger.error(
                        f'训练任务:{taskId}启动时检测到GPU资源不足,训练任务无法正常启动！')
                    # self.upload_train_result(
                    #     taskId, TrainStatusType.FAILED, "GPU资源不足,训练任务无法正常启动")
                    time.sleep(3)
                    continue
                else:
                    train_action_opt_log.logger.info(
                        f'训练任务:{taskId}启动时检测到可用GPU:{availableGPUId}的利用率最低,且利用率:{gpu_unit_use}')
                response = self.rds.xreadgroup(
                    groupname=group_name,
                    consumername=group_name,
                    streams={self.train_action_opt_topic_name: '>'},
                    block=1000, count=1)
                if response:
                    train_action_opt_log.logger.info(
                        f'训练服务监听到Redis消息...')
                else:
                    # train_action_opt_log.logger.warn('训练服务没监听到Redis消息!')
                    time.sleep(3)
                    continue
            except Exception as ex:
                train_action_opt_log.logger.error(
                    f'Redis监听异常,尝试重连Redis!\t异常信息:{ex}')
                time.sleep(3)
                continue
            try:
                stream_name, message_list = response[0]
                message_id, message_data = message_list[0]
                # 完成消费后,标记消息为已处理,并且任务被执行前删除该消息
                self.rds.execute_command(
                    'XACK', self.train_action_opt_topic_name, group_name, message_id)
                opt_msg = message_data[b"action"].decode()
                train_action_opt_log.logger.info(f'Redis消息内容:{opt_msg}')
                train_task_action_opt_msg = json.loads(opt_msg)
                if train_task_action_opt_msg['ip'] != IP:
                    continue
                self.rds.xdel(self.train_action_opt_topic_name, message_id)
                # 解析训练任务消息
                action, taskId, modelId, netType, modelType, datasets, labels, ratio, parameters = self.get_task_message(
                    train_task_action_opt_msg)
                # 启用模型训练
                if action == 'start':
                    # 用这个可以对应多个训练任务
                    self.start_train_process(
                        taskId,
                        modelId,
                        netType,
                        modelType,
                        datasets,
                        labels,
                        ratio,
                        parameters)
                    time.sleep(3)
                else:  # 停止模型训练--1:训练没结束时停止;2:训练已结束时停止
                    self.stop_train_process(taskId)
            except Exception as ex:
                train_action_opt_log.logger.error(f'训练服务启动发生异常!\terr:{ex}')


class exportService(BaseService):
    def __init__(self,
                 redis_ip,
                 redis_port,
                 redis_pwd,
                 redis_db,
                 logs,
                 export_action_opt_topic_name,
                 export_action_result_topic_name):
        super().__init__(redis_ip,
                         redis_port,
                         redis_pwd,
                         redis_db,
                         logs)
        # ... 子类初始化代码 ...
        self.export_action_opt_topic_name = export_action_opt_topic_name
        self.export_action_result_topic_name = export_action_result_topic_name
        self.export_tasks = set()  # 和训练不同,这里value不是进程对象,而是导出任务id

    def add_task(self, taskId: str, log: Logger):
        """_summary_
        添加任务
        """
        if taskId in self.export_tasks:
            log.logger.info(f"当前任务:{taskId}已存在,无需重复添加!")
            return False
        self.export_tasks.add(taskId)
        log.logger.info(f"当前任务:{taskId}添加成功!")
        return True

    def remove_task(self, taskId: str, log: Logger):
        """_summary_
        移除任务
        """
        if taskId not in self.export_tasks:
            log.logger.info(f"当前任务:{taskId}不存在,无需重复移除!")
            return False
        self.export_tasks.remove(taskId)
        log.logger.info(f"当前任务:{taskId}移除成功!")
        return True

    def has_task(self, taskId: str):
        """sumary_line
        查询任务是否存在
        """
        return taskId in self.export_tasks

    def get_task_message(self, export_task_action_opt_msg: dict):
        modelId = export_task_action_opt_msg['modelId']  # 当前待转换模型的训练任务的taskId
        format = export_task_action_opt_msg['format']
        taskId = export_task_action_opt_msg['taskId']  # 当前导出任务的taskId
        return (modelId, format, taskId)

    def upload_task_result(self, taskId: str, status: ExportStatusType, message: str):
        task_result = {
            "taskId": taskId,
            "status": status,
            "message": message
        }
        message_id = self.rds.xadd(self.export_action_result_topic_name, {
                                   'exportResult': json.dumps(task_result).encode()}, maxlen=100)

    def start_export_process(self, modelId, format, taskId):
        """
        modelId是之前训练时的taskId,用于找到对应的pt,而此处的taskId是导出的任务id
        """
        # 每个导出任务的唯一标识
        if self.has_task(taskId):  # 当前任务还存在
            repeat_export_log = Logger(
                f'{self.logs}/repeat_export_log_{taskId}.txt', level='info')

            self.upload_task_result(taskId, ExportStatusType.FAILED,
                                    f'导出任务:{taskId}已经启动运行中,无需重复启动！')
            repeat_export_log.logger.info(
                f'导出任务:{taskId}已经启动运行中,无需重复启动！')
            return
        # 正常启动导出--导出+上传(不耗时,同步)
        start_export_log = Logger(
            f'{self.logs}/start_export_log_{taskId}.txt', level='info')
        curr_task = ExportTask(
            taskId,
            modelId,
            format,
            self.minio_client,
            minio_bucket_prefix,
            minio_export_prefix,
            max_workers,
            start_export_log,
            self.rds)  # 传递Redis客户端，用于状态更新和消息发送
        # 这里和训练不一样,导出时间较短,直接在一个进程中启动导出、上传
        pe = mp.Process(target=curr_task.start_export_task)
        pe.start()
        # 导出时间短，但还是要在进程中启动，因为整体导出服务是在进程中执行，会加载模型而不释放
        # curr_task.start_export_task()

    def stop_export_process(self, taskId, taskName):
        """
        导出耗时短，暂不考虑停止
        """
        process_key = f"{taskId}_{taskName}"
        stop_export_log = Logger(
            f'{self.logs}/stop_export_log_{process_key}.txt', level='info')
        pass

    def run(self):
        # 消费平台下发的导出任务消息
        export_action_opt_log = Logger(
            f'{self.logs}/export_action_opt.txt', level='info')
        group_name = ''.join(['action_', IP])
        if not self.rds.exists(self.export_action_opt_topic_name) or not self.check_consumer_group_exists(self.export_action_opt_topic_name, group_name):
            self.rds.xgroup_create(
                name=self.export_action_opt_topic_name,
                groupname=group_name,
                id='$', mkstream=True)  # 创建消费者组--消费训练配置消息
        while True:
            try:
                response = self.rds.xreadgroup(
                    groupname=group_name,
                    consumername=group_name,
                    streams={self.export_action_opt_topic_name: '>'},
                    block=1000, count=1)
                if response:
                    export_action_opt_log.logger.info(
                        f'导出服务监听到Redis消息...')
                else:
                    # export_action_opt_log.logger.warn('训练服务没监听到Redis消息!')
                    time.sleep(3)
                    continue
            except Exception as ex:
                export_action_opt_log.logger.error(
                    f'Redis监听异常,尝试重连Redis!\t异常信息:{ex}')
                time.sleep(3)
                continue
            try:
                stream_name, message_list = response[0]
                message_id, message_data = message_list[0]
                # 完成消费后,标记消息为已处理,并且任务被执行前删除该消息
                self.rds.execute_command(
                    'XACK', self.export_action_opt_topic_name, group_name, message_id)
                opt_msg = message_data[b"action"].decode()
                export_action_opt_log.logger.info(f'Redis消息内容:{opt_msg}')
                export_task_action_opt_msg = json.loads(opt_msg)
                if export_task_action_opt_msg['ip'] != IP:
                    continue
                self.rds.xdel(self.export_action_opt_topic_name, message_id)
                # 解析导出任务消息
                modelId, format, taskId = self.get_task_message(
                    export_task_action_opt_msg)
                # 启用模型导出
                # 导出耗时比较短,每个任务同步实现
                self.start_export_process(modelId, format, taskId)
                time.sleep(10)
            except Exception as ex:
                export_action_opt_log.logger.error(f'导出服务启动发生异常!\terr:{ex}')
