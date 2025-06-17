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

import redis
import minio
import time
import datetime
import json
import multiprocessing as mp
from utils import Logger, Dataset, uploadMinio, TrainStatusType
from services.task import TrainTask, ExportTask
from config import (IP, minio_endpoint,
                    minio_access_key,
                    minio_secret_key,
                    data_cfg,
                    train_result,
                    minio_train_prefix,
                    minio_export_prefix,
                    max_workers)


class BaseService:
    """服务基类，包含共用方法."""

    def __init__(self,
                 redis_ip,
                 redis_port,
                 redis_pwd,
                 logs):
        self.redis_pool = redis.ConnectionPool(host=redis_ip,
                                               port=redis_port,
                                               password=redis_pwd,
                                               db=0,
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
                 logs,
                 train_action_opt_topic_name,
                 train_action_result_topic_name):
        super().__init__(redis_ip,
                         redis_port,
                         redis_pwd,
                         logs)
        # ... 子类的初始化代码 ...
        self.train_action_opt_topic_name = train_action_opt_topic_name
        self.train_action_result_topic_name = train_action_result_topic_name
        self.active_procs = {}

    def add_proc(self, process_key: str, process: mp.Process):
        """_summary_
        添加进程
        """
        self.active_procs[process_key] = process

    def remove_proc(self, process_key: str):
        """_summary_
        移除进程
        """
        del self.active_procs[process_key]

    def get_proc(self, process_key: str):
        """_summary_
        获取进程
        """
        return self.active_procs.get(process_key, None)

    def stop_proc(self, process_key: str):
        """_summary_
        停止进程
        """
        task_proc = self.get_proc(process_key)
        if not task_proc:
            print(f"当前任务:{process_key}进程不存在,无需退出！")
            return False
        elif task_proc.is_alive():
            task_proc.terminate()
            task_proc.join(5)
            if task_proc.is_alive():
                task_proc.kill()
            print(f"当前任务:{process_key}进程成功停止.")
            return True
        print(f"当前任务:{process_key}已完成,进程自己终止,无需停止.")
        return False

    def stop_all_procs(self):
        """_summary_
        停止所有进程
        """
        for proc_key in list(self.active_procs.keys()):
            self.remove_proc(proc_key)
            self.stop_proc(proc_key)
        print("All process have been stopped.")

    def get_task_message(self, train_task_action_opt_msg: dict):
        action = train_task_action_opt_msg['action']
        taskId = train_task_action_opt_msg['taskId']  # 做为唯一进程名
        modelId = train_task_action_opt_msg['modelId']
        netType = train_task_action_opt_msg['netType']
        modelType = train_task_action_opt_msg['modelType']
        prefix = train_task_action_opt_msg['datasets']  # list
        labels = train_task_action_opt_msg['labels']  # list
        ratio = train_task_action_opt_msg['ratio']
        train_params = train_task_action_opt_msg['train_params']
        return (action, taskId, modelId, netType, modelType, prefix, labels, ratio, train_params)

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
        # 设置训练状态--0空闲中,-1--训练出错,1--训练中
        self.rds.hset(f'{taskId}_train_status_info', mapping={
            "status": 0, "context": "空闲中"})
        # 查到有同名任务--重复启动--记录日志后直接退出
        if self.get_proc(taskId):
            repeat_train_log = Logger(
                f'{self.logs}/repeat_train_log_{taskId}.txt', level='info')
            repeat_train_log.logger.info(f"训练任务:{taskId}已经启动运行中,无需重复启动！")
            return
        # 正常启动训练--数据集下载、模型训练、结果上传(异步)
        # 1.数据集并发下载
        downlaod_datasets_log = Logger(
            f"{self.logs}/download_datasets_log_{taskId}.txt", level="info")
        data_yaml_path = f"{data_cfg}/{taskId}.yaml"
        if not modelId:  # 初始化训练
            trainType = "Init"
        else:  # 迭代训练
            trainType = "iteration"
        dataset = Dataset(self.minio_client,
                          "datasets",  # minio固定的数据集存放目录名
                          "train",
                          datasets,  # minio上数据集目录下的若干数据集名
                          labels,
                          ratio,
                          data_yaml_path,
                          downlaod_datasets_log)
        dataset.start()
        dataset.join()  # 这里阻塞,等待数据集的操作全部完成才能训练
        if not dataset.isFinished:
            self.upload_train_result(
                taskId, TrainStatusType.FAILED, "数据集下载失败")
            return
        # 2.异步训练--classify,detect,obb,segment,pose
        start_train_log = Logger(
            f"{self.logs}/start_train_log_{taskId}.txt", level="info")
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
        if p_task.is_alive():
            self.add_proc(taskId, p_task)  # 保存当前任务对应的进程:taskId:一个训练进程
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
            return
        # 3.上传minio
        upload_train_result_log = Logger(
            f'{self.logs}/upload_train_result_log_{taskId}.txt', level='info')
        upload_minio = uploadMinio(self.rds,
                                   self.minio_client,
                                   'train',
                                   train_result,
                                   taskId,
                                   minio_train_prefix,
                                   'train',
                                   # modelId是导出上次的模型名字(taskId),对于训练时,此参数可任意给
                                   "",
                                   upload_train_result_log,
                                   max_workers)
        upload_minio.start()
        if not upload_minio.isFinished:
            self.upload_train_result(
                taskId, TrainStatusType.FAILED, "训练结果上传Minio失败")

    def stop_train_process(self, taskId: str):
        """sumary_line
            taskId:做为每个训练进城的唯一key
        """

        stop_train_log = Logger(
            f'{self.logs}/stop_train_log_{taskId}.txt', level='info')
        if not self.stop_proc(taskId):
            stop_train_log.logger.info(f"训练任务{taskId}已正常训练完成,后台进程已正常退出！")
        else:  # 非空--该任务还在训练中--需要终止
            # 同时训练的状态流也该删除
            self.rds.hdel(f"{taskId}_train_status_info", ("status", "context"))
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
                    # 检查GPU资源
                    availableGPUId, gpu_unit_use = self.getAvailableGPUId(
                        train_action_opt_log)
                    if availableGPUId == '-2':
                        train_action_opt_log.logger.info(
                            f'训练任务:{taskId}启动时未检测到可用的GPU,训练任务无法正常启动！')
                    elif availableGPUId == '-1':
                        train_action_opt_log.logger.error(
                            f'训练任务:{taskId}启动时检测到GPU资源不足,训练任务无法正常启动！')
                    else:
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
                        train_action_opt_log.logger.info(
                            f'训练任务:{taskId}启动时检测到可用GPU:{availableGPUId}的利用率最低,且利用率:{gpu_unit_use}')
                        time.sleep(10)
                else:  # 停止模型训练--1:训练没结束时停止;2:训练已结束时停止
                    self.stop_train_process(taskId)
            except Exception as ex:
                train_action_opt_log.logger.error(f'训练服务启动发生异常!\terr:{ex}')


class exportService(BaseService):
    def __init__(self,
                 redis_ip,
                 redis_port,
                 redis_pwd,
                 logs,
                 export_action_opt_topic_name,
                 export_action_result_topic_name):
        super().__init__(redis_ip,
                         redis_port,
                         redis_pwd,
                         logs)
        # ... 子类初始化代码 ...
        self.export_action_opt_topic_name = export_action_opt_topic_name
        self.export_action_result_topic_name = export_action_result_topic_name
        self.export_tasks = {}  # 和训练不同,这里value不是进程对象

    def add_task(self, process_key: str, process: mp.Process):
        """_summary_
        添加任务
        """
        self.active_procs[process_key] = process

    def remove_task(self, process_key: str):
        """_summary_
        移除任务
        """
        del self.active_procs[process_key]

    def get_task(self, task_key: str):
        """sumary_line

        Keyword arguments:
        argument -- description
        Return: return_description
        """
        return self.export_tasks.get(task_key, None)

    def get_task_message(self, export_task_action_opt_msg: dict):
        action = export_task_action_opt_msg['action']
        taskId = export_task_action_opt_msg['taskId']
        taskName = export_task_action_opt_msg['taskName']
        modelId = export_task_action_opt_msg['modelId']
        exportType = export_task_action_opt_msg['exportType']
        return (action, taskId, taskName, modelId, exportType)

    def upload_task_result(self, action, taskId, taskName, exeResult, exeMsg, create_time):
        '''
        上传任务启动结果:
        1.当前任务启动:启动失败-启动重复->0 or 启动成功->1
        2.当前任务停止:停止失败-停止重复->0 or 停止成功->1
        '''
        task_result = {
            "action": action,
            "taskId": taskId,
            "taskName": taskName,
            "exeResult": exeResult,
            "exeMsg": exeMsg,
            "exeTime": create_time
        }
        message_id = self.rds.xadd(self.export_action_result_topic_name, {
                                   'result': str(task_result).encode()}, maxlen=100)

    def start_export_process(self, taskId, taskName, modelId, exportType):
        """
        taskId,taskName是本次导出的任务id和name,modelId(taskId_taskName)是之前训练时的id和name,用于找到pt
        """
        # 每个任务进程的唯一标识
        process_key = f"{taskId}_{taskName}"
        if not self.get_task(process_key):  # 没有当前重复的导出任务
            repeat_export_log = Logger(
                f'{self.logs}/repeat_export_log_{process_key}.txt', level='info')
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.upload_task_result(
                'start',
                taskId,
                taskName,
                0,
                f'导出任务:{process_key}已经启动运行中,无需重复启动！', create_time)
            repeat_export_log.logger.info(
                f'导出任务:{process_key}已经启动运行中,无需重复启动！')
            return
        # 正常启动导出--导出+上传(不耗时,同步)
        start_export_log = Logger(
            f'{self.logs}/start_export_log_{process_key}.txt', level='info')
        curr_task = ExportTask(
            taskId,
            taskName,
            modelId,
            exportType,
            self.minio_client,
            "train",
            minio_export_prefix,
            max_workers,
            start_export_log,
            self.rds)  # 传递Redis客户端，用于状态更新和消息发送
        # 这里和训练不一样,导出时间较短,直接在一个进程中启动导出、上传
        if curr_task.start_export_task():
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.upload_task_result(
                'start', taskId, taskName, 1, f'导出任务:{process_key}启动成功', create_time)
        else:
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.upload_task_result(
                'start', taskId, taskName, 0, f'导出任务:{process_key}启动失败', create_time)
            return  # 导出失败,直接返回,不上传导出结果

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
                if export_task_action_opt_msg['IP'] != IP:
                    continue
                self.rds.xdel(self.export_action_opt_topic_name, message_id)
                # 解析导出任务消息
                action, taskId, taskName, modelId, exportType = self.get_task_message(
                    export_task_action_opt_msg)
                # 启用模型导出
                if action == 'start':
                    # 检查GPU资源
                    availableGPUId, gpu_unit_use = self.getAvailableGPUId(
                        export_action_opt_log)
                    if availableGPUId == '-2':
                        export_action_opt_log.logger.info(
                            f'导出任务:{taskName}启动时未检测到可用的GPU,导出任务无法正常启动！')
                    elif availableGPUId == '-1':
                        export_action_opt_log.logger.error(
                            f'导出任务:{taskName}启动时检测到GPU资源不足,导出任务无法正常启动！')
                    else:  # 导出耗时比较短,每个任务同步实现
                        self.start_export_process(
                            taskId,
                            taskName,
                            modelId,
                            exportType)
                        export_action_opt_log.logger.info(
                            f'导出任务:{taskName}启动时检测到可用GPU:{availableGPUId}的利用率最低,且利用率:{gpu_unit_use}')
                        time.sleep(10)
                else:  # 停止模型训练--1:训练没结束时停止;2:训练已结束时停止
                    self.stop_export_process(taskId, taskName)
            except Exception as ex:
                export_action_opt_log.logger.error(f'导出服务启动发生异常!\terr:{ex}')
