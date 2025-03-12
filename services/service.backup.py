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
import psutil
import signal
import subprocess
from utils import Logger
from services import ProcessManager, TrainTask
from config import (IP, minio_endpoint,
                    minio_access_key,
                    minio_secret_key,
                    data_cfg,
                    train_result,
                    export_result,
                    minio_train_prefix,
                    minio_export_prefix)


class BaseService:
    """服务基类，包含共用方法."""

    def __init__(self,
                 redis_ip,
                 redis_port,
                 redis_pwd):
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
        self.process_manager = ProcessManager()

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

    def handle_signal(self, sig, frame):
        """_summary_

        Args:
            sig (_type_): 信号
            frame (_type_): 栈帧
        """
        print(f"Received signal is: {sig}\nReceived frame is: {frame}")
        self.process_manager.stop_all_tasks()
        print("All processes have been stopped.")
        exit(0)


class trainService(BaseService):
    def __init__(self,
                 redis_ip,
                 redis_port,
                 redis_pwd,
                 train_host_msg,
                 train_action_opt_topic_name,
                 train_action_result_topic_name,
                 train_data_download_topic_name):
        super().__init__(
            redis_ip,
            redis_port,
            redis_pwd)
        # ... 子类的初始化代码 ...
        self.train_host_msg = train_host_msg
        self.train_action_opt_topic_name = train_action_opt_topic_name
        self.train_action_result_topic_name = train_action_result_topic_name
        self.train_data_download_topic_name = train_data_download_topic_name
        # 已经运行的训练任务名--用于判断是否重复启动任务
        self.train_task_name = []
        # 已经运行的训练任务后台进程--用于判断是否能够正常停止当前任务
        self.train_task_obj = {}

    def get_task_message(self, train_task_action_opt_msg: dict):
        action = train_task_action_opt_msg['action']
        taskId = train_task_action_opt_msg['taskId']
        taskName = train_task_action_opt_msg['taskName']
        modelId = train_task_action_opt_msg['modelId']
        netType = train_task_action_opt_msg['netType']
        modelType = train_task_action_opt_msg['modelType']
        prefix = train_task_action_opt_msg['prefix']  # list
        labels = train_task_action_opt_msg['labels']  # list
        ratio = train_task_action_opt_msg['ratio']
        train_params = train_task_action_opt_msg['train_params']
        return (action, taskId, taskName, modelId, netType, modelType, prefix, labels, ratio, train_params)

    def upload_data_download_result(self, downlaod_finish):
        '''
        具体上传怎么内容需要和平台协定
        '''
        message_id = self.rds.xadd(self.train_data_download_topic_name,
                                   {'result': str({'ret': downlaod_finish}).encode()}, maxlen=100)
        # log.logger.info(f'data_downkload_result: {data_download_result}')

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
        message_id = self.rds.xadd(self.train_action_result_topic_name, {
                                   'result': str(task_result).encode()}, maxlen=100)
        # log.logger.info(f'task_result:{task_result}')

    def upload_train_result_minio(self, action, taskId, taskName, create_time):
        '''
        上传训练最终结果:打包后的训练内容,往Minio发
        '''
        pass

    def upload_train_info(self):
        '''
        上传训练中间过程:比如当前epoch/loss/percise/recall等内容
        '''
        pass

    def start_train_process(self, taskId: str, taskName: str, modelId: str, netType: str, modelType: str, prefix: list, labels: list, ratio: float, parameters: dict):
        # **************************************
        # 每个任务进程的唯一标识
        process_key = f"{taskId}_{taskName}"
        # 重复启动
        if self.process_manager.get_task(process_key):
            repeat_train_log = Logger(
                f'./log/repeat_train_{process_key}.txt', level='info')
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.upload_task_result(
                "start",
                taskId,
                taskName,
                0,
                f"训练任务:{process_key}已经启动运行中,无需重复启动！",
                create_time)
            return
        # 正常启动
        # 1.数据集并发下载
        from utils import Dataset, uploadMinio
        downlaod_datasets_log = Logger(
            f"./log/download_datasets_{process_key}.txt", level="info")
        data_yaml_path = f"{data_cfg}/{taskId}_{taskName}.yaml"
        if not modelId:  # 初始化训练
            trainType = "Init"
        else:  # 迭代训练
            trainType = "iteration"
        dataset = Dataset(self.minio_client,
                          "datasets",
                          "train",
                          prefix,
                          labels,
                          ratio,
                          data_yaml_path,
                          downlaod_datasets_log)
        dataset.start()
        dataset.join()  # 这里阻塞,等待数据集的操作全部完成才能训练
        self.upload_data_download_result(True)
        # 2.异步训练--classify,detect,obb,segment,pose
        curr_task = TrainTask(
            self.rds,
            taskId,
            taskName,
            netType,
            trainType,
            modelId,
            modelType,
            parameters,
            labels)
        if curr_task.start():
            self.process_manager.add_task(curr_task)
            create_time = datetime.datetime().now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.upload_task_result(
                "start", taskId, taskName, 1, f"训练任务:{process_key}启动成功", create_time)
        else:
            create_time = datetime.datetime().now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.upload_task_result(
                "start", taskId, taskName, 0, f"训练任务:{process_key}启动失败", create_time)
        # 3.异步上传结果
        upload_train_result_log = Logger(
            f'./log/upload_train_result_{process_key}.txt', level='info')
        upload_minio = uploadMinio(self.rds,
                                   self.minio_client,
                                   'train',
                                   train_result,
                                   taskId,
                                   taskName,
                                   minio_train_prefix,
                                   'train',
                                   # modelId是导出上次的模型名字(taskId_taskName),对于训练时,此参数不给
                                   "",
                                   upload_train_result_log)
        upload_minio.start()
        # **************************************
        # 确保没有相同训练任务
        enableFlag = False
        if taskName not in self.train_task_name:
            enableFlag = True
            self.train_task_name.append(taskName)
        else:  # 同名训练任务不启动
            enableFlag = False
        if enableFlag:
            from utils import Dataset
            from utils import uploadMinio
            # 开始训练--这里必须要异步
            if modelType == 'classify':
                pass
            if modelType == 'detect':
                from utils import trainYOLODetectTask
                #######################################
                download_datasets_log = Logger(
                    f'./log/download_datasets_{taskId}_{taskName}.txt', level='info')
                # 1.数据集处理
                # 数据集配置文件及数据要提前构建
                data_yaml_path = f'{data_cfg}/{taskId}_{taskName}.yaml'
                if not modelId:  # 初始化训练
                    trainType = 'Init'
                else:  # 迭代训练
                    trainType = 'iteration'
                dataset = Dataset(self.minio_client,
                                  'datasets',
                                  'train',
                                  prefix,
                                  labels,
                                  ratio,
                                  data_yaml_path,
                                  download_datasets_log)
                dataset.start()
                dataset.join()  # 阻塞此处,等待数据集的操作全部完成
                ##############################################
                # 2.异步训练
                # 往Redis发一条数据下载的完成情况消息
                self.upload_data_download_result(True)
                # 用类(继承Thread)来启动训练任务--用类的方式更好
                train_task_yolo_det = trainYOLODetectTask(self.rds,
                                                          taskId,
                                                          taskName,
                                                          self.train_task_name,
                                                          self.train_task_obj,
                                                          labels,
                                                          netType,
                                                          trainType,
                                                          modelId,
                                                          parameters,
                                                          self.upload_task_result)
                train_task_yolo_det.start()
                ##############################################
                # 3.上传结果
                # 上传训练结果到minio--都是异步的,但是一定要等到训练结束后才能开始上传
                upload_train_result_log = Logger(
                    f'./log/upload_train_result_{taskId}_{taskName}.txt', level='info')
                upload_minio = uploadMinio(self.rds,
                                           self.minio_client,
                                           'train',
                                           train_result,
                                           taskId,
                                           taskName,
                                           minio_train_prefix,
                                           'train',
                                           # modelId是导出上次的模型名字(taskId_taskName),对于训练时,此参数不给
                                           "",
                                           upload_train_result_log)
                upload_minio.start()
                # upload_minio.join()#这里不能等待,不然下个训练消息来了会堵住while循环
            elif modelType == 'segment':
                pass
            elif modelType == 'pose':
                pass
            else:
                pass

        else:
            repeat_train_log = Logger(
                f'./log/repeat_train_{taskName}.txt', level='info')
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            # 当前训练任务进程没结束
            if psutil.pid_exists(self.train_task_obj[taskName].pid):
                # 上传任务重复启动的消息到Redis(有用)
                self.upload_task_result(
                    'start', taskId, taskName, 0, f'训练任务:{taskName}已经启动运行中,无需重复启动！', create_time)
            else:
                repeat_train_log.logger.error(
                    '此条日志出现说明程序出现bug,bug原因是前一个任务没结束(pid存在),然后重启一个同名任务,应该显示任务运行中.')
            repeat_train_log.logger.info(f'训练任务:{taskName}的进程已经启动运行中,无需重复启动！')

    def stop_train_process(self, taskId, taskName):
        # **********************************
        process_key = f"{taskId}_{taskName}"
        stop_train_log = Logger(
            f'./log/stop_train_{process_key}.txt', level='info')
        task = self.process_manager.get_task(process_key)
        if not task:  # 为空--该任务进程训练已结束--无需终止
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.upload_task_result(
                "stop",
                taskId,
                taskName,
                0,
                f"训练任务{process_key}不存在或已经停止运行！", create_time)
            stop_train_log.logger.info(f"训练任务{process_key}已正常训练完成,后台进程已正常退出！")
            return
        else:  # 非空--该任务还在训练中--需要终止
            if task.stop():
                self.process_manager.remove_task(process_key)
                create_time = datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S.%f')[:-3]
                self.upload_task_result(
                    "stop", taskId, taskName, 1, f"训练任务{process_key}停止成功", create_time)
                stop_train_log.logger.info(f"已正常停止正在训练中的任务:{process_key}！")
        # ************************************
        stop_train_log = Logger(
            f'./log/stop_train_{taskId}_{taskName}.txt', level='info')
        create_time = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S.%f')[:-3]
        if self.train_task_obj:  # 非空--该任务进程还在训练中--终止
            subproc = self.train_task_obj[taskName]  # 后台训练进程对象
            subproc.terminate()
            try:
                subproc.wait(5)  # 等5s内退出--优雅退出
                stop_train_log.logger.info(f'已正常停止正在训练中的任务:{taskName}！')
            except subprocess.TimeoutExpired as st:
                # 退出超时--暴力退出
                subproc.kill()
                subproc.wait()
                stop_train_log.logger.info(
                    f'已暴力停止正在训练中的任务:{taskName},超时异常:{st}！')
            # 训练任务停止成功--向Redis发送消息(有用)
            self.upload_task_result(
                'stop', taskId, taskName, 1, f'训练任务{taskName}停止成功', create_time)
        else:  # 为空--该任务进程训练已结束--无需终止
            # 训练任务已正常结束--向Redis发消息(无用)
            # self.upload_task_result(
            #     'stop', '12345', taskName, 0, f'训练任务{taskName}已正常完成,', create_time)
            stop_train_log.logger.info(f'训练任务{taskName}已正常训练完成,后台进程已正常退出！')

    def run(self):
        signal.signal(signal.SIGINT, self.handle_signal)
        # 消费平台下发的训练任务消息
        train_action_opt_log = Logger(
            f'./log/train_action_opt.txt', level='info')
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
                if train_task_action_opt_msg['IP'] != IP:
                    continue
                self.rds.xdel(self.train_action_opt_topic_name, message_id)
                # 解析训练任务消息
                action, taskId, taskName, modelId, netType, modelType, prefix, labels, ratio, parameters = self.get_task_message(
                    train_task_action_opt_msg)
                # 启用模型训练
                if action == 'start':
                    # 检查GPU资源
                    availableGPUId, gpu_unit_use = self.getAvailableGPUId(
                        train_action_opt_log)
                    if availableGPUId == '-2':
                        train_action_opt_log.logger.info(
                            f'训练任务:{taskName}启动时未检测到可用的GPU,训练任务无法正常启动！')
                    elif availableGPUId == '-1':
                        train_action_opt_log.logger.error(
                            f'训练任务:{taskName}启动时检测到GPU资源不足,训练任务无法正常启动！')
                    else:
                        # 用这个可以对应多个训练任务
                        # 初始化训练进度0--未开始,1--进行中,-1--失败
                        # self.rds.hdel()--删除train_progress
                        self.rds.hset(f'{taskId}_{taskName}_train_progress', mapping={
                                      'status': 0, 'epoch': 0})

                        self.start_train_process(
                            taskId,
                            taskName,
                            modelId,
                            netType,
                            modelType,
                            prefix,
                            labels,
                            ratio,
                            parameters)
                        train_action_opt_log.logger.info(
                            f'训练任务:{taskName}启动时检测到可用GPU:{availableGPUId}的利用率最低,且利用率:{gpu_unit_use}')
                        time.sleep(10)
                else:  # 停止模型训练--1:训练没结束时停止;2:训练已结束时停止
                    self.stop_train_process(taskId, taskName)
            except Exception as ex:
                train_action_opt_log.logger.error(f'训练服务启动发生异常!\terr:{ex}')


class exportService(BaseService):
    def __init__(self,
                 redis_ip,
                 redis_port,
                 redis_pwd,
                 export_host_msg,
                 export_action_opt_topic_name,
                 export_action_result_topic_name):
        super().__init__(redis_ip, redis_port, redis_pwd)
        # ... 子类初始化代码 ...
        self.export_host_msg = export_host_msg
        self.export_action_opt_topic_name = export_action_opt_topic_name
        self.export_action_result_topic_name = export_action_result_topic_name
        # 已经运行的导出任务名--用于判断是否重复启动任务
        self.export_task_name = []
        # 已经运行的导出任务后台进程--用于判断是否能够正常停止当前任务
        self.export_task_obj = {}

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

    def signal_handler(self, sig, frame):
        """_summary_

        Args:
            sig (_type_): 信号
            frame (_type_): 栈帧
        """
        print("接收到终止信号,正在清理资源...")
        self.process_manager.stop_all_tasks()
        print("资源清理完成,退出程序!")
        exit(0)

    def start_export_process(self, taskId, taskName, modelId, exportType):
        """taskId,taskName是本次导出的任务id和name,modelId(taskId_taskName)是之前训练时的id和name,用于找到pt
        """
        # 确保没有相同导出任务
        enableFlag = False
        if taskName not in self.export_task_name:
            enableFlag = True
            self.export_task_name.append(taskName)
        else:  # 同名导出任务不启动
            enableFlag = False
        if enableFlag:
            # 开始导出--这里必须要异步
            if exportType == 'paddle':
                from utils import uploadMinio
                from utils import exportPaddleTask
                # 开启导出
                export_task = exportPaddleTask(
                    self.rds,
                    taskId,
                    taskName,
                    self.export_task_name,
                    self.export_task_obj,
                    modelId,
                    self.upload_task_result)
                export_task.start()
                export_task.join()  # 导出结束后才能上传
                # 上传结果
                upload_export_result_log = Logger(
                    f'./log/upload_export_result_{taskId}_{taskName}.txt', level='info')
                upload_minio = uploadMinio(self.rds,
                                           self.minio_client,
                                           'train',
                                           export_result,
                                           taskId,
                                           taskName,
                                           minio_export_prefix,
                                           'export',
                                           modelId,
                                           upload_export_result_log)
                upload_minio.start()
                # upload_minio.join()#这里不能等待,不然下个训练消息来了会堵住while循环
            else:  # rknn
                pass
        else:
            repeat_export_log = Logger(
                f'./log/repeat_export_{taskId}_{taskName}.txt', level='info')
            create_time = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')[:-3]
            # 当前训练任务进程没结束
            if psutil.pid_exists(self.export_task_obj[taskName].pid):
                # 上传任务重复启动的消息到Redis(有用)
                self.upload_task_result(
                    'start', taskId, taskName, 0, f'导出任务:{taskName}已经启动运行中,无需重复启动！', create_time)
            else:
                repeat_export_log.logger.error(
                    '此条日志出现说明程序出现bug,bug原因是前一个任务没结束(pid存在),然后重启一个同名任务,应该显示任务运行中.')
            repeat_export_log.logger.info(
                f'导出任务:{taskId}_{taskName}的进程已经启动运行中,无需重复启动！')

    def stop_export_process(self, taskId, taskName):
        stop_export_log = Logger(
            f'./log/stop_export_{taskId}_{taskName}.txt', level='info')
        create_time = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S.%f')[:-3]
        if self.export_task_obj:  # 非空--该任务进程还在训练中--终止
            subproc = self.export_task_obj[taskName]  # 后台训练进程对象
            subproc.terminate()
            try:
                subproc.wait(5)  # 等5s内退出--优雅退出
                stop_export_log.logger.info(f'已正常停止正在导出中的任务:{taskName}！')
            except subprocess.TimeoutExpired as st:
                # 退出超时--暴力退出
                subproc.kill()
                subproc.wait()
                stop_export_log.logger.info(
                    f'已暴力停止正在训练中的任务:{taskName},超时异常:{st}！')
            # 训练任务停止成功--向Redis发送消息(有用)
            self.upload_task_result(
                'stop', taskId, taskName, 1, f'导出任务{taskName}停止成功', create_time)
        else:  # 为空--该任务进程训练已结束--无需终止
            # 训练任务已正常结束--向Redis发消息(无用)
            # self.upload_task_result(
            #     'stop', '12345', taskName, 0, f'训练任务{taskName}已正常完成,', create_time)
            stop_export_log.logger.info(f'导出任务{taskName}已正常导出完成,后台进程已正常退出！')

    def run(self):
        signal.signal(signal.SIGINT, self.handle_signal)
        # 消费平台下发的导出任务消息
        export_action_opt_log = Logger(
            f'./log/export_action_opt.txt', level='info')
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
                    else:
                        # self.rds.hset(f'{taskId}_{taskName}_train_progress', mapping={
                        #               'status': 0, 'epoch': 0})
                        # 导出很快,不需要发送中间过程,只需要告诉最终的导出成功与否?
                        self.start_export_process(
                            taskId, taskName, modelId, exportType)
                        export_action_opt_log.logger.info(
                            f'导出任务:{taskName}启动时检测到可用GPU:{availableGPUId}的利用率最低,且利用率:{gpu_unit_use}')
                        time.sleep(10)
                else:  # 停止模型训练--1:训练没结束时停止;2:训练已结束时停止
                    self.stop_export_process(taskId, taskName)
            except Exception as ex:
                export_action_opt_log.logger.error(f'导出服务启动发生异常!\terr:{ex}')
