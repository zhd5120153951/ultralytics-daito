'''
@FileName   :train_service.py
@Description:训练服务入口程序，监听Redis消息并管理训练任务
@Date       :2025/03/06 10:00:00
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import json
import time
import signal
import psutil
import datetime
import multiprocessing as mp
from redis import Redis
from threading import Thread

# 导入配置和工具函数
from config import (
    redis_ip, redis_port, redis_pwd,
    train_action_opt_topic_name,
    train_action_result_topic_name,
    train_result_topic_name,
    data_cfg, train_result, pretrained_models
)
from utils import find_pt

class TrainingTask:
    """训练任务类，负责单个训练任务的执行和管理"""
    def __init__(self, task_id, task_name, net_type, train_type, model_id, parameters, labels):
        self.task_id = task_id
        self.task_name = task_name
        self.net_type = net_type
        self.train_type = train_type
        self.model_id = model_id
        self.parameters = parameters
        self.labels = labels
        self.process = None
        self.log_file = None
        self.start_time = None
        self.redis_client = None
        self.process_key = f"{task_id}_{task_name}"

    def _generate_train_script(self):
        """生成训练脚本"""
        script_path = f"./train_scripts/{self.process_key}.py"
        os.makedirs(os.path.dirname(script_path), exist_ok=True)

        train_params = {
            'data': f'{data_cfg}/{self.process_key}.yaml',
            'project': f'{train_result}',
            'name': f'{self.process_key}',
            'task': 'detect'
        }
        train_params.update(self.parameters)

        with open(script_path, 'w', encoding='utf-8') as file:
            file.write("# 此文件由程序自动生成.\n")
            file.write("# 请勿手动修改.\n\n")
            file.write('import os\n')
            file.write('import shutil\n')
            file.write('from redis import Redis\n')
            file.write('from ultralytics import YOLO\n\n')

            # 设置模型配置和路径
            if self.train_type == 'Init':
                if self.net_type.startswith('yolov5'):
                    file.write(f"model_cfg = '{pretrained_models}/{self.net_type}u.yaml'\n")
                    file.write(f"model_path = '{pretrained_models}/{self.net_type}u.pt'\n")
                else:
                    file.write(f"model_cfg = '{pretrained_models}/{self.net_type}.yaml'\n")
                    file.write(f"model_path = '{pretrained_models}/{self.net_type}.pt'\n")
            else:
                iter_pre_model = find_pt('export_results', self.model_id)
                if self.net_type.startswith('yolov5'):
                    file.write(f"model_cfg = '{pretrained_models}/{self.net_type}u.yaml'\n")
                    file.write(f"model_path = '{iter_pre_model}'\n")
                else:
                    file.write(f"model_cfg = '{pretrained_models}/{self.net_type}.yaml'\n")
                    file.write(f"model_path = '{iter_pre_model}'\n")

            # 连接Redis并加载模型
            file.write(f'redis_client = Redis("{redis_ip}", {redis_port}, password="{redis_pwd}", decode_responses=True)\n')
            file.write('model = YOLO(model_cfg).load(model_path, redis_client=redis_client)\n')
            file.write(f'train_params = {train_params}\n\n')
            file.write('model.train(**train_params)\n\n')

            # 训练完成后的模型移动操作
            model_trained_path = '/'.join([train_params['project'],
                                        train_params['name'], 'weights', 'best.pt'])
            file.write(f"model_trained_path = '{model_trained_path}'\n")
            file.write(f"if not os.path.exists('{'export_results'}'):\n")
            file.write(f"    os.makedirs('{'export_results'}')\n")
            file.write(f"model_rename = '{self.process_key}'\n")
            file.write("model_rename = ''.join([model_rename,'.pt'])\n")
            file.write(f"model_rename_path = '/'.join(['export_results',model_rename])\n")
            file.write("shutil.copy(model_trained_path,model_rename_path)\n")

        return script_path

    def _train_process(self, script_path):
        """训练进程的执行函数"""
        import sys
        sys.path.append(os.path.dirname(script_path))
        exec(open(script_path).read())

    def start(self):
        """启动训练任务"""
        script_path = self._generate_train_script()
        log_dir = "./log"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(f"{log_dir}/train_log_{self.process_key}.txt", 'w', encoding='utf-8')
        self.start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # 创建并启动训练进程
        self.process = mp.Process(target=self._train_process, args=(script_path,))
        self.process.start()

        return True

    def stop(self):
        """停止训练任务"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()

        if self.log_file:
            self.log_file.close()

        return True

class ProcessManager:
    """进程管理器类，负责管理所有训练进程"""
    def __init__(self):
        self.active_tasks = {}

    def add_task(self, task):
        """添加新的训练任务"""
        self.active_tasks[task.process_key] = task

    def remove_task(self, process_key):
        """移除训练任务"""
        if process_key in self.active_tasks:
            del self.active_tasks[process_key]

    def get_task(self, process_key):
        """获取训练任务"""
        return self.active_tasks.get(process_key)

    def stop_all_tasks(self):
        """停止所有训练任务"""
        for task in list(self.active_tasks.values()):
            task.stop()
            self.remove_task(task.process_key)

class TrainingService:
    """训练服务类，负责整体训练服务的管理"""
    def __init__(self):
        self.process_manager = ProcessManager()
        self.redis_client = Redis(host=redis_ip, port=redis_port,
                                password=redis_pwd, decode_responses=True)

    def _ensure_directories(self):
        """确保必要的目录存在"""
        for directory in [data_cfg, train_result, pretrained_models, 'log']:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

    def _task_callback(self, action, task_id, task_name, status, message, create_time):
        """训练任务回调函数"""
        result = {
            'taskId': task_id,
            'taskName': task_name,
            'status': status,
            'message': message,
            'createTime': create_time
        }
        self.redis_client.xadd(train_action_result_topic_name,
                              {'result': json.dumps(result)}, maxlen=100)

    def handle_train_message(self, message):
        """处理训练任务消息"""
        try:
            data = json.loads(message['data'])
            action = data.get('action')
            task_id = data.get('taskId')
            task_name = data.get('taskName')
            process_key = f"{task_id}_{task_name}"

            if action == 'start':
                if self.process_manager.get_task(process_key):
                    create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    self._task_callback('start', task_id, task_name, 0,
                                    f'当前训练任务:{process_key}已在运行中，无法重复启动', create_time)
                    return

                task = TrainingTask(
                    task_id=task_id,
                    task_name=task_name,
                    net_type=data.get('netType', 'yolov8s'),
                    train_type=data.get('trainType', 'Init'),
                    model_id=data.get('modelId', ''),
                    parameters=data.get('parameters', {}),
                    labels=data.get('labels', [])
                )

                if task.start():
                    self.process_manager.add_task(task)
                    create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    self._task_callback('start', task_id, task_name, 1,
                                    f'当前训练任务:{process_key}启动成功,进程已启动.', create_time)

            elif action == 'stop':
                task = self.process_manager.get_task(process_key)
                if not task:
                    create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    self._task_callback('stop', task_id, task_name, 0,
                                    f'当前训练任务:{process_key}不存在或已停止', create_time)
                    return

                if task.stop():
                    self.process_manager.remove_task(process_key)
                    create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    self._task_callback('stop', task_id, task_name, 1,
                                    f'当前训练任务:{process_key}已停止', create_time)

        except Exception as e:
            print(f"处理训练消息异常: {e}")

    def signal_handler(self, sig, frame):
        """信号处理函数"""
        print("接收到终止信号，正在清理资源...")
        self.process_manager.stop_all_tasks()
        print("资源清理完成，退出程序")
        exit(0)

    def run(self):
        """运行训练服务"""
        # 注册信号处理函数
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # 确保必要的目录存在
        self._ensure_directories()

        # 创建消费者组
        try:
            self.redis_client.xgroup_create(train_action_opt_topic_name,
                                        'train_service_group', mkstream=True)
        except Exception:
            pass

        print("训练服务已启动，等待训练任务...")

        while True:
            try:
                messages = self.redis_client.xreadgroup(
                    'train_service_group', 'train_service_consumer',
                    {train_action_opt_topic_name: '>'},
                    count=1, block=1000
                )

                if messages:
                    for stream, message_list in messages:
                        for message_id, message in message_list:
                            self.handle_train_message(message)
                            self.redis_client.xack(train_action_opt_topic_name,
                                                'train_service_group', message_id)

            except Exception as e:
                print(f"监听消息异常: {e}")
                time.sleep(1)

if __name__ == "__main__":
    service = TrainingService()
    service.run()
