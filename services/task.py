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
import shutil
import datetime
import multiprocessing as mp
from ultralytics import YOLO
from utils import find_pt
from config import (log,
                    data_cfg,
                    pretrained_models,
                    train_result,
                    export_result,
                    support_net_type)


class TrainTask:
    """_summary_
    训练任务类:负责单个训练任务的执行和管理
    """

    def __init__(self, rds, task_id, task_name, net_type, train_type, model_id, model_type, parameters, labels):
        self.rds = rds
        self.task_id = task_id
        self.task_name = task_name
        self.net_type = net_type
        self.train_type = train_type
        self.model_id = model_id
        self.model_type = model_type
        self.parameters = parameters
        self.labels = labels
        self.process = None
        self.log_file = None
        self.start_time = None
        self.redis_client = None
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

    def _train_process(self, log_file):
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
                model_cfg = f"{pretrained_models}/{self.net_type}.yaml"
                model_path = f"{pretrained_models}/{self.net_type}.pt"
            else:  # 迭代--modelId是上一次训练的taskId_taskName
                iter_pre_model = find_pt(export_result, self.model_id)
                if self.net_type.startswith("yolov5"):
                    model_cfg = f"{pretrained_models}/{self.net_type}u.yaml"
                    model_path = iter_pre_model
                model_cfg = f"{pretrained_models}/{self.net_type}.yaml"
                model_path = iter_pre_model
        else:
            # raise ValueError(f"不支持的模型类型:{self.net_type}")
            log_file.write(
                f"不支持的网络类型:{self.net_type},目前支持的网络类型有:{support_net_type}")
            return

        model = YOLO(model_cfg).load(
            self.rds,
            self.task_id,
            self.task_name,
            model_path)
        model.train(**train_params)
        model_trained_path = f"{train_result}/{self.process_key}/weights/best.pt"
        model_rename_path = f"{export_result}/{self.process_key}.pt"
        shutil.copy(model_trained_path, model_rename_path)
        log_file.write(f"训练任务:{self.process_key}完成")
        # log_file.write(f"训练任务:{self.process_key}完成时间:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        log_file.close()

    def start_train_task(self):
        """_summary_
        启动训练任务
        return: bool:是否成功启动训练任务.True--成功启动,False--启动失败
        """
        self.log_file = open(
            f"{log}/train_log_{self.process_key}.txt", "w", encoding="utf-8")
        self.start_time = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f")[:-3]
        # 启动进程
        self.process = mp.Process(
            target=self._train_process, args=(self.log_file,))
        self.process.start()
        if self.process.is_alive():
            return True
        return False

    def stop_train_task(self):
        """_summary_
        停止训练任务
        return: bool:是否成功停止训练任务.True--成功停止,False--停止失败
        """
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)  # 等待5s退出
            if self.process.is_alive():
                self.process.kill()  # 暴力退出
            self.log_file.close()
            return True
        self.log_file.close()
        return False

    def get_train_task_status(self):
        pass

    def get_train_task_result(self):
        pass

    def get_train_task_log(self):
        pass


class ExportTask:
    """_summary_
    导出任务类:负责单个导出任务的执行和管理
    """

    def __init__(self, task_id, task_name, model_id, parameters):
        self.task_id = task_id
        self.task_name = task_name
        self.model_id = model_id
        self.parameters = parameters
        self.process = None
        self.log_file = None
        self.start_time = None
        self.redis_client = None
        self.process_key = f"{task_id}_{task_name}"  # 进程名:任务id_任务名称

    def start_export_task(self):
        pass

    def stop_export_task(self):
        pass

    def get_export_task_status(self):
        pass

    def get_export_task_result(self):
        pass

    def get_export_task_log(self):
        pass
