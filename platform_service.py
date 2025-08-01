'''
@FileName   :platform_service.py
@Description:
@Date       :2024/09/10 17:40:41
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :LongRuan@copyright
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import multiprocessing
from config import (redis_ip,
                    redis_port,
                    redis_pwd,
                    redis_db,
                    data_cfg,
                    logs,
                    pretrained_models,
                    train_result,
                    export_result,
                    enhance_result,
                    local_data_dir,
                    train_action_opt_topic_name,
                    train_action_result_topic_name,
                    export_action_opt_topic_name,
                    export_action_result_topic_name,
                    enhance_action_opt_topic_name,
                    enhance_action_result_topic_name,
                    enhance_service_enable)


from ultralytics.utils import SettingsManager
from services import trainService, exportService, EnhanceService
settings = SettingsManager()
settings.update(datasets_dir="")


def main():
    '''
    训练平台主进程:启动导出任务、训练任务
    '''
    try:
        # 创建必要的目录
        for dir in [data_cfg, logs, pretrained_models, train_result, export_result, enhance_result, local_data_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
        if enhance_service_enable:  # 启用数据增强服务
            # 启动图像增强进程
            enhance_service = EnhanceService(redis_ip,
                                             redis_port,
                                             redis_pwd,
                                             redis_db,
                                             logs,
                                             enhance_action_opt_topic_name,
                                             enhance_action_result_topic_name)
            enhance_proc = multiprocessing.Process(target=enhance_service.run)
            enhance_proc.start()

        # 启动导出任务进程
        export_service = exportService(redis_ip,
                                       redis_port,
                                       redis_pwd,
                                       redis_db,
                                       logs,
                                       export_action_opt_topic_name,
                                       export_action_result_topic_name)
        export_proc = multiprocessing.Process(target=export_service.run)
        export_proc.start()

        # 启动训练任务主进程
        train_service = trainService(redis_ip,
                                     redis_port,
                                     redis_pwd,
                                     redis_db,
                                     logs,
                                     train_action_opt_topic_name,
                                     train_action_result_topic_name)
        train_service.run()
    except KeyboardInterrupt:
        print("中断服务,终止进程!")
        if enhance_service_enable:
            enhance_proc.terminate()
        export_proc.terminate()
        if enhance_service_enable:
            enhance_proc.join(5)
        export_proc.join(5)


if __name__ == '__main__':
    main()
