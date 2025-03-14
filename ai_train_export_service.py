'''
@FileName   :platform_service.py
@Description:
@Date       :2024/09/10 17:40:41
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import multiprocessing
from config import (redis_ip,
                    redis_port,
                    redis_pwd,
                    data_cfg,
                    logs,
                    pretrained_models,
                    train_result,
                    export_result,
                    train_host_msg,
                    train_action_opt_topic_name,
                    train_action_result_topic_name,
                    train_data_download_topic_name,
                    export_host_msg,
                    export_action_opt_topic_name,
                    export_action_result_topic_name)


from ultralytics.utils import SettingsManager
from services import trainService, exportService
settings = SettingsManager()
settings.update(datasets_dir="")


def main():
    '''
    训练平台主进程:启动导出任务、训练任务
    '''
    # 创建必要的目录
    for dir in [data_cfg, logs, pretrained_models, train_result, export_result]:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
    export_service = exportService(redis_ip,
                                   redis_port,
                                   redis_pwd,
                                   logs,
                                   export_host_msg,
                                   export_action_opt_topic_name,
                                   export_action_result_topic_name)
    train_service = trainService(redis_ip,
                                 redis_port,
                                 redis_pwd,
                                 logs,
                                 train_host_msg,
                                 train_action_opt_topic_name,
                                 train_action_result_topic_name,
                                 train_data_download_topic_name)
    # 用进程池
    pool = multiprocessing.Pool(processes=2)
    # 异步启动2个服务
    pool.apply_async(train_service.run)
    pool.apply_async(export_service.run)
    pool.close()  # 仅接受2个服务
    try:
        # 等待所有进程执行完毕
        # 因为任务是无限循环，所以这里会一直等待，
        pool.join()
    except Exception as ex:
        pool.terminate()
        pool.join()
    # 启动导出任务进程
    # export_service = exportService(redis_ip,
    #                                redis_port,
    #                                redis_pwd,
    #                                logs,
    #                                export_host_msg,
    #                                export_action_opt_topic_name,
    #                                export_action_result_topic_name)
    # export_proc = multiprocessing.Process(target=export_service.run)
    # export_proc.daemon = True
    # export_proc.start()
    # # 启动训练任务进程
    # train_service = trainService(redis_ip,
    #                              redis_port,
    #                              redis_pwd,
    #                              logs,
    #                              train_host_msg,
    #                              train_action_opt_topic_name,
    #                              train_action_result_topic_name,
    #                              train_data_download_topic_name)
    # train_service.run()


if __name__ == '__main__':
    main()
