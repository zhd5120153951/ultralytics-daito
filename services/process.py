'''
@FileName   :process.py
@Description:
@Date       :2025/03/12 13:37:26
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''


class ProcessManager:
    """_summary_
    进程管理类:负责进程的创建、启动、停止、查询等操作
    """

    def __init__(self):
        self.active_tasks = {}

    def add_task(self, task):
        """_summary_
        添加任务
        """
        self.active_tasks[task.process_key] = task

    def remove_task(self, process_key):
        """_summary_
        移除任务
        """
        if process_key in self.active_tasks:
            del self.active_tasks[process_key]

    def get_task(self, process_key):
        """_summary_
        获取任务
        """
        return self.active_tasks.get(process_key, None)

    def stop_all_tasks(self):
        """_summary_
        停止所有任务
        """
        for task in list(self.active_tasks.values()):
            task.stop_train_task()
            self.remove_task(task.process_key)
        print("All tasks have been stopped.")
