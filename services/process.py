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
        self.active_procs = {}

    def add_proc(self, process_key, process):
        """_summary_
        添加进程
        """
        self.active_procs[process_key] = process

    def remove_proc(self, process_key):
        """_summary_
        移除进程
        """
        del self.active_procs[process_key]

    def get_proc(self, process_key):
        """_summary_
        获取进程
        """
        return self.active_procs.get(process_key, None)

    def stop_proc(self, process_key):
        """_summary_

        停止进程
        """
        task_proc = self.get_proc(process_key)
        if not task_proc:
            print(f"当前任务:{process_key}进程不存在,无需退出！")
        elif task_proc.is_alive():
            task_proc.terminate()
            task_proc.join(5)
            if task_proc.is_alive():
                task_proc.kill()
            print(f"当前任务:{process_key}进程成功停止.")
        print(f"当前任务:{process_key}已完成,无需停止.")

    def stop_all_procs(self):
        """_summary_
        停止所有进程
        """
        for proc in list(self.active_procs.values()):
            proc.stop_train_task()
            self.remove_task(proc.process_key)
        print("All process have been stopped.")
