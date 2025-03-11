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
import subprocess
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

# 全局变量，用于存储当前运行的训练进程
active_processes = {}

# 确保必要的目录存在
def ensure_directories():
    """确保必要的目录存在"""
    for directory in [data_cfg, train_result, pretrained_models, 'log']:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

# 回调函数，用于处理训练任务的状态变化
def task_callback(action, task_id, task_name, status, message, create_time):
    """训练任务回调函数
    
    Args:
        action: 动作类型，如'start'或'stop'
        task_id: 任务ID
        task_name: 任务名称
        status: 状态码，1表示成功，0表示失败
        message: 状态消息
        create_time: 创建时间
    """
    rds = Redis(host=redis_ip, port=redis_port, password=redis_pwd, decode_responses=True)
    result = {
        'taskId': task_id,
        'taskName': task_name,
        'status': status,
        'message': message,
        'createTime': create_time
    }
    rds.xadd(train_action_result_topic_name, {'result': json.dumps(result)}, maxlen=100)
    rds.close()

# 生成训练脚本
def generate_train_script(task_id, task_name, net_type, train_type, model_id, parameters):
    """生成训练脚本
    
    Args:
        task_id: 任务ID
        task_name: 任务名称
        net_type: 网络类型，如'yolov8s'
        train_type: 训练类型，'Init'表示初始化训练，其他表示迭代训练
        model_id: 模型ID，迭代训练时使用
        parameters: 训练参数
    
    Returns:
        str: 生成的脚本文件路径
    """
    script_path = f"./train_scripts/{task_id}_{task_name}.py"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    
    train_params = {
        'data': f'{data_cfg}/{task_id}_{task_name}.yaml',
        'project': f'{train_result}',
        'name': f'{task_id}_{task_name}',
        'task': 'detect'
    }
    
    with open(script_path, 'w', encoding='utf-8') as file:
        file.write("# 此文件由程序自动生成.\n")
        file.write("# 请勿手动修改.\n\n")
        
        # 导入必要的库
        file.write('import os\n')
        file.write('import shutil\n')
        file.write('import redis\n')
        file.write('from ultralytics import YOLO\n\n')
        
        # 连接Redis
        file.write(f'rds = redis.Redis("{redis_ip}", {redis_port}, decode_responses=True)\n\n')
        
        # 设置模型配置和路径
        if train_type == 'Init':  # 初始化训练
            if net_type.startswith('yolov5'):
                file.write(f"model_cfg = '{pretrained_models}/{net_type}u.yaml'\n")
                file.write(f"model_path = '{pretrained_models}/{net_type}u.pt'\n")
            else:
                file.write(f"model_cfg = '{pretrained_models}/{net_type}.yaml'\n")
                file.write(f"model_path = '{pretrained_models}/{net_type}.pt'\n")
        else:  # 迭代训练
            iter_pre_model = find_pt('export_results', model_id)
            if net_type.startswith('yolov5'):
                file.write(f"model_cfg = '{pretrained_models}/{net_type}u.yaml'\n")
                file.write(f"model_path = '{iter_pre_model}'\n")
            else:
                file.write(f"model_cfg = '{pretrained_models}/{net_type}.yaml'\n")
                file.write(f"model_path = '{iter_pre_model}'\n")
        
        # 设置训练参数
        for key, value in parameters.items():
            train_params[key] = value
        
        file.write(f'train_params = {train_params}\n\n')
        
        # 加载模型并训练
        file.write('model = YOLO(model_cfg).load(model_path)\n')
        file.write('model.train(**train_params)\n\n')
        
        # 训练完成后的模型移动操作
        model_trained_path = '/'.join([train_params['project'], train_params['name'], 'weights', 'best.pt'])
        file.write(f"model_trained_path = '{model_trained_path}'\n")
        file.write(f"if not os.path.exists('{'export_results'}'):\n")
        file.write(f"    os.makedirs('{'export_results'}')\n")
        file.write(f"model_rename = '{task_id}_{task_name}'\n")
        file.write("model_rename = ''.join([model_rename,'.pt'])\n")
        file.write(f"model_rename_path = '/'.join(['export_results',model_rename])\n")
        file.write("shutil.copy(model_trained_path,model_rename_path)\n")
    
    return script_path

# 启动训练进程
def start_training_process(task_id, task_name, net_type, train_type, model_id, parameters, labels):
    """启动训练进程
    
    Args:
        task_id: 任务ID
        task_name: 任务名称
        net_type: 网络类型
        train_type: 训练类型
        model_id: 模型ID
        parameters: 训练参数
        labels: 标签列表
    
    Returns:
        bool: 是否成功启动训练进程
    """
    # 生成训练脚本
    script_path = generate_train_script(task_id, task_name, net_type, train_type, model_id, parameters)
    
    # 创建日志文件
    log_dir = "./log"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = f"{log_dir}/train_log_{task_id}_{task_name}.txt"
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # 启动训练进程
    process_key = f"{task_id}_{task_name}"
    rds = Redis(host=redis_ip, port=redis_port, password=redis_pwd, decode_responses=True)
    
    # 设置训练进度初始状态
    rds.hset(f'{process_key}_train_progress', mapping={'status': 1, 'epoch': 0})
    
    # 启动进程
    process = subprocess.Popen(
        args=["python", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # 记录进程信息
    active_processes[process_key] = {
        'process': process,
        'task_id': task_id,
        'task_name': task_name,
        'log_file': log_file,
        'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    }
    
    # 创建监控线程
    monitor_thread = Thread(target=monitor_training_process, args=(process_key, labels, parameters))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # 发送启动成功消息
    create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    task_callback('start', task_id, task_name, 1, 
                 f'当前训练任务:{task_id}_{task_name}启动成功,进程已启动.', create_time)
    
    return True

# 监控训练进程
def monitor_training_process(process_key, labels, parameters):
    """监控训练进程
    
    Args:
        process_key: 进程键名，格式为{task_id}_{task_name}
        labels: 标签列表
        parameters: 训练参数
    """
    if process_key not in active_processes:
        return
    
    process_info = active_processes[process_key]
    process = process_info['process']
    log_file = process_info['log_file']
    task_id = process_info['task_id']
    task_name = process_info['task_name']
    
    # 构建正则表达式用于匹配训练进度
    re_list = []
    re_list.append(f"(\\d+)/{parameters['epochs']}")
    re_list.append(f"^      Epoch")
    re_list.append(f"^                 Class")
    re_list.append(f"^                   all")
    for label in labels:
        re_list.append(f'^                   {label}')
    re_format = "|".join(re_list)
    
    # 连接Redis
    rds = Redis(host=redis_ip, port=redis_port, password=redis_pwd, decode_responses=True)
    
    # 监控进程输出
    import re
    for line in process.stdout:
        # 写入日志文件
        log_file.write(line)
        log_file.flush()
        
        # 匹配训练进度
        match_ret = re.search(re_format, line)
        if match_ret and match_ret.group(1):
            current_epoch = int(match_ret.group(1))
            # 更新Redis中的训练进度
            rds.hset(f'{process_key}_train_progress', mapping={'status': 1, 'epoch': current_epoch})
    
    # 等待进程结束
    returncode = process.wait()
    status = 0 if returncode == 0 else -1
    
    # 更新训练状态
    rds.hset(f'{process_key}_train_progress', mapping={'status': status, 'epoch': 0})
    
    # 清理进程信息
    if process_key in active_processes:
        # 关闭日志文件
        log_file.close()
        
        # 发送训练结束消息
        if status == 0:
            rds.xadd(train_result_topic_name, {
                'result': f'当前训练任务:{process_key}正常结束'
            }, maxlen=100)
        else:
            rds.xadd(train_result_topic_name, {
                'result': f'当前训练任务:{process_key}异常退出'
            }, maxlen=100)
        
        # 删除进程信息
        del active_processes[process_key]
    
    # 删除Redis中的训练进度信息
    rds.hdel(f'{process_key}_train_progress', *rds.hkeys(f'{process_key}_train_progress'))
    rds.close()

# 停止训练进程
def stop_training_process(task_id, task_name):
    """停止训练进程
    
    Args:
        task_id: 任务ID
        task_name: 任务名称
    
    Returns:
        bool: 是否成功停止训练进程
    """
    process_key = f"{task_id}_{task_name}"
    
    if process_key not in active_processes:
        return False
    
    process_info = active_processes[process_key]
    process = process_info['process']
    log_file = process_info['log_file']
    
    # 尝试优雅地终止进程
    try:
        # 获取进程及其所有子进程
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        
        # 先终止子进程
        for child in children:
            child.terminate()
        
        # 等待子进程结束
        gone, alive = psutil.wait_procs(children, timeout=3)
        
        # 如果还有存活的子进程，强制终止
        for p in alive:
            p.kill()
        
        # 终止父进程
        parent.terminate()
        
        # 等待父进程结束
        parent.wait(timeout=5)
        
        # 如果父进程还存活，强制终止
        if parent.is_running():
            parent.kill()
    except psutil.NoSuchProcess:
        pass
    
    # 关闭日志文件
    log_file.close()
    
    # 连接Redis
    rds = Redis(host=redis_ip, port=redis_port, password=redis_pwd, decode_responses=True)
    
    # 更新训练状态
    rds.hset(f'{process_key}_train_progress', mapping={'status': -1, 'epoch': 0})
    
    # 发送停止成功消息
    create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    task_callback('stop', task_id, task_name, 1, 
                 f'当前训练任务:{process_key}已停止', create_time)
    
    # 删除Redis中的训练进度信息
    rds.hdel(f'{process_key}_train_progress', *rds.hkeys(f'{process_key}_train_progress'))
    rds.close()
    
    # 删除进程信息
    del active_processes[process_key]
    
    return True

# 处理训练任务消息
def handle_train_message(message):
    """处理训练任务消息
    
    Args:
        message: 消息内容
    """
    try:
        # 解析消息
        data = json.loads(message['data'])
        action = data.get('action')
        task_id = data.get('taskId')
        task_name = data.get('taskName')
        
        # 处理开始训练动作
        if action == 'start':
            # 检查是否已有同名任务在运行
            process_key = f"{task_id}_{task_name}"
            if process_key in active_processes:
                create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                task_callback('start', task_id, task_name, 0, 
                             f'当前训练任务:{process_key}已在运行中，无法重复启动', create_time)
                return
            
            # 获取训练参数
            net_type = data.get('netType', 'yolov8s')
            train_type = data.get('trainType', 'Init')
            model_id = data.get('modelId', '')
            parameters = data.get('parameters', {})
            labels = data.get('labels', [])
            
            # 启动训练进程
            start_training_process(task_id, task_name, net_type, train_type, model_id, parameters, labels)
        
        # 处理停止训练动作
        elif action == 'stop':
            # 检查任务是否存在
            process_key = f"{task_id}_{task_name}"
            if process_key not in active_processes:
                create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                task_callback('stop', task_id, task_name, 0, 
                             f'当前训练任务:{process_key}不存在或已停止', create_time)
                return
            
            # 停止训练进程
            stop_training_process(task_id, task_name)
    
    except Exception as e:
        # 记录异常
        print(f"处理训练消息异常: {e}")

# 信号处理函数
def signal_handler(sig, frame):
    """处理信号
    
    Args:
        sig: 信号
        frame: 帧
    """
    print("接收到终止信号，正在清理资源...")
    
    # 停止所有训练进程
    for process_key in list(active_processes.keys()):
        process_info = active_processes[process_key]
        task_id = process_info['task_id']
        task_name = process_info['task_name']
        stop_training_process(task_id, task_name)
    
    print("资源清理完成，退出程序")
    exit(0)

# 主函数
def main():
    """主函数"""
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 确保必要的目录存在
    ensure_directories()
    
    # 连接Redis
    rds = Redis(host=redis_ip, port=redis_port, password=redis_pwd, decode_responses=True)
    
    # 创建消费者组
    try:
        rds.xgroup_create(train_action_opt_topic_name, 'train_service_group', mkstream=True)
    except Exception:
        # 消费者组可能已存在
        pass
    
    print("训练服务已启动，等待训练任务...")
    
    # 循环监听消息
    while True:
        try:
            # 读取消息
            messages = rds.xreadgroup(
                'train_service_group', 'train_service_consumer',
                {train_action_opt_topic_name: '>'},
                count=1, block=1000
            )
            
            # 处理消息
            if messages:
                for stream, message_list in messages:
                    for message_id, message in message_list:
                        # 处理消息
                        handle_train_message(message)
                        
                        # 确认消息
                        rds.xack(train_action_opt_topic_name, 'train_service_group', message_id)
        
        except Exception as e:
            # 记录异常
            print(f"监听消息异常: {e}")
            time.sleep(1)

# 程序入口
if __name__ == "__main__":
    main()