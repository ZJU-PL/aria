#!/usr/bin/env python3
import os
import sys
import logging
import time
import csv
import signal
import multiprocessing
import shutil
import psutil
import argparse
from datetime import datetime
from tempfile import template
from tqdm import tqdm
import contextlib
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
import Mate_new as mate_solver
from env_config import setup_environment
prove_run = mate_solver.prove_run

# 加载配置
config = setup_environment()

# 移除了setup_task_logger函数，因为在多进程环境中每个进程会独立处理日志

class TaskTimeoutError(Exception):
    """任务超时异常"""
    pass

def timeout_handler(signum, frame):
    """超时信号处理函数"""
    raise TaskTimeoutError("任务执行超时")

def cleanup_process_tree(pid):
    """清理进程及其所有子进程"""
    try:
        if psutil.pid_exists(pid):
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # 首先尝试优雅地终止所有子进程
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # 等待一段时间让进程优雅退出
            gone, alive = psutil.wait_procs(children, timeout=3)
            
            # 强制杀死仍然存活的进程
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            
            # 最后终止父进程
            try:
                parent.terminate()
                parent.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
                    
            logging.debug(f"已清理进程树 PID: {pid}")
    except Exception as e:
        logging.error(f"清理进程树失败 PID: {pid}, 错误: {e}")

def run_task_with_timeout(folder, template_name, task_timeout, result_queue, strategy_mode, baseline_mode=None):
    """在独立进程中运行任务的函数"""
    try:
        # 生成日志文件名（以开始时间命名）
        start_datetime = datetime.now()
        log_filename = start_datetime.strftime("%Y%m%d_%H%M%S.log")
        log_filepath = os.path.join(folder, log_filename)
        
        # 配置日志记录器：同时输出到控制台和文件
        import logging
        
        # 配置根日志记录器，因为Mate_new.py使用的是根logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()  # 清除现有处理器
        root_logger.setLevel(logging.INFO)
        
        # 添加文件处理器到根logger
        file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                         datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # 添加控制台处理器到根logger
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                            datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # 执行prove_run
        start_time = time.time()
        # 只使用命令行参数控制baseline模式
        actual_baseline_mode = baseline_mode if baseline_mode is not None else False
        
        if actual_baseline_mode:
            logging.info(f"🎯 Baseline模式: 开始执行任务: {os.path.basename(folder)}, 超时时间: {task_timeout}秒")
        else:
            logging.info(f"开始执行任务: {os.path.basename(folder)}, 策略模式: {strategy_mode}, 超时时间: {task_timeout}秒")
        final_status = prove_run(folder, template_name, 0, strategy_mode=strategy_mode, baseline_only=actual_baseline_mode)
        end_time = time.time()
        
        # 关闭处理器
        file_handler.close()
        root_logger.removeHandler(file_handler)
        root_logger.removeHandler(console_handler)
        
        # 将结果放入队列
        result_queue.put((folder, final_status, end_time - start_time, None))
        
    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        # 将错误结果放入队列
        result_queue.put((folder, False, elapsed_time, str(e)))

def process_single_task(task_info):
    """处理单个任务的函数，用于多进程执行，使用进程级超时控制"""
    folder, template_name, strategy_mode, baseline_mode = task_info
    task_timeout = config['TASK_TIMEOUT']
    
    # 创建进程间通信队列
    result_queue = multiprocessing.Queue()
    
    # 创建子进程执行任务
    process = multiprocessing.Process(
        target=run_task_with_timeout,
        args=(folder, template_name, task_timeout, result_queue, strategy_mode, baseline_mode)
    )
    
    start_time = time.time()
    process.start()
    
    try:
        # 等待进程完成或超时
        process.join(timeout=task_timeout)
        
        if process.is_alive():
            # 进程超时，强制终止整个进程树
            logging.warning(f"任务超时，强制终止进程树: {os.path.basename(folder)}")
            cleanup_process_tree(process.pid)
            process.join(timeout=5)  # 给进程5秒时间优雅退出
            
            if process.is_alive():
                # 如果还没退出，强制杀死
                process.kill()
                process.join()
            
            elapsed_time = time.time() - start_time
            return folder, False, elapsed_time, f"任务超时 ({task_timeout}秒)"
        
        # 进程正常完成，获取结果
        if not result_queue.empty():
            return result_queue.get()
        else:
            # 进程完成但没有结果，可能是异常退出
            elapsed_time = time.time() - start_time
            return folder, False, elapsed_time, "进程异常退出，无返回结果"
            
    except Exception as e:
        # 处理异常
        if process.is_alive():
            cleanup_process_tree(process.pid)
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        
        elapsed_time = time.time() - start_time
        return folder, False, elapsed_time, f"进程管理异常: {str(e)}"

def find_template_folders(root_path, template_name="template.smt2"):
    """递归查找包含template.smt2文件的文件夹"""
    template_folders = []
    
    for root, dirs, files in os.walk(root_path):
        if template_name in files:
            template_folders.append(root)
    
    return sorted(template_folders)

def copy_folder_for_experiment(original_path):
    """
    复制原始文件夹到result_files目录下的时间戳命名文件夹
    返回新文件夹的路径
    """
    # 获取脚本所在的根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_files_dir = os.path.join(script_dir, "result_files")
    
    # 创建result_files目录（如果不存在）
    os.makedirs(result_files_dir, exist_ok=True)
    
    # 获取原始文件夹的最后一层名称
    original_folder_name = os.path.basename(original_path.rstrip(os.sep))
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建新的文件夹名称：时间戳_原始文件夹名
    new_folder_name = f"{timestamp}_{original_folder_name}"
    new_folder_path = os.path.join(result_files_dir, new_folder_name)
    
    # 复制文件夹
    print(f"正在复制文件夹: {original_path} -> {new_folder_path}")
    shutil.copytree(original_path, new_folder_path)
    print(f"文件夹复制完成: {new_folder_path}")
    
    return new_folder_path

def save_results_to_csv(results, output_path, original_root_path):
    """将结果保存到CSV文件"""
    # 按照文件夹名称的字母顺序排序
    sorted_results = sorted(results, key=lambda x: os.path.basename(x[0]))
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)        
        for folder, status, duration in sorted_results:
            result_str = "unsat" if status else ""
            # 获取最后两层文件夹名称作为任务名称
            path_parts = folder.rstrip(os.sep).split(os.sep)
            if len(path_parts) >= 2:
                folder_name = os.path.join(path_parts[-2], path_parts[-1])
            else:
                folder_name = os.path.basename(folder)  # 如果路径层级不够，回退到原来的方式
            
            # 计算从原始根目录开始的相对路径
            try:
                relative_path = os.path.relpath(folder, original_root_path)
            except ValueError:
                # 如果无法计算相对路径（比如在不同的驱动器上），使用绝对路径
                relative_path = folder
            
            writer.writerow([folder_name, result_str, f"{duration:.2f}", relative_path])

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ProofMate实验运行器')
    parser.add_argument('--baseline', action='store_true', 
                       help='启用baseline模式（仅运行初始验证）')
    parser.add_argument('--root-path', type=str, 
                       default="/home/ssdllm/ProofMate/preprocessed/all-int",
                       help='原始文件夹路径')
    parser.add_argument('--strategy-mode', type=str, 
                       choices=['default', 'zero_shot', 'naive'],
                       default='default',
                       help='选择提示词策略模式: default(默认), zero_shot(零样本), naive(朴素)')
    args = parser.parse_args()
    
    # 设置多进程启动方法（在某些系统上需要）
    multiprocessing.set_start_method('spawn', force=True)
    # 使用命令行参数设置原始文件夹路径
    # original_root_path = "/home/ssdllm/ProofMate/int-ben-llm/0911-10mins-llm-vmcai15dt/vmcai15-dt/hipspec/nosg"
    # original_root_path = "/home/ssdllm/ProofMate/int-ben-llm/0911-10mins-llm-vmcai15dt/vmcai15-dt/"
    # original_root_path = "/home/ssdllm/ProofMate/int-ben-llm/0911-10mins-llm-vmcai15dt/ssd_debug_test2"
    # original_root_path = "/home/ssdllm/ProofMate/preprocessed/autoproof"
    # original_root_path = "/home/ssdllm/ProofMate/preprocessed/dtt"
    # original_root_path = "/home/ssdllm/ProofMate/preprocessed/ind-ben"
    # original_root_path = "/home/ssdllm/ProofMate/preprocessed/vmcai15-dt"
    # 原始文件夹路径
    # original_root_path = "/home/ssdllm/ProofMate/int-ben-llm/0922dtt/dtt_quick"
    # original_root_path = "/home/ssdllm/ProofMate/int-ben-llm/0911-10mins-llm-vmcai15dt/ssd_debug_test2"
    # original_root_path = "/home/ssdllm/ProofMate/preprocessed/autoproof"
    # original_root_path = "/home/ssdllm/ProofMate/preprocessed/debug_test"
    original_root_path = args.root_path
    
    # 复制文件夹到result_files目录，避免污染原始文件
    root_path = copy_folder_for_experiment(original_root_path)

    template_name = "template"
    folders = find_template_folders(root_path, template_name + ".smt2")
    
    print(f"找到 {len(folders)} 个待求解任务")
    
    # 显示运行模式
    if args.baseline:
        print("🎯 运行模式: Baseline模式 (仅执行初始验证)")
    else:
        print(f"🚀 运行模式: 完整模式 (包含LLM引理生成)")
        print(f"📝 策略模式: {args.strategy_mode}")
    
    # 存储所有结果
    results = []
    total_start_time = time.time()
    
    # 准备任务列表，包含baseline模式和策略模式参数
    task_list = [(folder, template_name, args.strategy_mode, args.baseline) for folder in folders]
    
    # 使用进程池并行执行，最多MAX_PARALLEL_TASKS个并行进程
    max_parallel_tasks = config['MAX_PARALLEL_TASKS']
    max_workers = min(max_parallel_tasks, len(folders), multiprocessing.cpu_count())
    print(f"使用 {max_workers} 个并行进程处理 {len(folders)} 个任务")
    
    # 添加unsat任务计数器
    unsat_count = 0
    total_tasks = len(folders)
    
    # 使用进度条和进程池
    with tqdm(total=len(folders), desc=f"处理任务 (unsat: {unsat_count}/{total_tasks})", unit="task") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_folder = {executor.submit(process_single_task, task): task[0] 
                              for task in task_list}
            
            # 收集结果
            for future in as_completed(future_to_folder):
                folder_path = future_to_folder[future]
                try:
                    folder, final_status, duration, error = future.result()
                    
                    if error:
                        print(f"\n任务执行出错: {os.path.basename(folder)} - 错误: {error}")
                        # 使用实际的执行时间，而不是硬编码为0.0
                        results.append((folder, False, duration))
                    else:
                        # 记录结果
                        results.append((folder, final_status, duration))
                        
                        # 如果是unsat结果，更新计数器
                        if final_status:
                            unsat_count += 1
                        
                        # 显示任务完成反馈
                        result_text = "unsat" if final_status else "失败"
                        print(f"\n任务完成: {os.path.basename(folder)} - {result_text} - 用时: {duration:.2f}秒")
                        
                except Exception as e:
                    print(f"\n任务执行出错: {os.path.basename(folder_path)} - 错误: {str(e)}")
                    # 对于无法获取结果的异常情况，时间记录为-1
                    results.append((folder_path, False, -1))
                
                # 更新进度条描述和进度
                pbar.set_description(f"处理任务 (unsat: {unsat_count}/{total_tasks})")
                pbar.update(1)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 生成CSV报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 提取数据集名称（root-path的最后一级文件夹名称）
    dataset_name = os.path.basename(os.path.normpath(original_root_path))
    if args.baseline:
        csv_filename = f"results_{timestamp}_{dataset_name}_baseline.csv"
    else:
        csv_filename = f"results_{timestamp}_{dataset_name}_{args.strategy_mode}.csv"
    
    # 创建result_csv文件夹（如果不存在）
    result_csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_csv")
    os.makedirs(result_csv_dir, exist_ok=True)
    
    csv_filepath = os.path.join(result_csv_dir, csv_filename)
    save_results_to_csv(results, csv_filepath, original_root_path)
    
    # 统计结果
    successful_count = sum(1 for _, status, _ in results if status)
    total_count = len(results)
    
    print(f"\n=== 执行完成 ===")
    print(f"总任务数: {total_count}")
    print(f"成功求解: {successful_count}")
    print(f"失败任务: {total_count - successful_count}")
    print(f"总执行时间: {total_duration:.2f}秒")
    print(f"平均每任务时间: {total_duration/total_count:.2f}秒")
    print(f"结果已保存到: {csv_filepath}")