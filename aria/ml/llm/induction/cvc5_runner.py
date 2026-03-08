import subprocess
import logging
import time
import os
import signal
import psutil
from env_config import setup_environment

# 获取配置
config = setup_environment()

def run_cvc_solver_with_timeout(smt2_path, timeout=60):
    """
    使用多种CVC5/CVC4配置并行验证SMT2文件，返回第一个成功的结果
    
    Args:
        smt2_path (str): SMT2文件路径
        timeout (int): 超时时间（秒），默认60秒
        
    Returns:
        bool: 验证成功返回True，失败或超时返回False
    """
    # 从配置中获取CVC5和CVC4可执行文件路径
    cvc4_binary = config['CVC4_BINARY']
    cvc5_binary = config['CVC5_BINARY']
    
    # 定义所有验证策略配置，包含策略类型和选项
    strategies = {
        # CVC5策略
        'cvc5_simple': {
            'binary': cvc5_binary,
            'options': ["--full-saturate-quant"],
            'type': 'CVC5'
        },
        'cvc5_inductive': {
            'binary': cvc5_binary,
            'options': ["--full-saturate-quant", "--quant-ind", "--conjecture-gen"],
            'type': 'CVC5'
        },
        'cvc5_inductive_no_ematching': {
            'binary': cvc5_binary,
            'options': ["--full-saturate-quant", "--quant-ind", "--conjecture-gen", "--no-e-matching"],
            'type': 'CVC5'
        },
        # CVC4策略
        'cvc4_default': {
            'binary': cvc4_binary,
            'options': ["--quant-ind", "--quant-cf", "--conjecture-gen", "--full-saturate-quant", "--lang=smt2.6"],
            'type': 'CVC4'
        }
    }
    
    # 启动所有策略的并行进程
    processes = {}
    try:
        for strategy_name, strategy_config in strategies.items():
            binary = strategy_config['binary']
            options = strategy_config['options']
            strategy_type = strategy_config['type']
            
            command = [binary] + options + [smt2_path]
            try:
                # 使用进程组来管理子进程，便于清理
                proc = subprocess.Popen(
                    command, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True,
                    preexec_fn=os.setsid  # 创建新的进程组
                )
                processes[strategy_name] = proc
                logging.debug(f"启动{strategy_type}进程: {strategy_name} (PID: {proc.pid})")
            except FileNotFoundError:
                logging.error(f"{strategy_type}可执行文件未找到: {binary}")
                continue
            except Exception as e:
                logging.error(f"启动{strategy_name}进程失败: {e}")
                continue
        
        if not processes:
            logging.error("没有成功启动任何CVC5/CVC4进程")
            return False
        
        # 监控进程执行
        start_time = time.time()
        completed_processes = set()
        
        while time.time() - start_time < timeout:
            # 检查每个进程的状态
            for strategy_name, proc in processes.items():
                if strategy_name in completed_processes:
                    continue
                    
                if proc.poll() is not None:  # 进程已完成
                    completed_processes.add(strategy_name)
                    
                    try:
                        stdout, stderr = proc.communicate(timeout=1)
                        
                        if proc.returncode == 0 and 'unsat' in stdout:
                            # 获取策略类型用于日志
                            strategy_type = strategies[strategy_name]['type']
                            logging.info(f"{strategy_type}验证成功: unsat (策略: {strategy_name})")
                            # 清理其他进程并返回成功
                            _cleanup_processes(processes, exclude=strategy_name)
                            return True
                        else:
                            # 记录失败信息但继续等待其他进程
                            if stderr.strip():
                                logging.debug(f"策略{strategy_name}失败: {stderr.strip()}")
                            else:
                                logging.debug(f"策略{strategy_name}未返回unsat结果")
                                
                    except subprocess.TimeoutExpired:
                        logging.warning(f"获取{strategy_name}进程输出超时")
                        proc.kill()
                        completed_processes.add(strategy_name)
                    except Exception as e:
                        logging.error(f"处理{strategy_name}进程结果时出错: {e}")
                        completed_processes.add(strategy_name)
            
            # 如果所有进程都完成了，退出循环
            if len(completed_processes) == len(processes):
                break
                
            time.sleep(0.05)  # 减少CPU占用的轮询间隔
        
        # 超时或所有进程都失败
        logging.warning(f"CVC5/CVC4验证超时或失败 (耗时: {time.time() - start_time:.2f}秒)")
        return False
        
    finally:
        # 确保清理所有进程
        _cleanup_processes(processes)


def _cleanup_processes(processes, exclude=None):
    """
    清理进程字典中的所有进程，包括其子进程
    
    Args:
        processes (dict): 进程字典
        exclude (str): 要排除的进程名称
    """
    for strategy_name, proc in processes.items():
        if exclude and strategy_name == exclude:
            continue
            
        if proc.poll() is None:  # 进程仍在运行
            try:
                # 首先尝试优雅地终止进程组
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    logging.debug(f"发送SIGTERM到{strategy_name}进程组 (PGID: {os.getpgid(proc.pid)})")
                except (OSError, ProcessLookupError):
                    # 如果进程组不存在，直接终止进程
                    proc.terminate()
                    logging.debug(f"直接终止{strategy_name}进程 (PID: {proc.pid})")
                
                # 给进程一些时间优雅退出
                try:
                    proc.wait(timeout=3)
                    logging.debug(f"已优雅终止{strategy_name}进程")
                except subprocess.TimeoutExpired:
                    # 强制终止进程组
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        logging.debug(f"强制终止{strategy_name}进程组")
                    except (OSError, ProcessLookupError):
                        # 如果进程组不存在，直接杀死进程
                        proc.kill()
                        logging.debug(f"强制终止{strategy_name}进程")
                    proc.wait()
                    
            except Exception as e:
                logging.error(f"终止{strategy_name}进程时出错: {e}")
                # 最后的清理尝试：使用psutil强制清理
                try:
                    if psutil.pid_exists(proc.pid):
                        parent = psutil.Process(proc.pid)
                        children = parent.children(recursive=True)
                        for child in children:
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        parent.kill()
                        logging.debug(f"使用psutil强制清理{strategy_name}进程及其子进程")
                except Exception as cleanup_error:
                    logging.error(f"使用psutil清理{strategy_name}进程失败: {cleanup_error}")