import subprocess
import logging
import time
import os
import signal
from pathlib import Path
from env_config import setup_environment

# 获取配置
config = setup_environment()

def run_vampire_with_timeout(smt2_path, timeout=60):
    """
    使用Vampire验证SMT2文件
    
    Args:
        smt2_path (str or Path): SMT2文件路径
        timeout (int): 超时时间（秒），默认60秒
        
    Returns:
        bool: 验证成功返回True（unsatisfiable），失败或超时返回False
    """
    # 从配置中获取Vampire可执行文件路径
    vampire_binary = config.get('VAMPIRE_BINARY')
    if not vampire_binary:
        logging.error("VAMPIRE_BINARY未在配置中找到")
        return False
    
    # 确保smt2_path是Path对象
    if isinstance(smt2_path, str):
        smt2_path = Path(smt2_path)
    
    # 构建命令，timeout需要在后面加's'
    # --input_syntax 指定输入语法类型为 smtlib2，文件路径作为最后一个参数
    command = [
        vampire_binary,
        '-t', f'{timeout}s',
        '--mode', 'portfolio',
        '--schedule', 'induction',
        '--output_mode', 'smtcomp',
        '--input_syntax', 'smtlib2',
        str(smt2_path)
    ]
    
    try:
        logging.debug(f"启动Vampire进程: {' '.join(command)}")
        
        # 使用进程组来管理子进程，便于清理
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # 创建新的进程组
        )
        
        start_time = time.time()
        
        try:
            # 等待进程完成，但设置超时
            stdout, stderr = proc.communicate(timeout=timeout + 5)  # 多给5秒缓冲
            
            elapsed_time = time.time() - start_time
            
            # 检查输出，只检查stdout中的结果，忽略stderr中的警告信息
            # Vampire在证明unsatisfiable时输出中包含"unsatisfiable"或"unsat"
            output = stdout.lower().strip()
            
            # 检查是否包含unsat结果（通常出现在最后）
            if 'unsatisfiable' in output or 'unsat' in output:
                logging.info(f"Vampire验证成功: unsatisfiable (耗时: {elapsed_time:.2f}秒, 返回码: {proc.returncode})")
                return True
            else:
                # 即使返回码为0，如果没有找到unsat，也认为失败
                logging.debug(f"Vampire未找到unsatisfiable结果 (返回码: {proc.returncode}, 耗时: {elapsed_time:.2f}秒)")
                if stdout.strip():
                    logging.debug(f"Vampire stdout (最后100字符): {stdout.strip()[-100:]}")
                if stderr.strip():
                    logging.debug(f"Vampire stderr (前200字符): {stderr.strip()[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            # 超时，终止进程
            elapsed_time = time.time() - start_time
            logging.warning(f"Vampire验证超时 (耗时: {elapsed_time:.2f}秒)")
            _cleanup_process(proc)
            return False
            
    except FileNotFoundError:
        logging.error(f"Vampire可执行文件未找到: {vampire_binary}")
        return False
    except Exception as e:
        logging.error(f"启动Vampire进程失败: {e}")
        return False


def _cleanup_process(proc):
    """
    清理进程，包括其子进程
    
    Args:
        proc: subprocess.Popen对象
    """
    if proc.poll() is None:  # 进程仍在运行
        try:
            # 首先尝试优雅地终止进程组
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                logging.debug(f"发送SIGTERM到Vampire进程组 (PGID: {os.getpgid(proc.pid)})")
            except (OSError, ProcessLookupError):
                # 如果进程组不存在，直接终止进程
                proc.terminate()
                logging.debug(f"直接终止Vampire进程 (PID: {proc.pid})")
            
            # 给进程一些时间优雅退出
            try:
                proc.wait(timeout=3)
                logging.debug(f"已优雅终止Vampire进程")
            except subprocess.TimeoutExpired:
                # 强制终止进程组
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    logging.debug(f"强制终止Vampire进程组")
                except (OSError, ProcessLookupError):
                    # 如果进程组不存在，直接杀死进程
                    proc.kill()
                    logging.debug(f"强制终止Vampire进程")
                proc.wait()
                
        except Exception as e:
            logging.error(f"终止Vampire进程时出错: {e}")
