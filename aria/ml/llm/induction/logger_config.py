import logging
import colorlog

def setup_colored_logger():
    """配置彩色日志记录器"""
    # 配置彩色日志
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    # 创建并配置日志处理器
    handler = colorlog.StreamHandler()
    handler.setFormatter(color_formatter)
    logger = colorlog.getLogger()
    logger.addHandler(handler)
    # 静默非关键日志，保留 CRITICAL
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG) # 如果需要查看网络发送信息，可以切换到DEBUG级的Log记录

    # 清除默认的处理器以避免重复输出
    logger.handlers = [handler]
    
    return logger