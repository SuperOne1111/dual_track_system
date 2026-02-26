"""
日志配置

配置日志输出
"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = 'dual_track',
                level: str = 'INFO',
                log_file: str = None) -> logging.Logger:
    """
    设置日志
    
    Args:
        name: 日志名称
        level: 日志级别
        log_file: 日志文件路径
    
    Returns:
        Logger: 日志对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有处理器
    logger.handlers = []
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
