"""
配置模块

提供配置加载和管理功能
"""

import yaml
import os


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为项目根目录的config.yaml
    
    Returns:
        dict: 配置字典
    """
    if config_path is None:
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(root_dir, 'config.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


__all__ = ['load_config']
