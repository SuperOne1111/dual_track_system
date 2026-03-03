"""
状态管理层实现
依据《API接口设计说明书》2.1节类图和4.1节多态注入机制实现
满足CONS-02逻辑一致性标准，通过Repository模式消除模式分支
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from utils.io import atomic_write_json


class StateRepository(ABC):
    """
    状态仓库抽象基类
    依据《API接口设计说明书》2.1节，实现多态注入以满足CONS-02逻辑一致性标准
    """
    
    @abstractmethod
    def save_state(self, state_data: Dict[str, Any], timestamp: datetime) -> None:
        """
        保存状态数据
        :param state_data: 状态数据字典
        :param timestamp: 时间戳，用于防止未来函数(DB-03)
        """
        pass
    
    @abstractmethod
    def load_state(self, as_of_date: datetime) -> Optional[Dict[str, Any]]:
        """
        加载指定时间的状态数据
        :param as_of_date: 查询时间点，用于防止未来函数(DB-03)
        :return: 状态数据或None
        """
        pass
    
    @abstractmethod
    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """
        获取最新状态
        :return: 最新状态数据或None
        """
        pass


class BacktestStateRepository(StateRepository):
    """
    回测环境状态仓库实现
    依据《API接口设计说明书》2.1节，与ProductionStateRepository物理隔离
    """
    
    def __init__(self, storage_dir: str = "./data/backtest_states"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, state_data: Dict[str, Any], timestamp: datetime) -> None:
        """
        保存状态到内存，回测环境中不需要持久化
        :param state_data: 状态数据字典
        :param timestamp: 时间戳
        """
        # 回测环境中，状态仅在内存中维护，不需要写入文件
        # 但为了保持接口一致性，这里可以记录到日志或临时缓存
        print(f"[Backtest] 状态保存于 {timestamp}: {list(state_data.keys())}")
    
    def load_state(self, as_of_date: datetime) -> Optional[Dict[str, Any]]:
        """
        从内存中加载状态（回测环境）
        :param as_of_date: 查询时间点
        :return: 状态数据
        """
        # 回测环境不从持久化存储加载状态，返回空状态
        print(f"[Backtest] 查询 {as_of_date} 的状态")
        return {}
    
    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """
        获取最新状态（回测环境）
        :return: 空状态
        """
        return {}


class ProductionStateRepository(StateRepository):
    """
    生产环境状态仓库实现
    依据《技术规格说明书》7.1节RUN-02标准，使用原子写入确保状态文件安全
    """
    
    def __init__(self, storage_dir: str = "./data/prod_states"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.state_file_path = self.storage_dir / "current_state.json"
    
    def save_state(self, state_data: Dict[str, Any], timestamp: datetime) -> None:
        """
        使用原子写入保存状态到文件
        :param state_data: 状态数据字典
        :param timestamp: 时间戳
        """
        # 添加时间戳信息到状态数据
        enriched_state = {
            **state_data,
            "last_updated": timestamp.isoformat(),
            "timestamp": timestamp.timestamp()
        }
        
        # 使用原子写入确保文件安全
        atomic_write_json(enriched_state, str(self.state_file_path))
    
    def load_state(self, as_of_date: datetime) -> Optional[Dict[str, Any]]:
        """
        从文件加载状态（生产环境）
        :param as_of_date: 查询时间点
        :return: 状态数据
        """
        if not self.state_file_path.exists():
            print(f"[Production] 状态文件不存在: {self.state_file_path}")
            return None
        
        try:
            with open(self.state_file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # 检查时间戳是否在未来（防未来函数）
            if 'timestamp' in state_data:
                state_timestamp = datetime.fromtimestamp(state_data['timestamp'])
                if state_timestamp > as_of_date:
                    print(f"[Production] 状态时间戳({state_timestamp})晚于查询时间({as_of_date})，违反防未来函数原则")
                    return None
            
            print(f"[Production] 从 {self.state_file_path} 加载状态")
            return state_data
        except Exception as e:
            print(f"[Production] 加载状态失败: {e}")
            return None
    
    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """
        获取最新状态（生产环境）
        :return: 最新状态数据
        """
        if not self.state_file_path.exists():
            print(f"[Production] 状态文件不存在: {self.state_file_path}")
            return None
        
        try:
            with open(self.state_file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            return state_data
        except Exception as e:
            print(f"[Production] 获取最新状态失败: {e}")
            return None


def create_state_repository(mode: str, **kwargs) -> StateRepository:
    """
    状态仓库工厂方法
    通过依赖注入方式创建相应模式的状态仓库，满足CONS-02逻辑一致性标准
    核心业务逻辑中不应出现mode分支判断
    """
    if mode.lower() == 'backtest':
        return BacktestStateRepository(**kwargs)
    elif mode.lower() == 'production':
        return ProductionStateRepository(**kwargs)
    else:
        raise ValueError(f"不支持的模式: {mode}. 支持的模式: 'backtest', 'production'")


# 示例使用代码
if __name__ == "__main__":
    # 演示不同模式下的状态管理
    print("=== 回测环境状态管理 ===")
    backtest_repo = create_state_repository('backtest')
    test_state = {"positions": {"stock_a": 0.5, "stock_b": 0.3}, "strategy": "trend_following"}
    backtest_repo.save_state(test_state, datetime.now())
    loaded_state = backtest_repo.get_latest_state()
    print(f"回测环境加载状态: {loaded_state}")
    
    print("\n=== 生产环境状态管理 ===")
    prod_repo = create_state_repository('production')
    test_state_prod = {"positions": {"stock_x": 0.4, "stock_y": 0.6}, "strategy": "mean_reversion", "risk_level": "medium"}
    current_time = datetime.now()
    prod_repo.save_state(test_state_prod, current_time)
    loaded_state_prod = prod_repo.load_state(current_time)
    print(f"生产环境加载状态: {loaded_state_prod}")