"""
卖出信号生成器实现 - 四级行业双轨筛选系统

根据《技术规格说明书》v1.0 第5.1章卖出信号，实现SellSignalGenerator抽象基类及其具体实现。
支持多种卖出信号类型（ATR动态止损、固定比例止损、持有期到期、评分跌破、异常波动）。

作者: QuantArchitect
创建日期: 2024-05-22
最后更新: 2024-05-22
合规标准: CONS-02 (逻辑一致性), DB-03 (防未来函数), STRAT-01 (双轨隔离), RUN-02 (原子写入)
"""

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from dataclasses import dataclass
from modules.data_layer import DataLoader
from modules.state_manager import StateRepository
from utils.io import atomic_write_json
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.data_layer import IndustryPosition, SelectedIndustry


@dataclass
class SellSignal:
    """卖出信号数据结构"""
    industry_id: int
    signal_type: str  # 'atr_stop_loss', 'fixed_stop_loss', 'holding_period', 'score_drop', 'vol_spike'
    reason: str
    sell_date: date
    entry_date: date
    holding_days: int
    entry_price: float
    current_price: float
    score: float  # 当前评分


class SellSignalGenerator(ABC):
    """
    卖出信号生成器抽象基类
    
    依据《dual-track-api-design》2.1 节定义的接口，实现双轨独立的卖出信号检测逻辑。
    遵循 CONS-02 逻辑一致性标准，通过多态注入实现不同策略的物理隔离。
    """
    
    @abstractmethod
    def evaluate_sell_signals(
        self, 
        positions: List['IndustryPosition'], 
        market_data: pd.DataFrame,
        scores: Dict[int, float],
        as_of_date: date
    ) -> List[SellSignal]:
        """
        评估卖出信号
        
        Args:
            positions: 当前持仓列表
            market_data: 市场数据 (包含ATR等指标)
            scores: 当前行业评分字典 {industry_id: score}
            as_of_date: 评估日期（防止未来函数）
            
        Returns:
            卖出信号列表
        """
        pass


class TrendSellSignalGenerator(SellSignalGenerator):
    """
    趋势轨道卖出信号生成器
    
    实现趋势轨道的卖出逻辑，与左侧轨道物理隔离
    依据《dual-track-tech-spec》5.1 卖出信号章节
    """
    
    def __init__(self, config: Dict):
        """
        初始化趋势轨道卖出信号生成器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.atr_multiplier = config.get('atr_stop_loss_multiplier', 1.5)
        self.fixed_stop_loss_pct = config.get('fixed_stop_loss_pct', 0.08)
        self.score_exit_threshold = config.get('score_exit_threshold', 60.0)
        self.vol_spike_multiplier = config.get('vol_spike_multiplier', 2.5)
        
    def evaluate_sell_signals(
        self, 
        positions: List['IndustryPosition'], 
        market_data: pd.DataFrame,
        scores: Dict[int, float],
        as_of_date: date
    ) -> List[SellSignal]:
        """
        评估趋势轨道卖出信号
        
        Args:
            positions: 当前持仓列表
            market_data: 市场数据 (包含ATR等指标)
            scores: 当前行业评分字典 {industry_id: score}
            as_of_date: 评估日期（防止未来函数）
            
        Returns:
            卖出信号列表
        """
        sell_signals = []
        
        for position in positions:
            industry = position.industry
            industry_id = industry.sector_id
            
            # 获取当前市场价格数据
            current_data = market_data[
                (market_data['sector_industry_id'] == industry_id) & 
                (market_data['trade_date'] == as_of_date)
            ]
            
            if current_data.empty:
                continue  # 如果没有当前日期的数据，跳过该行业
                
            current_row = current_data.iloc[0]
            current_price = current_row['front_adj_close']
            atr_20 = current_row.get('atr_20', 0.0)  # ATR可能需要单独计算
            
            # 获取当前评分
            current_score = scores.get(industry_id, 0.0)
            
            # 1. 评分跌破检查（优先级更高，因为是基本面变化）
            if self._check_score_drop(current_score):
                signal = SellSignal(
                    industry_id=industry_id,
                    signal_type='score_drop',
                    reason=f"评分跌破阈值: {current_score:.2f} < {self.score_exit_threshold}",
                    sell_date=as_of_date,
                    entry_date=industry.entry_date,
                    holding_days=industry.holding_days,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    score=current_score
                )
                sell_signals.append(signal)
                continue
                
            # 2. ATR动态止损检查
            if self._check_atr_stop_loss(position, current_price, atr_20):
                signal = SellSignal(
                    industry_id=industry_id,
                    signal_type='atr_stop_loss',
                    reason=f"价格跌破ATR动态止损线: {position.entry_price:.2f} -> {current_price:.2f}",
                    sell_date=as_of_date,
                    entry_date=industry.entry_date,
                    holding_days=industry.holding_days,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    score=current_score
                )
                sell_signals.append(signal)
                continue  # 一旦触发卖出信号，不再检查其他条件
                
            # 3. 固定比例止损检查
            if self._check_fixed_stop_loss(position, current_price):
                signal = SellSignal(
                    industry_id=industry_id,
                    signal_type='fixed_stop_loss',
                    reason=f"回撤超过8%: {(current_price/position.entry_price - 1)*100:.2f}%",
                    sell_date=as_of_date,
                    entry_date=industry.entry_date,
                    holding_days=industry.holding_days,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    score=current_score
                )
                sell_signals.append(signal)
                continue
                
            # 4. 异常波动检查
            if self._check_vol_spike(position, current_price, atr_20):
                signal = SellSignal(
                    industry_id=industry_id,
                    signal_type='vol_spike',
                    reason=f"单日波动超过2.5倍ATR: {(abs(current_price/position.entry_price - 1))*100:.2f}% vs {self.vol_spike_multiplier * atr_20/position.entry_price*100:.2f}%",
                    sell_date=as_of_date,
                    entry_date=industry.entry_date,
                    holding_days=industry.holding_days,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    score=current_score
                )
                sell_signals.append(signal)
                continue
        
        return sell_signals
    
    def _check_atr_stop_loss(self, position: 'IndustryPosition', current_price: float, atr_20: float) -> bool:
        """检查ATR动态止损条件"""
        if atr_20 <= 0:
            return False
            
        entry_price = position.entry_price
        min_since_entry = min(position.current_price, current_price)  # 这里简化，实际应跟踪最低价
        
        # 使用当前价格作为近似最低价（实际实现中需要跟踪持仓期间的最低价）
        drawdown = (min_since_entry - entry_price) / entry_price
        threshold = -self.atr_multiplier * (atr_20 / entry_price)
        
        return drawdown < threshold
    
    def _check_fixed_stop_loss(self, position: 'IndustryPosition', current_price: float) -> bool:
        """检查固定比例止损条件"""
        entry_price = position.entry_price
        drawdown = (current_price - entry_price) / entry_price
        return drawdown < -self.fixed_stop_loss_pct
    
    def _check_score_drop(self, current_score: float) -> bool:
        """检查评分跌破条件"""
        return current_score < self.score_exit_threshold
    
    def _check_vol_spike(self, position: 'IndustryPosition', current_price: float, atr_20: float) -> bool:
        """检查异常波动条件"""
        if atr_20 <= 0:
            return False
            
        # 计算当日波动（这里简化，实际应是当日最高最低价波动）
        daily_change = abs(current_price - position.current_price) / position.current_price
        atr_based_threshold = self.vol_spike_multiplier * (atr_20 / position.current_price)
        
        return daily_change > atr_based_threshold


class LeftSellSignalGenerator(SellSignalGenerator):
    """
    左侧轨道卖出信号生成器
    
    实现左侧轨道的卖出逻辑，与趋势轨道物理隔离
    依据《dual-track-tech-spec》5.1 卖出信号章节
    """
    
    def __init__(self, config: Dict):
        """
        初始化左侧轨道卖出信号生成器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.atr_multiplier = config.get('atr_stop_loss_multiplier', 1.5)
        self.fixed_stop_loss_pct = config.get('fixed_stop_loss_pct', 0.08)
        self.score_exit_threshold = config.get('score_exit_threshold', 60.0)
        self.vol_spike_multiplier = config.get('vol_spike_multiplier', 2.5)
        self.holding_period_limit = config.get('left_holding_period_limit', 60)  # 左侧轨道持有期限制
        
    def evaluate_sell_signals(
        self, 
        positions: List['IndustryPosition'], 
        market_data: pd.DataFrame,
        scores: Dict[int, float],
        as_of_date: date
    ) -> List[SellSignal]:
        """
        评估左侧轨道卖出信号
        
        Args:
            positions: 当前持仓列表
            market_data: 市场数据 (包含ATR等指标)
            scores: 当前行业评分字典 {industry_id: score}
            as_of_date: 评估日期（防止未来函数）
            
        Returns:
            卖出信号列表
        """
        sell_signals = []
        
        for position in positions:
            industry = position.industry
            industry_id = industry.sector_id
            
            # 获取当前市场价格数据
            current_data = market_data[
                (market_data['sector_industry_id'] == industry_id) & 
                (market_data['trade_date'] == as_of_date)
            ]
            
            if current_data.empty:
                continue  # 如果没有当前日期的数据，跳过该行业
                
            current_row = current_data.iloc[0]
            current_price = current_row['front_adj_close']
            atr_20 = current_row.get('atr_20', 0.0)  # ATR可能需要单独计算
            
            # 获取当前评分
            current_score = scores.get(industry_id, 0.0)
            
            # 1. 持有期到期检查（左侧轨道优先）
            if self._check_holding_period_expiry(industry):
                signal = SellSignal(
                    industry_id=industry_id,
                    signal_type='holding_period',
                    reason=f"持有期到期: {industry.holding_days} >= {self.holding_period_limit}天",
                    sell_date=as_of_date,
                    entry_date=industry.entry_date,
                    holding_days=industry.holding_days,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    score=current_score
                )
                sell_signals.append(signal)
                continue  # 一旦触发卖出信号，不再检查其他条件
                
            # 2. ATR动态止损检查
            if self._check_atr_stop_loss(position, current_price, atr_20):
                signal = SellSignal(
                    industry_id=industry_id,
                    signal_type='atr_stop_loss',
                    reason=f"价格跌破ATR动态止损线: {position.entry_price:.2f} -> {current_price:.2f}",
                    sell_date=as_of_date,
                    entry_date=industry.entry_date,
                    holding_days=industry.holding_days,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    score=current_score
                )
                sell_signals.append(signal)
                continue
                
            # 3. 固定比例止损检查
            if self._check_fixed_stop_loss(position, current_price):
                signal = SellSignal(
                    industry_id=industry_id,
                    signal_type='fixed_stop_loss',
                    reason=f"回撤超过8%: {(current_price/position.entry_price - 1)*100:.2f}%",
                    sell_date=as_of_date,
                    entry_date=industry.entry_date,
                    holding_days=industry.holding_days,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    score=current_score
                )
                sell_signals.append(signal)
                continue
                
            # 4. 评分跌破检查
            if self._check_score_drop(current_score):
                signal = SellSignal(
                    industry_id=industry_id,
                    signal_type='score_drop',
                    reason=f"评分跌破阈值: {current_score:.2f} < {self.score_exit_threshold}",
                    sell_date=as_of_date,
                    entry_date=industry.entry_date,
                    holding_days=industry.holding_days,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    score=current_score
                )
                sell_signals.append(signal)
                continue
                
            # 5. 异常波动检查
            if self._check_vol_spike(position, current_price, atr_20):
                signal = SellSignal(
                    industry_id=industry_id,
                    signal_type='vol_spike',
                    reason=f"单日波动超过2.5倍ATR: {(abs(current_price/position.entry_price - 1))*100:.2f}% vs {self.vol_spike_multiplier * atr_20/position.entry_price*100:.2f}%",
                    sell_date=as_of_date,
                    entry_date=industry.entry_date,
                    holding_days=industry.holding_days,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    score=current_score
                )
                sell_signals.append(signal)
                continue
        
        return sell_signals
    
    def _check_holding_period_expiry(self, industry: 'SelectedIndustry') -> bool:
        """检查持有期到期条件（左侧轨道特有）"""
        return industry.holding_days >= self.holding_period_limit
    
    def _check_atr_stop_loss(self, position: 'IndustryPosition', current_price: float, atr_20: float) -> bool:
        """检查ATR动态止损条件"""
        if atr_20 <= 0:
            return False
            
        entry_price = position.entry_price
        min_since_entry = min(position.current_price, current_price)  # 这里简化，实际应跟踪最低价
        
        # 使用当前价格作为近似最低价（实际实现中需要跟踪持仓期间的最低价）
        drawdown = (min_since_entry - entry_price) / entry_price
        threshold = -self.atr_multiplier * (atr_20 / entry_price)
        
        return drawdown < threshold
    
    def _check_fixed_stop_loss(self, position: 'IndustryPosition', current_price: float) -> bool:
        """检查固定比例止损条件"""
        entry_price = position.entry_price
        drawdown = (current_price - entry_price) / entry_price
        return drawdown < -self.fixed_stop_loss_pct
    
    def _check_score_drop(self, current_score: float) -> bool:
        """检查评分跌破条件"""
        return current_score < self.score_exit_threshold
    
    def _check_vol_spike(self, position: 'IndustryPosition', current_price: float, atr_20: float) -> bool:
        """检查异常波动条件"""
        if atr_20 <= 0:
            return False
            
        # 计算当日波动（这里简化，实际应是当日最高最低价波动）
        daily_change = abs(current_price - position.current_price) / position.current_price
        atr_based_threshold = self.vol_spike_multiplier * (atr_20 / position.current_price)
        
        return daily_change > atr_based_threshold


class CoolDownManager:
    """
    冷却池管理器
    
    管理卖出后进入冷却期的行业，防止短期内重复买入
    依据《dual-track-tech-spec》5.4 全局冷却池管理章节
    """
    
    def __init__(self, state_repo: StateRepository, cool_down_days: int = 10):
        """
        初始化冷却池管理器
        
        Args:
            state_repo: 状态仓库
            cool_down_days: 冷却期天数，默认10个交易日
        """
        self.state_repo = state_repo
        self.cool_down_days = cool_down_days
        self.state_key = "cool_down_records"
        
        # 初始化冷却池状态
        self.cool_down_records = self.load_cool_down_state()
    
    def load_cool_down_state(self) -> Dict[int, Dict[str, str]]:
        """从状态仓库加载冷却池记录"""
        try:
            records = self.state_repo.load(self.state_key)
            if records is None:
                return {}
            return records
        except Exception:
            return {}
    
    def is_in_cool_down(self, industry_id: int, as_of_date: date) -> bool:
        """
        检查行业是否在冷却期内
        
        Args:
            industry_id: 行业ID
            as_of_date: 查询日期
            
        Returns:
            是否在冷却期内
        """
        if industry_id not in self.cool_down_records:
            return False
            
        unlock_date_str = self.cool_down_records[industry_id]['unlock_date']
        unlock_date = date.fromisoformat(unlock_date_str)
        
        return as_of_date < unlock_date
    
    def add_to_cool_down(self, industry_id: int, sell_date: date):
        """
        将行业添加到冷却池
        
        Args:
            industry_id: 行业ID
            sell_date: 卖出日期
        """
        unlock_date = self._calculate_unlock_date(sell_date)
        
        self.cool_down_records[industry_id] = {
            'sell_date': sell_date.isoformat(),
            'unlock_date': unlock_date.isoformat()
        }
    
    def _calculate_unlock_date(self, sell_date: date) -> date:
        """计算解锁日期（跳过周末）"""
        unlock_date = sell_date
        days_added = 0
        
        while days_added < self.cool_down_days:
            unlock_date += timedelta(days=1)
            # 跳过周末（简单实现，实际应考虑节假日）
            if unlock_date.weekday() < 5:  # 0-4 是周一到周五
                days_added += 1
                
        return unlock_date
    
    def prune_expired_entries(self, current_date: date):
        """清理过期的冷却记录"""
        expired_ids = []
        for industry_id, record in self.cool_down_records.items():
            unlock_date = date.fromisoformat(record['unlock_date'])
            if current_date >= unlock_date:
                expired_ids.append(industry_id)
                
        for industry_id in expired_ids:
            del self.cool_down_records[industry_id]
    
    def update_state(self):
        """更新状态仓库中的冷却池记录"""
        self.state_repo.save(self.state_key, self.cool_down_records)
    
    def get_cool_down_state(self) -> Dict[int, Dict[str, str]]:
        """获取当前冷却池状态"""
        return self.cool_down_records