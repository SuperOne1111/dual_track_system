"""风险管理模块实现 - 四级行业双轨筛选系统

根据《技术规格说明书》v1.0 第5章动态调仓与风控实现RiskManager类。
包含波动率目标控制、组合风险监控、异常交易检测等功能。

依据《dual-track-api-design》2.1 节类图定义RiskManager接口
"""

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass




@dataclass
class IndustryPosition:
    """持仓头寸定义"""
    sector_id: int
    name: str
    level: int
    score: float
    entry_date: date
    entry_price: float
    current_price: float
    weight: float
    holding_days: int = 0
    atr_20: Optional[float] = None  # 20日平均真实波幅
    current_score: Optional[float] = None  # 当前评分


class RiskManager(ABC):
    """风险管理器抽象基类 - 满足CONS-02逻辑一致性标准"""
    
    @abstractmethod
    def apply_volatility_target(
        self, 
        positions: List[IndustryPosition], 
        forecast_vol: float,
        as_of_date: date
    ) -> float:
        """应用波动率目标控制
        
        Args:
            positions: 当前持仓列表
            forecast_vol: 预测组合波动率
            as_of_date: 当前日期
            
        Returns:
            target_position: 目标仓位比例
        """
        pass
    
    @abstractmethod
    def monitor_portfolio_risk(
        self, 
        positions: List[IndustryPosition],
        as_of_date: date
    ) -> Dict[str, float]:
        """监控组合风险指标
        
        Args:
            positions: 当前持仓列表
            as_of_date: 当前日期
            
        Returns:
            风险指标字典
        """
        pass
    
    @abstractmethod
    def should_sell_atr_drawdown(self, industry: IndustryPosition) -> bool:
        """ATR动态止损判断
        
        依据《dual-track-tech-spec》5.1节，当价格从入场点的最大回撤超过1.5倍ATR时触发
        """
        pass
    
    @abstractmethod
    def should_sell_fixed_drawdown(self, industry: IndustryPosition) -> bool:
        """固定比例止损判断
        
        依据《dual-track-tech-spec》5.1节，当回撤超过8%时触发
        """
        pass
    
    @abstractmethod
    def should_sell_on_score_drop(self, industry: IndustryPosition) -> bool:
        """评分跌破判断
        
        依据《dual-track-tech-spec》5.1节，当评分低于60分时触发
        """
        pass
    
    @abstractmethod
    def should_force_exit_on_vol_spikes(self, industry: IndustryPosition) -> bool:
        """异常波动强制退出判断
        
        依据《dual-track-tech-spec》5.1节，当单日波动超过2.5倍ATR时触发
        """
        pass
    
    @abstractmethod
    def is_position_expired(self, industry: IndustryPosition) -> bool:
        """持仓过期判断
        
        依据《dual-track-tech-spec》5.1节，左侧轨道持仓超过60天时触发
        """
        pass


class VolatilityRiskManager(RiskManager):
    """波动率风险管理者实现"""
    
    def __init__(self, config: Dict):
        """初始化风险管理者
        
        Args:
            config: 风险管理配置参数
        """
        self.config = config
        self.volatility_target = config.get('volatility_target', 0.15)  # 15%波动率目标
        self.atr_multiplier = config.get('atr_multiplier', 1.5)  # ATR止损倍数
        self.fixed_drawdown_threshold = config.get('fixed_drawdown_threshold', 0.08)  # 固定止损阈值8%
        self.score_exit_threshold = config.get('score_exit_threshold', 60.0)  # 评分退出阈值
        self.vol_spike_multiplier = config.get('vol_spike_multiplier', 2.5)  # 异常波动倍数
        self.max_holding_days = config.get('max_holding_days', 60)  # 最大持仓天数
        self.ewma_lambda = config.get('ewma_lambda', 0.94)  # EWMA平滑参数
    
    def apply_volatility_target(
        self, 
        positions: List[IndustryPosition], 
        forecast_vol: float,
        as_of_date: date
    ) -> float:
        """应用波动率目标控制 - 依据《dual-track-tech-spec》4.3节
        
        使用EWMA预测模型(λ=0.94, 20日窗口)进行波动率预测
        """
        if forecast_vol <= 0:
            return 1.0  # 如果预测波动率为0或负数，则满仓
        
        # 计算目标仓位：min(1.0, 目标波动率/预测波动率)
        target_position = min(1.0, self.volatility_target / forecast_vol)
        
        return target_position
    
    def calculate_ewma_volatility(
        self, 
        returns: pd.Series, 
        lambda_param: float = 0.94
    ) -> float:
        """计算EWMA预测波动率 - 依据《dual-track-tech-spec》4.3节
        
        Args:
            returns: 收益率序列
            lambda_param: EWMA平滑参数
            
        Returns:
            预测波动率
        """
        if len(returns) < 2:
            return 0.0
        
        # 使用EWMA公式计算波动率
        squared_returns = returns ** 2
        ewma_variance = 0.0
        
        for i, ret in enumerate(squared_returns[::-1]):  # 从最新到最旧
            weight = (1 - lambda_param) * (lambda_param ** i)
            ewma_variance += weight * ret
        
        # 归一化权重
        total_weight = sum((1 - lambda_param) * (lambda_param ** i) for i in range(len(squared_returns)))
        if total_weight > 0:
            ewma_variance /= total_weight
        
        return np.sqrt(ewma_variance)
    
    def monitor_portfolio_risk(
        self, 
        positions: List[IndustryPosition],
        as_of_date: date
    ) -> Dict[str, float]:
        """监控组合风险指标"""
        if not positions:
            return {
                'concentration_risk': 0.0,
                'exposure_by_level1': {},
                'exposure_by_level2': {},
                'avg_holding_days': 0.0,
                'total_positions': 0
            }
        
        # 计算集中度风险（最大单一持仓占比）
        total_weight = sum(pos.weight for pos in positions)
        if total_weight == 0:
            return {
                'concentration_risk': 0.0,
                'exposure_by_level1': {},
                'exposure_by_level2': {},
                'avg_holding_days': 0.0,
                'total_positions': 0
            }
        
        max_single_weight = max(pos.weight for pos in positions) / total_weight if total_weight > 0 else 0.0
        
        # 按层级计算暴露度
        exposure_by_level1 = {}
        exposure_by_level2 = {}
        
        for pos in positions:
            # Level 1暴露度
            l1_key = f"L1_{pos.name[:2]}"  # 假设前两个字符代表Level 1
            exposure_by_level1[l1_key] = exposure_by_level1.get(l1_key, 0) + pos.weight
            
            # Level 2暴露度
            l2_key = f"L2_{pos.name[:4]}"  # 假设前四个字符代表Level 2
            exposure_by_level2[l2_key] = exposure_by_level2.get(l2_key, 0) + pos.weight
        
        # 计算平均持仓天数
        avg_holding_days = sum(pos.holding_days for pos in positions) / len(positions)
        
        return {
            'concentration_risk': max_single_weight,
            'exposure_by_level1': exposure_by_level1,
            'exposure_by_level2': exposure_by_level2,
            'avg_holding_days': avg_holding_days,
            'total_positions': len(positions)
        }
    
    def should_sell_atr_drawdown(self, industry: IndustryPosition) -> bool:
        """ATR动态止损判断 - 依据《dual-track-tech-spec》5.1节"""
        if not industry.atr_20 or industry.entry_price <= 0:
            return False
        
        # 计算从入场点的最大回撤
        max_drawdown = (industry.current_price - industry.entry_price) / industry.entry_price
        
        # ATR止损阈值
        atr_stop_threshold = -self.atr_multiplier * (industry.atr_20 / industry.entry_price)
        
        return max_drawdown < atr_stop_threshold
    
    def should_sell_fixed_drawdown(self, industry: IndustryPosition) -> bool:
        """固定比例止损判断 - 依据《dual-track-tech-spec》5.1节"""
        if industry.entry_price <= 0:
            return False
        
        drawdown = (industry.current_price - industry.entry_price) / industry.entry_price
        return drawdown < -self.fixed_drawdown_threshold
    
    def should_sell_on_score_drop(self, industry: IndustryPosition) -> bool:
        """评分跌破判断 - 依据《dual-track-tech-spec》5.1节"""
        if industry.current_score is None:
            return False
        
        return industry.current_score < self.score_exit_threshold
    
    def should_force_exit_on_vol_spikes(self, industry: IndustryPosition) -> bool:
        """异常波动强制退出判断 - 依据《dual-track-tech-spec》5.1节"""
        if not industry.atr_20:
            return False
        
        # 假设当日波动为当前价格相对于前一日的变化
        # 这里我们使用当前价格与入场价格的单日变化作为近似
        # 实际应用中应该使用当日最高最低价计算真实波幅
        if hasattr(industry, 'prev_close') and industry.prev_close > 0:
            daily_change = abs(industry.current_price - industry.prev_close) / industry.prev_close
        else:
            # 如果没有前收盘价，使用当前价格相对于入场价的日变化作为近似
            daily_change = abs(industry.current_price - industry.entry_price) / industry.entry_price if industry.entry_price > 0 else 0
        
        return daily_change > self.vol_spike_multiplier * industry.atr_20
    
    def is_position_expired(self, industry: IndustryPosition) -> bool:
        """持仓过期判断 - 依据《dual-track-tech-spec》5.1节"""
        return industry.holding_days >= self.max_holding_days


class CompositeRiskManager(VolatilityRiskManager):
    """复合风险管理者 - 结合多种风险管理策略"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.risk_limits = config.get('risk_limits', {
            'max_concentration': 0.20,  # 最大单一持仓20%
            'max_l1_exposure': 0.30,    # 最大L1暴露度30%
            'max_l2_exposure': 0.15,    # 最大L2暴露度15%
        })
    
    def check_position_risk(self, industry: IndustryPosition) -> Tuple[bool, str]:
        """检查单个持仓的风险状态
        
        Returns:
            (是否需要卖出, 风险类型)
        """
        # 检查ATR动态止损
        if self.should_sell_atr_drawdown(industry):
            return True, "ATR_DRAWDOWN"
        
        # 检查固定比例止损
        if self.should_sell_fixed_drawdown(industry):
            return True, "FIXED_DRAWDOWN"
        
        # 检查评分跌破
        if self.should_sell_on_score_drop(industry):
            return True, "SCORE_DROP"
        
        # 检查异常波动
        if self.should_force_exit_on_vol_spikes(industry):
            return True, "VOL_SPIKE"
        
        # 检查持仓过期（主要针对左侧轨道）
        if self.is_position_expired(industry):
            return True, "EXPIRED"
        
        return False, "OK"


# 工厂方法
def create_risk_manager(config: Dict) -> RiskManager:
    """创建风险管理者实例"""
    return CompositeRiskManager(config)