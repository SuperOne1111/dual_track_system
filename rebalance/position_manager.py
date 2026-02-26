"""
持仓管理器

管理持仓记录、权重计算和调整
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


class PositionManager:
    """持仓管理器"""
    
    def __init__(self, config: dict):
        """
        初始化持仓管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.risk_config = config.get('risk_control', {})
        self.position_config = self.risk_config.get('position', {})
        
        self.positions = {}
        self.trade_history = []
    
    def add_position(self, sector_industry_id: str, weight: float,
                    entry_date: datetime, entry_price: float,
                    sector_type: str, score: float,
                    enforce_limits: bool = True):
        """
        添加持仓
        
        Args:
            sector_industry_id: 行业ID
            weight: 权重
            entry_date: 入场日期
            entry_price: 入场价格
            sector_type: 类型（trend/left）
            score: 入场评分
        """
        if enforce_limits and not self.check_position_limits(sector_industry_id, weight, sector_type):
            raise ValueError(
                f"仓位限制不通过: sector={sector_industry_id}, type={sector_type}, weight={weight:.4f}"
            )

        self.positions[sector_industry_id] = {
            'weight': weight,
            'type': sector_type,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'current_score': score,
            'entry_score': score
        }
        
        # 记录交易
        self.trade_history.append({
            'date': entry_date,
            'sector_industry_id': sector_industry_id,
            'action': 'buy',
            'price': entry_price,
            'weight': weight,
            'type': sector_type,
            'score': score
        })
    
    def remove_position(self, sector_industry_id: str, exit_date: datetime = None,
                       exit_price: float = None, reason: str = ''):
        """
        移除持仓
        
        Args:
            sector_industry_id: 行业ID
            exit_date: 出场日期
            exit_price: 出场价格
            reason: 出场原因
        """
        if sector_industry_id not in self.positions:
            return
        
        position = self.positions[sector_industry_id]
        
        # 记录交易
        self.trade_history.append({
            'date': exit_date,
            'sector_industry_id': sector_industry_id,
            'action': 'sell',
            'price': exit_price,
            'weight': position['weight'],
            'type': position['type'],
            'score': position['current_score'],
            'reason': reason
        })
        
        # 移除持仓
        del self.positions[sector_industry_id]
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        更新权重
        
        Args:
            new_weights: 新的权重字典
        """
        for sector_industry_id, weight in new_weights.items():
            if sector_industry_id in self.positions:
                self.positions[sector_industry_id]['weight'] = weight
    
    def update_scores(self, scores: Dict[str, float]):
        """
        更新评分
        
        Args:
            scores: 评分字典
        """
        for sector_industry_id, score in scores.items():
            if sector_industry_id in self.positions:
                self.positions[sector_industry_id]['current_score'] = score
    
    def get_portfolio(self) -> Dict:
        """
        获取当前组合
        
        Returns:
            Dict: 组合信息
        """
        total_trend_weight = 0.0
        total_left_weight = 0.0
        
        for position in self.positions.values():
            if position['type'] == 'trend':
                total_trend_weight += position['weight']
            else:
                total_left_weight += position['weight']
        
        return {
            'positions': self.positions.copy(),
            'total_trend_weight': total_trend_weight,
            'total_left_weight': total_left_weight,
            'total_weight': total_trend_weight + total_left_weight
        }
    
    def get_current_holdings(self) -> List[str]:
        """
        获取当前持仓列表
        
        Returns:
            List[str]: 持仓行业ID列表
        """
        return list(self.positions.keys())
    
    def get_position(self, sector_industry_id: str) -> Dict:
        """
        获取单个持仓信息
        
        Args:
            sector_industry_id: 行业ID
        
        Returns:
            Dict: 持仓信息
        """
        return self.positions.get(sector_industry_id, {}).copy()
    
    def check_position_limits(self, sector_industry_id: str, 
                             new_weight: float,
                             sector_type: str) -> bool:
        """
        检查仓位限制
        
        Args:
            sector_industry_id: 行业ID
            new_weight: 新权重
            sector_type: 类型
        
        Returns:
            bool: 是否通过检查
        """
        max_single = self.position_config.get('max_single_sector', 0.10)
        max_left_single = self.position_config.get('max_left_single', 0.02)
        max_left_total = self.position_config.get('max_left_total', 0.18)
        max_trend_total = self.position_config.get(
            'max_trend_total',
            self.config.get('backtest', {}).get('allocation', {}).get('trend', 1.0)
        )
        
        # 检查单行业限制
        if sector_type == 'left' and new_weight > max_left_single:
            return False
        
        if sector_type == 'trend' and new_weight > max_single:
            return False
        
        # 检查左侧总仓位
        if sector_type == 'left':
            portfolio = self.get_portfolio()
            current_left = portfolio['total_left_weight']
            
            # 减去当前该行业的权重（如果是更新）
            if sector_industry_id in self.positions:
                current_left -= self.positions[sector_industry_id]['weight']
            
            if current_left + new_weight > max_left_total:
                return False

        if sector_type == 'trend':
            portfolio = self.get_portfolio()
            current_trend = portfolio['total_trend_weight']

            if sector_industry_id in self.positions:
                current_trend -= self.positions[sector_industry_id]['weight']

            if current_trend + new_weight > max_trend_total:
                return False
        
        return True
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        获取交易历史
        
        Returns:
            DataFrame: 交易记录
        """
        return pd.DataFrame(self.trade_history)
