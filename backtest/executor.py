"""
交易执行器

模拟买入/卖出执行，扣除交易成本
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


class TradeExecutor:
    """交易执行器"""
    
    def __init__(self, config: dict):
        """
        初始化执行器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.transaction_cost = config.get('backtest', {}).get('transaction_cost', {})
    
    def execute_buy(self, sector_industry_id: str, weight: float,
                   trade_date: datetime, price: float,
                   portfolio_value: float) -> Dict:
        """
        执行买入
        
        Args:
            sector_industry_id: 行业ID
            weight: 目标权重
            trade_date: 交易日期
            price: 交易价格
            portfolio_value: 组合市值
        
        Returns:
            Dict: 交易记录
        """
        buy_cost = self.transaction_cost.get('buy', 0.001)
        
        # 计算买入金额
        target_value = weight * portfolio_value
        
        # 扣除交易成本
        cost = target_value * buy_cost
        actual_value = target_value - cost
        
        # 计算买入数量（简化，使用金额）
        shares = actual_value / price if price > 0 else 0
        
        return {
            'date': trade_date,
            'sector_industry_id': sector_industry_id,
            'action': 'buy',
            'price': price,
            'weight': weight,
            'value': actual_value,
            'cost': cost,
            'shares': shares
        }
    
    def execute_sell(self, sector_industry_id: str, weight: float,
                    trade_date: datetime, price: float,
                    portfolio_value: float) -> Dict:
        """
        执行卖出
        
        Args:
            sector_industry_id: 行业ID
            weight: 当前权重
            trade_date: 交易日期
            price: 交易价格
            portfolio_value: 组合市值
        
        Returns:
            Dict: 交易记录
        """
        sell_cost = self.transaction_cost.get('sell', 0.001)
        
        # 计算卖出金额
        sell_value = weight * portfolio_value
        
        # 扣除交易成本
        cost = sell_value * sell_cost
        actual_value = sell_value - cost
        
        return {
            'date': trade_date,
            'sector_industry_id': sector_industry_id,
            'action': 'sell',
            'price': price,
            'weight': weight,
            'value': actual_value,
            'cost': cost
        }
    
    def execute_rebalance(self, current_positions: Dict,
                         target_weights: Dict,
                         trade_date: datetime,
                         prices: Dict[str, float],
                         portfolio_value: float) -> List[Dict]:
        """
        执行再平衡
        
        Args:
            current_positions: 当前持仓
            target_weights: 目标权重
            trade_date: 交易日期
            prices: 价格字典
            portfolio_value: 组合市值
        
        Returns:
            List[Dict]: 交易记录列表
        """
        trades = []
        
        # 卖出不在目标权重中的持仓
        for sector_industry_id in current_positions:
            if sector_industry_id not in target_weights:
                price = prices.get(sector_industry_id, 0)
                weight = current_positions[sector_industry_id].get('weight', 0)
                
                if weight > 0 and price > 0:
                    trade = self.execute_sell(
                        sector_industry_id, weight, trade_date, price, portfolio_value
                    )
                    trades.append(trade)
        
        # 调整权重
        for sector_industry_id, target_weight in target_weights.items():
            current_weight = current_positions.get(sector_industry_id, {}).get('weight', 0)
            weight_diff = target_weight - current_weight
            
            price = prices.get(sector_industry_id, 0)
            
            if abs(weight_diff) > 0.001 and price > 0:  # 最小调整阈值
                if weight_diff > 0:
                    # 买入
                    trade = self.execute_buy(
                        sector_industry_id, weight_diff, trade_date, price, portfolio_value
                    )
                else:
                    # 卖出
                    trade = self.execute_sell(
                        sector_industry_id, abs(weight_diff), trade_date, price, portfolio_value
                    )
                trades.append(trade)
        
        return trades
    
    def calculate_total_cost(self, trades: List[Dict]) -> float:
        """
        计算总交易成本
        
        Args:
            trades: 交易记录列表
        
        Returns:
            float: 总成本
        """
        return sum(trade.get('cost', 0) for trade in trades)
