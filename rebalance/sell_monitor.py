"""
卖出信号监控器

监控持仓行业是否触发卖出信号
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta


class SellMonitor:
    """卖出信号监控器"""
    
    def __init__(self, config: dict):
        """
        初始化监控器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.risk_config = config.get('risk_control', {})
        self.stop_loss_config = self.risk_config.get('stop_loss', {})
    
    def get_sell_signals(self, portfolio: Dict,
                        current_data: pd.DataFrame,
                        current_date: datetime,
                        current_scores: Dict[str, float] = None) -> List[Dict]:
        """
        获取卖出信号
        
        Args:
            portfolio: 当前持仓
            current_data: 当前市场数据
            current_date: 当前日期
        
        Returns:
            List[Dict]: 卖出信号列表
        """
        sell_signals = []
        
        positions = portfolio.get('positions', {})
        
        # 确定ID列名
        id_col = 'sector_industry_id' if 'sector_industry_id' in current_data.columns else 'sector_industry_id'
        
        for sector_industry_id, position in positions.items():
            # 确保类型匹配
            try:
                sector_industry_id_match = int(sector_industry_id) if id_col == 'sector_industry_id' else sector_industry_id
            except (ValueError, TypeError):
                sector_industry_id_match = sector_industry_id
            
            # 获取当前价格
            sector_data = current_data[current_data[id_col] == sector_industry_id_match]
            
            if len(sector_data) == 0:
                continue
            
            current_price = sector_data['close'].iloc[0]
            atr_20 = sector_data.get('ATR_20', sector_data.get('volatility_20', pd.Series([0]))).iloc[0]
            if current_scores is not None:
                score_key = str(sector_industry_id)
                current_score = current_scores.get(score_key, position.get('current_score', 0))
            else:
                current_score = position.get('current_score', 0)
            
            # 检查各种卖出条件
            signal = self._check_sell_conditions(
                sector_industry_id, position, current_price,
                atr_20, current_score, current_date
            )
            
            if signal:
                sell_signals.append(signal)
        
        return sell_signals
    
    def _check_sell_conditions(self, sector_industry_id: str,
                               position: Dict,
                               current_price: float,
                               atr_20: float,
                               current_score: float,
                               current_date: datetime) -> Dict:
        """
        检查卖出条件
        
        任一条件触发即卖出：
        1. ATR动态止损
        2. 固定止损
        3. 持有期到期
        4. 评分跌破阈值
        
        Args:
            sector_industry_id: 行业ID
            position: 持仓信息
            current_price: 当前价格
            atr_20: 20日ATR
            current_score: 当前评分
            current_date: 当前日期
        
        Returns:
            Dict or None: 卖出信号或None
        """
        entry_price = position.get('entry_price', current_price)
        entry_date = position.get('entry_date', current_date)
        entry_score = position.get('entry_score', current_score)
        
        # 计算回撤
        drawdown = (current_price - entry_price) / entry_price
        
        # 计算持有天数
        holding_days = (current_date - entry_date).days

        # 建仓当日不触发卖出，避免同日买卖冲突
        if holding_days <= 0:
            return None
        
        # 1. ATR动态止损
        atr_multiplier = self.stop_loss_config.get('atr_multiplier', 1.5)
        atr_stop_level = -atr_multiplier * (atr_20 / entry_price) if entry_price > 0 else -np.inf
        
        if drawdown < atr_stop_level:
            return {
                'sector_industry_id': sector_industry_id,
                'signal_type': 'atr_stop',
                'current_price': current_price,
                'entry_price': entry_price,
                'holding_days': holding_days,
                'current_score': current_score,
                'drawdown': drawdown
            }
        
        # 2. 固定止损
        fixed_stop = self.stop_loss_config.get('fixed_stop', 0.08)
        
        if drawdown < -fixed_stop:
            return {
                'sector_industry_id': sector_industry_id,
                'signal_type': 'fixed_stop',
                'current_price': current_price,
                'entry_price': entry_price,
                'holding_days': holding_days,
                'current_score': current_score,
                'drawdown': drawdown
            }
        
        # 3. 持有期到期
        holding_period = self.stop_loss_config.get('holding_period', 60)
        
        if holding_days >= holding_period:
            return {
                'sector_industry_id': sector_industry_id,
                'signal_type': 'holding_period',
                'current_price': current_price,
                'entry_price': entry_price,
                'holding_days': holding_days,
                'current_score': current_score,
                'drawdown': drawdown
            }
        
        # 4. 评分跌破阈值
        score_exit = self.stop_loss_config.get('score_exit', 50)
        
        if current_score < score_exit:
            return {
                'sector_industry_id': sector_industry_id,
                'signal_type': 'score_drop',
                'current_price': current_price,
                'entry_price': entry_price,
                'holding_days': holding_days,
                'current_score': current_score,
                'entry_score': entry_score,
                'drawdown': drawdown
            }
        
        return None
