"""
趋势轨道 - 价格结构类特征计算器

计算价格效率、突破质量、收盘位置比
"""

import pandas as pd
import numpy as np


class PriceStructureCalculator:
    """价格结构类特征计算器"""
    
    def __init__(self, window: int = 5):
        """
        初始化计算器
        
        Args:
            window: 价格效率计算窗口
        """
        self.window = window
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格结构特征
        
        Args:
            data: 价格数据
        
        Returns:
            DataFrame: 添加了价格结构特征的数据
        """
        df = data.copy()
        
        # 1. 价格效率（按行业5日平均）
        df['E_eff'] = self._calculate_price_efficiency(df)

        # 2. 突破质量（按行业）
        df['B_high'] = self._calculate_breakout_quality(df)
        
        # 3. 收盘位置比
        df['C_pos'] = self._calculate_close_position(df)
        
        return df
    
    def _calculate_price_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """
        计算价格效率
        
        E_eff = mean(|(close - open) / (high - low)|)
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 价格效率
        """
        df_local = df.sort_values(['sector_industry_id', 'trade_date']).copy()
        daily_efficiency = (
            (df_local['close'] - df_local['open']).abs() /
            (df_local['high'] - df_local['low']).replace(0, np.nan)
        ).fillna(0).clip(0, 1)
        efficiency = daily_efficiency.groupby(df_local['sector_industry_id']).transform(
            lambda s: s.rolling(window=self.window, min_periods=1).mean()
        )
        return efficiency.reindex(df_local.index).reindex(df.index)
    
    def _calculate_breakout_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        计算突破质量
        
        B_high = 1 if (close > max(high_t-1, high_t-2, high_t-3) and close > open) else 0
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 突破质量（布尔值）
        """
        df_local = df.sort_values(['sector_industry_id', 'trade_date']).copy()
        g = df_local.groupby('sector_industry_id')['high']
        high_t1 = g.shift(1)
        high_t2 = g.shift(2)
        high_t3 = g.shift(3)
        recent_high = pd.concat([high_t1, high_t2, high_t3], axis=1).max(axis=1)
        breakout = (df_local['close'] > recent_high) & (df_local['close'] > df_local['open'])
        series = breakout.astype(int)
        return series.reindex(df_local.index).reindex(df.index)
    
    def _calculate_close_position(self, df: pd.DataFrame) -> pd.Series:
        """
        计算收盘位置比
        
        C_pos = (close - low) / (high - low)
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 收盘位置比
        """
        total_range = df['high'] - df['low']
        close_position = (df['close'] - df['low']) / total_range
        
        # 处理除0情况
        close_position = close_position.fillna(0.5)
        
        # 限制范围
        return close_position.clip(0, 1)
