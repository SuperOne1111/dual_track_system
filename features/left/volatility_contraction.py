"""
左侧轨道 - 波动收缩类特征计算器

计算波动率收缩得分、波动+影线组合得分
"""

import pandas as pd
import numpy as np


class VolatilityContractionCalculator:
    """波动收缩类特征计算器"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        """
        初始化计算器
        
        Args:
            short_window: 短期波动率窗口
            long_window: 长期波动率窗口
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动收缩特征
        
        Args:
            data: 价格数据
        
        Returns:
            DataFrame: 添加了波动收缩特征的数据
        """
        df = data.copy()
        
        # 1. 波动率收缩得分
        df['S_vol'] = self._calculate_volatility_contraction(df)
        
        # 2. 波动+影线组合得分
        df['S_vol_shadow'] = self._calculate_vol_shadow_combo(df)
        
        return df
    
    def _calculate_volatility_contraction(self, df: pd.DataFrame) -> pd.Series:
        """
        计算波动率收缩得分
        
        vol_ratio = std_5 / std_20
        vol_ratio < 0.7: 20分
        vol_ratio < 0.8: 10分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 波动率收缩得分
        """
        df_local = df.sort_values(['sector_industry_id', 'trade_date']).copy()
        vol_5 = df_local.groupby('sector_industry_id')['close'].transform(
            lambda s: s.rolling(window=self.short_window, min_periods=1).std()
        )
        vol_20 = df_local.groupby('sector_industry_id')['close'].transform(
            lambda s: s.rolling(window=self.long_window, min_periods=1).std()
        )
        
        # 避免除以0
        vol_ratio = vol_5 / vol_20.replace(0, np.nan)
        vol_ratio = vol_ratio.fillna(1)
        
        score = pd.Series(0.0, index=df.index)
        score.loc[vol_ratio < 0.7] = 20.0
        score.loc[(vol_ratio >= 0.7) & (vol_ratio < 0.8)] = 10.0
        
        return score.reindex(df_local.index).reindex(df.index).fillna(0)
    
    def _calculate_vol_shadow_combo(self, df: pd.DataFrame) -> pd.Series:
        """
        计算波动+影线组合得分
        
        vol_ratio < 0.7 and S_shadow > 20: 30分
        vol_ratio < 0.7 or S_shadow > 20: 15分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 波动+影线组合得分
        """
        df_local = df.sort_values(['sector_industry_id', 'trade_date']).copy()
        vol_5 = df_local.groupby('sector_industry_id')['close'].transform(
            lambda s: s.rolling(window=self.short_window, min_periods=1).std()
        )
        vol_20 = df_local.groupby('sector_industry_id')['close'].transform(
            lambda s: s.rolling(window=self.long_window, min_periods=1).std()
        )
        vol_ratio = vol_5 / vol_20.replace(0, np.nan)
        vol_ratio = vol_ratio.fillna(1)

        # 计算下影线强度
        total_range = df_local['high'] - df_local['low']
        shadow_length = df_local['open'] - df_local['low']
        shadow_ratio = shadow_length / total_range.replace(0, np.nan)
        shadow_ratio = shadow_ratio.fillna(0)
        S_shadow = shadow_ratio * 30
        
        # 组合条件
        low_vol = vol_ratio < 0.7
        strong_shadow = S_shadow > 20
        
        score = pd.Series(0.0, index=df.index)
        score.loc[low_vol & strong_shadow] = 30.0
        score.loc[(low_vol | strong_shadow) & ~(low_vol & strong_shadow)] = 15.0
        
        return score.reindex(df_local.index).reindex(df.index).fillna(0)
