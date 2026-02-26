"""
左侧轨道 - 超跌类特征计算器

计算超跌乖离得分、RSI超卖得分
"""

import pandas as pd
import numpy as np


class OversoldCalculator:
    """超跌类特征计算器"""
    
    def __init__(self, rsi_period: int = 14):
        """
        初始化计算器
        
        Args:
            rsi_period: RSI计算周期
        """
        self.rsi_period = rsi_period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算超跌特征
        
        Args:
            data: 价格数据
        
        Returns:
            DataFrame: 添加了超跌特征的数据
        """
        df = data.copy()
        
        # 1. 超跌乖离得分
        df['S_bias'] = self._calculate_bias_score(df)
        
        # 2. RSI超卖得分
        df['S_rsi'] = self._calculate_rsi_score(df)
        
        return df
    
    def _calculate_bias_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算超跌乖离得分
        
        deviation = (P - MA60) / MA60
        deviation < -0.15: 30分
        deviation < -0.10: 25分
        deviation < -0.05: 20分
        deviation < -0.02: 10分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 超跌乖离得分
        """
        deviation = (df['close'] - df['ma_60']) / df['ma_60']
        
        score = pd.Series(0.0, index=df.index)
        score.loc[deviation < -0.15] = 30.0
        score.loc[(deviation < -0.10) & (deviation >= -0.15)] = 25.0
        score.loc[(deviation < -0.05) & (deviation >= -0.10)] = 20.0
        score.loc[(deviation < -0.02) & (deviation >= -0.05)] = 10.0
        
        return score
    
    def _calculate_rsi_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算RSI超卖得分
        
        RSI < 30: 20分
        30 <= RSI < 40: 10分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: RSI超卖得分
        """
        # 按行业计算RSI
        rsi = self._calculate_rsi(df, self.rsi_period)
        
        score = pd.Series(0.0, index=df.index)
        score.loc[rsi < 30] = 20.0
        score.loc[(rsi >= 30) & (rsi < 40)] = 10.0
        
        return score
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算RSI指标
        
        Args:
            df: 价格数据
            period: 计算周期
        
        Returns:
            Series: RSI值
        """
        df_local = df.sort_values(['sector_industry_id', 'trade_date']).copy()
        prices = df_local['close']
        delta = prices.groupby(df_local['sector_industry_id']).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.groupby(df_local['sector_industry_id']).transform(
            lambda s: s.rolling(window=period, min_periods=1).mean()
        )
        avg_loss = loss.groupby(df_local['sector_industry_id']).transform(
            lambda s: s.rolling(window=period, min_periods=1).mean()
        )
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.reindex(df_local.index).reindex(df.index).fillna(50)
