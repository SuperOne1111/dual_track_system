"""
趋势轨道 - 技术指标类特征计算器

计算价格偏离度得分、波动率调整收益率、均线排列强度
"""

import pandas as pd
import numpy as np


class TechnicalCalculator:
    """技术指标类特征计算器"""
    
    def __init__(self):
        pass
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标特征
        
        Args:
            data: 价格数据
        
        Returns:
            DataFrame: 添加了技术指标特征的数据
        """
        df = data.copy()
        
        # 1. 价格偏离度得分
        df['S_dev'] = self._calculate_deviation_score(df)
        
        # 2. 波动率调整收益率
        df['r_tilde'] = self._calculate_risk_adjusted_return(df)
        
        # 3. 均线排列强度
        df['S_ma'] = self._calculate_ma_strength(df)
        
        return df
    
    def _calculate_deviation_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算价格偏离度得分
        
        deviation = (P - MA60) / MA60
        deviation > 0.1: 30分
        deviation > 0.05: 20分
        deviation > 0: 10分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 价格偏离度得分
        """
        deviation = (df['close'] - df['ma_60']) / df['ma_60']
        
        score = pd.Series(0.0, index=df.index)
        score.loc[deviation > 0.1] = 30.0
        score.loc[(deviation > 0.05) & (deviation <= 0.1)] = 20.0
        score.loc[(deviation > 0) & (deviation <= 0.05)] = 10.0
        
        return score
    
    def _calculate_risk_adjusted_return(self, df: pd.DataFrame) -> pd.Series:
        """
        计算波动率调整收益率
        
        r_tilde = r_7 / volatility_20
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 波动率调整收益率
        """
        # 避免除以0
        vol_safe = df['volatility_20'].replace(0, np.nan)
        
        risk_adjusted = df['r_7'] / vol_safe
        
        # 限制极端值
        return risk_adjusted.clip(-10, 10).fillna(0)
    
    def _calculate_ma_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        计算均线排列强度
        
        如果 MA10 > MA20 > MA60:
            ratio_10_20 = (MA10 - MA20) / MA20
            ratio_20_60 = (MA20 - MA60) / MA60
            S_ma = 10 + ratio_10_20 * 100 + ratio_20_60 * 100
        else:
            S_ma = 0
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 均线排列强度
        """
        ma_10 = df['ma_10']
        ma_20 = df['ma_20']
        ma_60 = df['ma_60']
        
        # 多头排列条件
        bullish_arrangement = (ma_10 > ma_20) & (ma_20 > ma_60)
        
        # 计算比例
        ratio_10_20 = (ma_10 - ma_20) / ma_20
        ratio_20_60 = (ma_20 - ma_60) / ma_60
        
        # 计算得分
        score = 10 + ratio_10_20 * 100 + ratio_20_60 * 100
        
        # 非多头排列得0分
        score = score.where(bullish_arrangement, 0)
        
        # 限制范围
        return score.clip(0, 50).fillna(0)
