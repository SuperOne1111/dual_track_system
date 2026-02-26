"""
左侧轨道 - 下跌减速类特征计算器

计算下跌减速得分、下影线强度、收盘位置改善
"""

import pandas as pd
import numpy as np


class SlowdownCalculator:
    """下跌减速类特征计算器"""
    
    def __init__(self):
        pass
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算下跌减速特征
        
        Args:
            data: 价格数据
        
        Returns:
            DataFrame: 添加了下跌减速特征的数据
        """
        df = data.copy()
        
        # 1. 下跌减速得分（二阶导数）
        df['S_slow'] = self._calculate_slowdown_score(df)
        
        # 2. 下影线强度
        df['S_shadow'] = self._calculate_shadow_strength(df)
        
        # 3. 收盘位置改善
        df['delta_C_pos'] = self._calculate_close_position_improvement(df)
        
        return df
    
    def _calculate_slowdown_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算下跌减速得分
        
        alpha_3 = r_3 - r_7
        alpha_7 = r_7 - r_14
        alpha_3 > 0 and alpha_7 > 0: 30分
        alpha_3 > 0 or alpha_7 > 0: 20分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 下跌减速得分
        """
        alpha_3 = df['r_3'] - df['r_7']
        alpha_7 = df['r_7'] - df['r_14']
        
        score = pd.Series(0.0, index=df.index)
        
        # 强减速信号
        strong_slowdown = (alpha_3 > 0) & (alpha_7 > 0)
        score.loc[strong_slowdown] = 30.0
        
        # 弱减速信号
        weak_slowdown = ((alpha_3 > 0) | (alpha_7 > 0)) & (~strong_slowdown)
        score.loc[weak_slowdown] = 20.0
        
        return score
    
    def _calculate_shadow_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        计算下影线强度
        
        shadow_ratio = (open - low) / (high - low)
        S_shadow = shadow_ratio * 30
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 下影线强度得分
        """
        total_range = df['high'] - df['low']
        shadow_length = df['open'] - df['low']
        
        # 避免除以0
        shadow_ratio = shadow_length / total_range.replace(0, np.nan)
        shadow_ratio = shadow_ratio.fillna(0)
        
        # 映射到0-30分
        score = shadow_ratio * 30
        
        return score.clip(0, 30)
    
    def _calculate_close_position_improvement(self, df: pd.DataFrame) -> pd.Series:
        """
        计算收盘位置改善
        
        delta_C_pos = 近3日C_pos均值 - 前3日C_pos均值
        delta_C_pos > 0.1: 20分
        delta_C_pos > 0.05: 10分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 收盘位置改善得分
        """
        df_local = df.sort_values(['sector_industry_id', 'trade_date']).copy()
        c_pos = (df_local['close'] - df_local['low']) / (df_local['high'] - df_local['low']).replace(0, np.nan)
        c_pos = c_pos.fillna(0.5)

        recent_c_pos = c_pos.groupby(df_local['sector_industry_id']).transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )
        previous_c_pos = c_pos.groupby(df_local['sector_industry_id']).transform(
            lambda s: s.shift(3).rolling(window=3, min_periods=1).mean()
        )
        
        # 改善幅度
        delta_c_pos = recent_c_pos - previous_c_pos
        
        score = pd.Series(0.0, index=df.index)
        score.loc[delta_c_pos > 0.1] = 20.0
        score.loc[(delta_c_pos > 0.05) & (delta_c_pos <= 0.1)] = 10.0
        
        return score.reindex(df_local.index).reindex(df.index).fillna(0)
