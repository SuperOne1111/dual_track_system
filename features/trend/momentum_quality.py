"""
趋势轨道 - 动量质量类特征计算器

计算动量持续性、成交量确认、资金流确认、CMF确认
"""

import pandas as pd
import numpy as np


class MomentumQualityCalculator:
    """动量质量类特征计算器"""
    
    def __init__(self, volume_window: int = 20):
        """
        初始化计算器
        
        Args:
            volume_window: 成交量计算窗口
        """
        self.volume_window = volume_window
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量质量特征
        
        Args:
            data: 价格数据
        
        Returns:
            DataFrame: 添加了动量质量特征的数据
        """
        df = data.copy()
        
        # 1. 动量持续性
        df['S_persist'] = self._calculate_momentum_persistence(df)
        
        # 2. 成交量确认
        df['S_volConfirm'] = self._calculate_volume_confirmation(df)
        
        # 3. 资金流确认
        df['S_mfv'] = self._calculate_mfv_confirmation(df)
        
        # 4. CMF确认
        df['S_cmf'] = self._calculate_cmf_confirmation(df)
        
        return df
    
    def _calculate_momentum_persistence(self, df: pd.DataFrame) -> pd.Series:
        """
        计算动量持续性得分
        
        a_3 > a_7 > a_14: 20分
        a_3 > a_7 or a_7 > a_14: 10分
        else: 0分
        
        Args:
            df: 包含动量加速度的数据
        
        Returns:
            Series: 动量持续性得分
        """
        score = pd.Series(0.0, index=df.index)
        
        # 强持续性
        strong_persist = (df['a_3'] > df['a_7']) & (df['a_7'] > df['a_14'])
        score.loc[strong_persist] = 20.0
        
        # 弱持续性
        weak_persist = ((df['a_3'] > df['a_7']) | (df['a_7'] > df['a_14'])) & (~strong_persist)
        score.loc[weak_persist] = 10.0
        
        return score
    
    def _calculate_volume_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """
        计算成交量确认得分
        
        volume_ratio = volume_today / avg_volume
        volume_ratio > 1.2: 20分
        volume_ratio > 1.0: 10分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 成交量确认得分
        """
        df_local = df.sort_values(['sector_industry_id', 'trade_date']).copy()
        avg_volume = df_local.groupby('sector_industry_id')['amount'].transform(
            lambda s: s.rolling(window=self.volume_window, min_periods=1).mean()
        )
        volume_ratio = df_local['amount'] / avg_volume.replace(0, np.nan)
        
        score = pd.Series(0.0, index=df.index)
        score.loc[volume_ratio > 1.2] = 20.0
        score.loc[(volume_ratio > 1.0) & (volume_ratio <= 1.2)] = 10.0
        
        return score.reindex(df_local.index).reindex(df.index).fillna(0)
    
    def _calculate_mfv_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """
        计算资金流确认得分
        
        mfv_ratio = MFV_today / MFV_avg
        S_mfv = min(mfv_ratio * 20, 30)
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 资金流确认得分
        """
        df_local = df.sort_values(['sector_industry_id', 'trade_date']).copy()
        avg_mfv = df_local.groupby('sector_industry_id')['daily_mfv'].transform(
            lambda s: s.rolling(window=self.volume_window, min_periods=1).mean()
        )
        
        # 避免除以0
        avg_mfv_safe = avg_mfv.replace(0, np.nan)
        mfv_ratio = df_local['daily_mfv'] / avg_mfv_safe
        
        score = (mfv_ratio * 20).clip(0, 30).fillna(0)
        
        return score.reindex(df_local.index).reindex(df.index).fillna(0)
    
    def _calculate_cmf_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """
        计算CMF确认得分
        
        CMF > 0.1: 20分
        CMF > 0: 10分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: CMF确认得分
        """
        score = pd.Series(0.0, index=df.index)
        score.loc[df['cmf_20'] > 0.1] = 20.0
        score.loc[(df['cmf_20'] > 0) & (df['cmf_20'] <= 0.1)] = 10.0
        
        return score
