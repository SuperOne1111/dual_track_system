"""
左侧轨道 - 成交量萎缩类特征计算器

计算成交量萎缩得分
"""

import pandas as pd
import numpy as np


class VolumeShrinkCalculator:
    """成交量萎缩类特征计算器"""
    
    def __init__(self, window: int = 20):
        """
        初始化计算器
        
        Args:
            window: 成交量计算窗口
        """
        self.window = window
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量萎缩特征
        
        Args:
            data: 价格数据
        
        Returns:
            DataFrame: 添加了成交量萎缩特征的数据
        """
        df = data.copy()
        
        # 计算成交量萎缩得分
        df['S_volShrink'] = self._calculate_volume_shrink(df)
        
        return df
    
    def _calculate_volume_shrink(self, df: pd.DataFrame) -> pd.Series:
        """
        计算成交量萎缩得分
        
        volume_ratio = volume_today / avg_volume
        volume_ratio < 0.7: 20分（成交量显著萎缩，抛压减弱）
        volume_ratio < 0.9: 10分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 成交量萎缩得分
        """
        df_local = df.sort_values(['sector_industry_id', 'trade_date']).copy()
        avg_volume = df_local.groupby('sector_industry_id')['amount'].transform(
            lambda s: s.rolling(window=self.window, min_periods=1).mean()
        )
        
        # 避免除以0
        volume_ratio = df_local['amount'] / avg_volume.replace(0, np.nan)
        volume_ratio = volume_ratio.fillna(1)
        
        score = pd.Series(0.0, index=df.index)
        score.loc[volume_ratio < 0.7] = 20.0
        score.loc[(volume_ratio >= 0.7) & (volume_ratio < 0.9)] = 10.0
        
        return score.reindex(df_local.index).reindex(df.index).fillna(0)
