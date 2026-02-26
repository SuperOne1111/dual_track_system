"""
趋势轨道 - 动量类特征计算器

计算动量加速度特征：a_3, a_7, a_14
"""

import pandas as pd
import numpy as np


class MomentumCalculator:
    """动量类特征计算器"""
    
    def __init__(self):
        pass
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量加速度特征
        
        公式:
        a_3 = (r_3 - r_7) / |r_7|
        a_7 = (r_7 - r_14) / |r_14|
        a_14 = (r_14 - r_30) / |r_30|
        
        Args:
            data: 包含收益率的数据
        
        Returns:
            DataFrame: 添加了动量特征的数据
        """
        df = data.copy()
        
        # 计算动量加速度
        df['a_3'] = self._calculate_acceleration(df['r_3'], df['r_7'])
        df['a_7'] = self._calculate_acceleration(df['r_7'], df['r_14'])
        df['a_14'] = self._calculate_acceleration(df['r_14'], df['r_30'])
        
        return df
    
    def _calculate_acceleration(self, r_short: pd.Series, 
                                r_long: pd.Series) -> pd.Series:
        """
        计算动量加速度
        
        Args:
            r_short: 短期收益率
            r_long: 长期收益率
        
        Returns:
            Series: 动量加速度
        """
        # 避免除以0
        r_long_safe = r_long.replace(0, np.nan)
        acceleration = (r_short - r_long) / r_long_safe.abs()
        
        # 限制极端值
        acceleration = acceleration.clip(-10, 10)
        
        return acceleration.fillna(0)
