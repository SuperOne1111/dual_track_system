"""
左侧轨道特征模块

提供左侧轨道特征计算功能
"""

import pandas as pd

from .slowdown import SlowdownCalculator
from .oversold import OversoldCalculator
from .divergence import DivergenceCalculator
from .volatility_contraction import VolatilityContractionCalculator
from .volume_shrink import VolumeShrinkCalculator
from .score_calculator import LeftScoreCalculator


class LeftFeatureCalculator:
    """左侧特征计算器主类"""
    
    def __init__(self, config: dict):
        """
        初始化左侧特征计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.slowdown_calc = SlowdownCalculator()
        self.oversold_calc = OversoldCalculator()
        self.divergence_calc = DivergenceCalculator()
        self.volatility_calc = VolatilityContractionCalculator()
        self.volume_calc = VolumeShrinkCalculator()
        self.score_calc = LeftScoreCalculator(config)
    
    def calculate_all_features(self, data: pd.DataFrame, 
                               hierarchy: dict = None) -> pd.DataFrame:
        """
        计算所有左侧特征
        
        Args:
            data: 预处理后的数据
            hierarchy: 行业层级关系映射
        
        Returns:
            DataFrame: 包含所有特征的数据
        """
        df = data.copy()
        
        # 1. 下跌减速类特征
        df = self.slowdown_calc.calculate(df)
        
        # 2. 超跌类特征
        df = self.oversold_calc.calculate(df)
        
        # 3. 资金背离类特征
        if hierarchy is not None:
            df = self.divergence_calc.calculate(df, hierarchy)
        
        # 4. 波动收缩类特征
        df = self.volatility_calc.calculate(df)
        
        # 5. 成交量萎缩类特征
        df = self.volume_calc.calculate(df)
        
        return df

    def calculate_snapshot(self, full_data: pd.DataFrame, asof_date: pd.Timestamp,
                           hierarchy: dict = None) -> pd.DataFrame:
        """
        基于历史数据计算特征，并提取asof_date截面
        """
        history = full_data[full_data['trade_date'] <= pd.to_datetime(asof_date)].copy()
        if history.empty:
            return history

        features = self.calculate_all_features(history, hierarchy)
        snapshot = features[features['trade_date'] == pd.to_datetime(asof_date)].copy()
        return snapshot
    
    def calculate_score(self, features: pd.DataFrame) -> pd.Series:
        """
        计算左侧综合得分
        
        Args:
            features: 特征数据
        
        Returns:
            Series: 综合得分
        """
        return self.score_calc.calculate(features)


__all__ = [
    'LeftFeatureCalculator',
    'SlowdownCalculator',
    'OversoldCalculator',
    'DivergenceCalculator',
    'VolatilityContractionCalculator',
    'VolumeShrinkCalculator',
    'LeftScoreCalculator'
]
