"""
趋势轨道特征模块

提供趋势轨道特征计算功能
"""

import pandas as pd
from typing import Optional

from .momentum import MomentumCalculator
from .price_structure import PriceStructureCalculator
from .technical import TechnicalCalculator
from .relative_strength import RelativeStrengthCalculator
from .momentum_quality import MomentumQualityCalculator
from .score_calculator import TrendScoreCalculator


class TrendFeatureCalculator:
    """趋势特征计算器主类"""
    
    def __init__(self, config: dict):
        """
        初始化趋势特征计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.momentum_calc = MomentumCalculator()
        self.price_calc = PriceStructureCalculator()
        self.technical_calc = TechnicalCalculator()
        self.relative_calc = RelativeStrengthCalculator()
        self.quality_calc = MomentumQualityCalculator()
        self.score_calc = TrendScoreCalculator(config)
    
    def calculate_all_features(self, data: pd.DataFrame, 
                               hierarchy: dict = None) -> pd.DataFrame:
        """
        计算所有趋势特征
        
        Args:
            data: 预处理后的数据
            hierarchy: 行业层级关系映射
        
        Returns:
            DataFrame: 包含所有特征的数据
        """
        df = data.copy()
        
        # 1. 动量类特征
        df = self.momentum_calc.calculate(df)
        
        # 2. 价格结构类特征
        df = self.price_calc.calculate(df)
        
        # 3. 技术指标类特征
        df = self.technical_calc.calculate(df)
        
        # 4. 相对强度类特征
        if hierarchy is not None:
            df = self.relative_calc.calculate(df, hierarchy)
        
        # 5. 动量质量类特征
        df = self.quality_calc.calculate(df)
        
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
    
    def calculate_score(self, features: pd.DataFrame, level: Optional[int] = None) -> pd.Series:
        """
        计算趋势综合得分
        
        Args:
            features: 特征数据
            level: 行业层级；为空时按features中每条记录的level字段自动分层打分
        
        Returns:
            Series: 综合得分
        """
        return self.score_calc.calculate(features, level)


__all__ = [
    'TrendFeatureCalculator',
    'MomentumCalculator',
    'PriceStructureCalculator',
    'TechnicalCalculator',
    'RelativeStrengthCalculator',
    'MomentumQualityCalculator',
    'TrendScoreCalculator'
]
