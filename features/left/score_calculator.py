"""
左侧轨道 - 综合得分计算器

计算左侧轨道综合得分
"""

import pandas as pd
import numpy as np


class LeftScoreCalculator:
    """左侧综合得分计算器"""
    
    def __init__(self, config: dict):
        """
        初始化得分计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.weights = config.get('left', {}).get('score_weights', {
            'slowdown': 0.20,
            'bias': 0.25,
            'shadow': 0.30,
            'flow': 0.15,
            'vol_shadow': 0.10
        })
    
    def calculate(self, features: pd.DataFrame) -> pd.Series:
        """
        计算左侧综合得分
        
        S_recovery = 0.20*S_slow + 0.25*S_bias + 0.30*S_shadow + 0.15*S_flow + 0.10*S_vol_shadow
        
        各特征原始范围：0-30分
        加权后最大得分：0.20*30 + 0.25*30 + 0.30*30 + 0.15*30 + 0.10*30 = 30分
        标准化到0-100范围
        
        Args:
            features: 特征数据
        
        Returns:
            Series: 综合得分（0-100范围）
        """
        # 获取各特征得分（处理可能缺失的列）
        S_slow = features.get('S_slow', pd.Series(0, index=features.index)).fillna(0)
        S_bias = features.get('S_bias', pd.Series(0, index=features.index)).fillna(0)
        S_shadow = features.get('S_shadow', pd.Series(0, index=features.index)).fillna(0)
        S_flow = features.get('S_flow', pd.Series(0, index=features.index)).fillna(0)
        S_vol_shadow = features.get('S_vol_shadow', pd.Series(0, index=features.index)).fillna(0)
        
        # 加权求和
        raw_score = (
            self.weights.get('slowdown', 0.20) * S_slow +
            self.weights.get('bias', 0.25) * S_bias +
            self.weights.get('shadow', 0.30) * S_shadow +
            self.weights.get('flow', 0.15) * S_flow +
            self.weights.get('vol_shadow', 0.10) * S_vol_shadow
        )
        
        # 标准化到0-100范围（最大原始得分约30分）
        score = (raw_score / 30 * 100).clip(0, 100)
        
        return score
    
    def get_feature_contributions(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        获取各特征对总得分的贡献
        
        Args:
            features: 特征数据
        
        Returns:
            DataFrame: 各特征的贡献
        """
        contributions = pd.DataFrame(index=features.index)
        
        contributions['slowdown'] = features['S_slow'].fillna(0) * self.weights.get('slowdown', 0.20)
        contributions['bias'] = features['S_bias'].fillna(0) * self.weights.get('bias', 0.25)
        contributions['shadow'] = features['S_shadow'].fillna(0) * self.weights.get('shadow', 0.30)
        contributions['flow'] = features['S_flow'].fillna(0) * self.weights.get('flow', 0.15)
        contributions['vol_shadow'] = features['S_vol_shadow'].fillna(0) * self.weights.get('vol_shadow', 0.10)
        
        return contributions
