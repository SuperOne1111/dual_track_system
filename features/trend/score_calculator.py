"""
趋势轨道 - 综合得分计算器

按层级加权计算趋势综合得分
"""

import pandas as pd
import numpy as np
from typing import Optional


class TrendScoreCalculator:
    """趋势综合得分计算器"""
    
    def __init__(self, config: dict):
        """
        初始化得分计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.weights = config.get('trend', {}).get('weights', {})
    
    def calculate(self, features: pd.DataFrame, level: Optional[int] = None) -> pd.Series:
        """
        计算趋势综合得分
        
        S_trend = w1*动量 + w2*偏离度 + w3*波动率调整 + w4*相对强度 + w5*动量质量
        
        Args:
            features: 特征数据
            level: 行业层级 (1/2/3/4)
        
        Returns:
            Series: 综合得分
        """
        # 标准化各特征到0-100范围
        momentum_score = self._normalize_momentum(features)
        deviation_score = self._normalize_deviation(features)
        vol_adjust_score = self._normalize_vol_adjust(features)
        relative_score = self._normalize_relative(features)
        quality_score = self._normalize_quality(features)

        # 兼容旧逻辑：显式传入level时仍使用该层权重
        if level is not None:
            weights = self._get_level_weights(level)
            return self._weighted_sum(
                weights, momentum_score, deviation_score, vol_adjust_score, relative_score, quality_score
            )

        # 默认按每条记录自身level使用对应权重
        if 'level' not in features.columns:
            weights = self._get_level_weights(4)
            return self._weighted_sum(
                weights, momentum_score, deviation_score, vol_adjust_score, relative_score, quality_score
            )

        levels = pd.to_numeric(features['level'], errors='coerce').fillna(4).astype(int)
        default_weights = self._get_level_weights(4)
        score = self._weighted_sum(
            default_weights, momentum_score, deviation_score, vol_adjust_score, relative_score, quality_score
        )

        for lv in sorted(levels.unique()):
            if lv == 4:
                continue
            mask = levels == lv
            if not mask.any():
                continue
            lv_weights = self._get_level_weights(lv)
            lv_score = self._weighted_sum(
                lv_weights, momentum_score, deviation_score, vol_adjust_score, relative_score, quality_score
            )
            score.loc[mask] = lv_score.loc[mask]

        return score

    def _get_level_weights(self, level: int) -> dict:
        try:
            lv = int(level)
        except (TypeError, ValueError):
            lv = 4
        return self.weights.get(f'level{lv}', self.weights.get('level4', {}))

    @staticmethod
    def _weighted_sum(weights: dict,
                      momentum_score: pd.Series,
                      deviation_score: pd.Series,
                      vol_adjust_score: pd.Series,
                      relative_score: pd.Series,
                      quality_score: pd.Series) -> pd.Series:
        return (
            weights.get('momentum', 0.25) * momentum_score +
            weights.get('deviation', 0.20) * deviation_score +
            weights.get('vol_adjust', 0.10) * vol_adjust_score +
            weights.get('relative', 0.30) * relative_score +
            weights.get('quality', 0.15) * quality_score
        )
    
    def _normalize_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        标准化动量得分
        
        Args:
            df: 特征数据
        
        Returns:
            Series: 标准化后的动量得分
        """
        # 综合动量加速度得分（处理可能缺失的列）
        a_3 = df.get('a_3', pd.Series(0, index=df.index))
        a_7 = df.get('a_7', pd.Series(0, index=df.index))
        a_14 = df.get('a_14', pd.Series(0, index=df.index))
        momentum = (a_3.fillna(0) + a_7.fillna(0) + a_14.fillna(0)) / 3
        
        # 使用sigmoid函数映射到0-100（处理极端值更好）
        score = 100 / (1 + np.exp(-momentum))
        
        return score.clip(0, 100)
    
    def _normalize_deviation(self, df: pd.DataFrame) -> pd.Series:
        """
        标准化偏离度得分
        
        Args:
            df: 特征数据
        
        Returns:
            Series: 标准化后的偏离度得分
        """
        # S_dev 原始范围是 0-30，直接映射到 0-100
        s_dev = df.get('S_dev', pd.Series(0, index=df.index)).fillna(0)
        return (s_dev / 30 * 100).clip(0, 100)
    
    def _normalize_relative(self, df: pd.DataFrame) -> pd.Series:
        """
        标准化相对强度得分
        
        Args:
            df: 特征数据
        
        Returns:
            Series: 标准化后的相对强度得分
        """
        # S_rel 原始范围是 0-30，直接映射到 0-100
        s_rel = df.get('S_rel', pd.Series(0, index=df.index)).fillna(0)
        return (s_rel / 30 * 100).clip(0, 100)
    
    def _normalize_vol_adjust(self, df: pd.DataFrame) -> pd.Series:
        """
        标准化波动率调整得分
        
        Args:
            df: 特征数据
        
        Returns:
            Series: 标准化后的波动率调整得分
        """
        r_tilde = df.get('r_tilde', pd.Series(0, index=df.index)).fillna(0)
        
        # 使用sigmoid函数映射到0-100
        score = 100 / (1 + np.exp(-r_tilde))
        
        return score.clip(0, 100)
    
    def _normalize_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        标准化动量质量得分
        
        Args:
            df: 特征数据
        
        Returns:
            Series: 标准化后的动量质量得分
        """
        # 综合动量质量得分（每个特征最大20-30分，总和最大约80分）
        quality = (
            df.get('S_persist', pd.Series(0, index=df.index)).fillna(0) +
            df.get('S_volConfirm', pd.Series(0, index=df.index)).fillna(0) +
            df.get('S_mfv', pd.Series(0, index=df.index)).fillna(0) +
            df.get('S_cmf', pd.Series(0, index=df.index)).fillna(0)
        )
        
        # 标准化到0-100范围（假设最大总分80分）
        return (quality / 80 * 100).clip(0, 100)
