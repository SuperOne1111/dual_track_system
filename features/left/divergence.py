"""
左侧轨道 - 资金背离类特征计算器

计算资金背离得分、相对抗跌得分
"""

import pandas as pd
import numpy as np


class DivergenceCalculator:
    """资金背离类特征计算器"""
    
    def __init__(self, window: int = 20):
        """
        初始化计算器
        
        Args:
            window: 计算窗口
        """
        self.window = window

    def _to_int_or_none(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    
    def calculate(self, data: pd.DataFrame, hierarchy: dict = None) -> pd.DataFrame:
        """
        计算资金背离特征
        
        Args:
            data: 价格数据
            hierarchy: 行业层级关系映射
        
        Returns:
            DataFrame: 添加了资金背离特征的数据
        """
        df = data.copy()
        
        # 1. 资金背离得分
        df['S_flow'] = self._calculate_flow_divergence(df)

        # 2. 相对抗跌得分
        df['S_rel'] = self._calculate_relative_strength(df, hierarchy)
        
        return df
    
    def _calculate_flow_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        计算资金背离得分
        
        is_price_low = (close == min(close[-20:]))
        is_mfv_high = (mfv_today > avg_mfv)
        is_price_low and is_mfv_high: 30分
        is_price_low or is_mfv_high: 15分
        else: 0分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 资金背离得分
        """
        df_local = df.sort_values(['sector_industry_id', 'trade_date']).copy()
        low_20 = df_local.groupby('sector_industry_id')['close'].transform(
            lambda s: s.rolling(window=self.window, min_periods=1).min()
        )
        is_price_low = (df_local['close'] <= low_20 * 1.01)  # 允许1%误差

        # 计算MFV均值
        avg_mfv = df_local.groupby('sector_industry_id')['daily_mfv'].transform(
            lambda s: s.rolling(window=self.window, min_periods=1).mean()
        )
        is_mfv_high = df_local['daily_mfv'] > avg_mfv
        
        score = pd.Series(0.0, index=df.index)
        score.loc[is_price_low & is_mfv_high] = 30.0
        score.loc[(is_price_low | is_mfv_high) & ~(is_price_low & is_mfv_high)] = 15.0
        
        return score.reindex(df_local.index).reindex(df.index).fillna(0)
    
    def _calculate_relative_strength(self, df: pd.DataFrame, hierarchy: dict = None) -> pd.Series:
        """
        计算相对抗跌得分
        
        当父级下跌超过3%时：
        - 行业收益率 > 父级收益率: 20分
        - 行业收益率 > 父级收益率 + 2%: 30分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 相对抗跌得分
        """
        if hierarchy is None or 'parent_map' not in hierarchy:
            raise ValueError("缺少 parent_map，无法计算左侧相对抗跌得分")
        parent_map = hierarchy['parent_map']

        score = pd.Series(0.0, index=df.index)

        # 按日期分组计算
        for date in df['trade_date'].unique():
            day_data = df[df['trade_date'] == date].copy()
            day_returns = day_data.set_index('sector_industry_id')['r_7'].to_dict()
            parent_returns = day_data['sector_industry_id'].map(
                lambda sid: day_returns.get(parent_map.get(self._to_int_or_none(sid)), np.nan)
            )
            parent_fall = parent_returns < -0.03
            rel = day_data['r_7'] - parent_returns
            day_score = pd.Series(0.0, index=day_data.index)
            day_score.loc[parent_fall & (rel > 0)] = 20.0
            day_score.loc[parent_fall & (rel > 0.02)] = 30.0
            score.loc[day_data.index] = day_score

        return score
