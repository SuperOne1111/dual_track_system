"""
趋势轨道 - 相对强度类特征计算器

计算绝对跑赢得分、市场位置得分、综合相对强度
"""

import pandas as pd
import numpy as np
from scipy import stats


class RelativeStrengthCalculator:
    """相对强度类特征计算器"""
    
    def __init__(self):
        pass

    def _to_int_or_none(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    
    def calculate(self, data: pd.DataFrame, hierarchy: dict) -> pd.DataFrame:
        """
        计算相对强度特征
        
        Args:
            data: 价格数据
            hierarchy: 行业层级关系映射
        
        Returns:
            DataFrame: 添加了相对强度特征的数据
        """
        df = data.copy()
        
        # 1. 绝对跑赢得分（相对于父级）
        df['S_out'] = self._calculate_outperformance_score(df, hierarchy)
        
        # 2. 市场位置得分（同层级百分位）
        df['S_rank'] = self._calculate_rank_score(df)
        
        # 3. 综合相对强度
        df['S_rel'] = 0.6 * df['S_out'] + 0.4 * df['S_rank']
        
        return df
    
    def _calculate_outperformance_score(self, df: pd.DataFrame,
                                       hierarchy: dict) -> pd.Series:
        """
        计算绝对跑赢得分
        
        假设每个Level 4行业有一个父级Level 3行业
        计算行业收益率相对于父级的超额收益
        
        Args:
            df: 价格数据
            hierarchy: 行业层级关系
        
        Returns:
            Series: 绝对跑赢得分
        """
        parent_map = hierarchy.get('parent_map', {})
        if not parent_map:
            raise ValueError("缺少 parent_map，无法计算父级相对强度")

        score = pd.Series(0.0, index=df.index)
        for date in df['trade_date'].unique():
            day_data = df[df['trade_date'] == date].copy()
            day_returns = day_data.set_index('sector_industry_id')['r_7'].to_dict()
            parent_returns = day_data['sector_industry_id'].map(
                lambda sid: day_returns.get(parent_map.get(self._to_int_or_none(sid)), np.nan)
            )
            outperformance = day_data['r_7'] - parent_returns
            day_score = pd.Series(0.0, index=day_data.index)
            day_score.loc[outperformance > 0.05] = 30.0
            day_score.loc[(outperformance > 0.02) & (outperformance <= 0.05)] = 20.0
            day_score.loc[(outperformance > 0) & (outperformance <= 0.02)] = 10.0
            score.loc[day_data.index] = day_score
        return score
    
    def _calculate_rank_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算市场位置得分
        
        同层级收益率百分位 -> 映射到0-30分
        
        Args:
            df: 价格数据
        
        Returns:
            Series: 市场位置得分
        """
        score = pd.Series(0.0, index=df.index)
        
        # 按日期、层级分组计算百分位
        for date in df['trade_date'].unique():
            day_data = df[df['trade_date'] == date]
            if day_data.empty:
                continue
            for _, level_group in day_data.groupby('level'):
                if len(level_group) <= 1:
                    continue
                returns = level_group['r_7'].values
                percentiles = stats.rankdata(returns) / len(returns) * 100
                day_score = pd.Series(percentiles / 100 * 30, index=level_group.index)
                score.loc[level_group.index] = day_score

        return score
