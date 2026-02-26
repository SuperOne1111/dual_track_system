"""
左侧轨道 - 独立筛选器

执行左侧独立筛选，从候选池中选择超跌反转行业
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class LeftSelector:
    """左侧轨道筛选器"""
    
    def __init__(self, config: dict):
        """
        初始化筛选器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.left_config = config.get('left', {})
        self.selection_config = self.left_config.get('selection', {})
    
    def select(self, scores: pd.DataFrame, trend_pool: List,
               returns: pd.Series, trade_date) -> Dict:
        """
        执行左侧独立筛选
        
        Args:
            scores: 左侧综合得分数据
            trend_pool: 趋势池（需要排除）
            returns: 30日收益率数据
            trade_date: 交易日期
        
        Returns:
            Dict: 筛选结果
        """
        # 确定ID列名
        id_col = 'sector_industry_id' if 'sector_industry_id' in scores.columns else 'sector_industry_id'
        
        # Step 1: 构建候选池
        candidate_pool = self._build_candidate_pool(
            scores, trend_pool, returns, id_col
        )
        
        # Step 2: 计算反转综合得分
        candidate_scores = scores[scores[id_col].isin(candidate_pool)]
        
        # Step 3: 验证入选条件
        valid_candidates = self._validate_candidates(
            candidate_scores, returns, id_col
        )
        
        # Step 4: 排序与选择
        left_pool = self._select_left_pool(valid_candidates, id_col)
        
        return {
            'trade_date': trade_date,
            'left_pool': left_pool,
            'scores': dict(zip(
                valid_candidates[id_col].tolist(),
                valid_candidates['S_recovery'].tolist()
            )) if len(valid_candidates) > 0 else {},
            'candidate_pool': candidate_pool
        }
    
    def _build_candidate_pool(self, scores: pd.DataFrame,
                             trend_pool: List,
                             returns: pd.Series,
                             id_col: str = 'sector_industry_id') -> List:
        """
        构建候选池
        
        候选池 = 所有Level 4 - 趋势池
        过滤条件：30日跌幅 > 10%
        
        Args:
            scores: 得分数据
            trend_pool: 趋势池
            returns: 30日收益率
            id_col: ID列名
        
        Returns:
            List: 候选池行业ID
        """
        return_threshold = self.selection_config.get('return_threshold', -0.10)
        
        # 获取所有Level 4行业
        all_level4 = scores[id_col].unique().tolist()
        
        # 排除趋势池
        candidates = [s for s in all_level4 if s not in trend_pool]
        
        # 过滤：30日跌幅 > 阈值
        filtered_candidates = []
        for sector_industry_id in candidates:
            if sector_industry_id in returns.index:
                if returns[sector_industry_id] < return_threshold:
                    filtered_candidates.append(sector_industry_id)
        
        print(f"左侧候选池: {len(filtered_candidates)} 个行业（30日收益<={return_threshold:.0%}）")
        return filtered_candidates
    
    def _validate_candidates(self, candidate_scores: pd.DataFrame,
                            returns: pd.Series,
                            id_col: str = 'sector_industry_id') -> pd.DataFrame:
        """
        验证入选条件
        
        必须同时满足：
        - 中期下跌：r_30 < -10%
        - 阈值：S_recovery >= 65
        
        Args:
            candidate_scores: 候选行业得分
            returns: 30日收益率
            id_col: ID列名
        
        Returns:
            DataFrame: 有效候选
        """
        score_threshold = self.selection_config.get('score_threshold', 65)
        return_threshold = self.selection_config.get('return_threshold', -0.10)
        
        valid = candidate_scores.copy()
        
        # 评分阈值过滤
        valid = valid[valid['S_recovery'] >= score_threshold]
        
        print(f"有效候选: {len(valid)} 个行业（得分>={score_threshold}）")
        return valid
    
    def _select_left_pool(self, valid_candidates: pd.DataFrame,
                         id_col: str = 'sector_industry_id') -> List:
        """
        选择左侧池
        
        按S_recovery降序，选前3个
        
        Args:
            valid_candidates: 有效候选
            id_col: ID列名
        
        Returns:
            List: 选中的左侧行业ID
        """
        max_selections = self.selection_config.get('max_selections', 3)
        
        # 按得分降序排序
        sorted_candidates = valid_candidates.sort_values(
            'S_recovery', ascending=False
        )
        
        # 选择前N个
        selected = sorted_candidates.head(max_selections)[id_col].tolist()
        
        print(f"左侧池: 选中 {len(selected)} 个行业")
        return selected
