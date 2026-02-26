"""
趋势轨道 - 漏斗式筛选器

执行层级漏斗筛选：Level 2 → Level 3 → Level 4
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class TrendSelector:
    """趋势轨道筛选器"""
    
    def __init__(self, config: dict):
        """
        初始化筛选器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.selection_config = config.get('selection', {}).get('trend', {})
    
    def select(self, scores: pd.DataFrame, hierarchy: dict,
               trade_date: pd.Timestamp) -> Dict:
        """
        执行漏斗式筛选
        
        Args:
            scores: 综合得分数据，包含 sector_industry_id, level, S_trend
            hierarchy: 行业层级关系映射
            trade_date: 交易日期
        
        Returns:
            Dict: 筛选结果
        """
        # Step 1: Level 2筛选
        level2_selection = self._select_level2(scores)
        
        if not isinstance(hierarchy, dict) or 'children_by_parent' not in hierarchy:
            raise ValueError("TrendSelector 需要 hierarchy 映射，且包含 children_by_parent")

        # Step 2: Level 3筛选（按父级分组）
        level3_selection = self._select_level3(scores, level2_selection, hierarchy)

        # Step 3: Level 4筛选（按父级分组）
        level4_selection = self._select_level4(scores, level3_selection, hierarchy)

        return {
            'trade_date': trade_date,
            'level2': level2_selection,
            'level3': level3_selection,
            'level4': level4_selection,
            'trend_pool': level4_selection,
            'audit': {
                'l2_count': len(level2_selection),
                'l3_count': len(level3_selection),
                'l4_count': len(level4_selection)
            }
        }
    
    def _select_level2(self, scores: pd.DataFrame) -> List:
        """
        Level 2筛选
        
        对所有Level 2行业按得分降序排序，选前8个
        
        Args:
            scores: 得分数据
        
        Returns:
            List: 选中的Level 2行业ID
        """
        level2_config = self.selection_config.get('level2', {})
        count = level2_config.get('count', 8)
        
        # 筛选Level 2行业
        level2_scores = scores[scores['level'] == 2].copy()
        
        # 按得分降序排序
        level2_scores = level2_scores.sort_values('S_trend', ascending=False)
        
        # 选择前N个 - 使用sector_industry_id列
        id_col = 'sector_industry_id' if 'sector_industry_id' in level2_scores.columns else 'sector_industry_id'
        selected = level2_scores.head(count)[id_col].tolist()
        
        print(f"Level 2筛选: 从 {len(level2_scores)} 个行业中选中 {len(selected)} 个")
        return selected
    
    def _select_level3(self, scores: pd.DataFrame, 
                       selected_l2: List,
                       hierarchy: dict) -> List:
        """
        Level 3筛选（分组筛选）
        
        对每个选中的Level 2父级，在其子Level 3中选前2个
        
        Args:
            scores: 得分数据
            selected_l2: 选中的Level 2行业ID
            hierarchy: 层级关系
        
        Returns:
            List: 选中的Level 3行业ID
        """
        level3_config = self.selection_config.get('level3', {})
        per_parent = level3_config.get('per_parent', 2)
        max_total = level3_config.get('max_total', 16)

        children_map = hierarchy.get('children_by_parent', {})
        id_col = 'sector_industry_id'
        level3_scores = scores[scores['level'] == 3].copy()
        selected = []

        for parent_id in selected_l2:
            children = set(children_map.get(int(parent_id), []))
            if not children:
                continue
            child_scores = level3_scores[level3_scores[id_col].astype(int).isin(children)]
            child_scores = child_scores.sort_values('S_trend', ascending=False)
            selected.extend(child_scores.head(per_parent)[id_col].tolist())

        if selected:
            selected_df = level3_scores[level3_scores[id_col].isin(selected)].copy()
            selected_df = selected_df.sort_values('S_trend', ascending=False).drop_duplicates(subset=[id_col])
            selected = selected_df.head(max_total)[id_col].tolist()

        print(
            f"Level 3筛选: 基于 {len(selected_l2)} 个L2父级，每父级选 {per_parent} 个，"
            f"总上限 {max_total}，实际选中 {len(selected)} 个"
        )
        return selected
    
    def _select_level4(self, scores: pd.DataFrame,
                       selected_l3: List,
                       hierarchy: dict) -> List:
        """
        Level 4筛选（分组筛选）
        
        对每个选中的Level 3父级，在其子Level 4中选前1个
        
        Args:
            scores: 得分数据
            selected_l3: 选中的Level 3行业ID
            hierarchy: 层级关系
        
        Returns:
            List: 选中的Level 4行业ID
        """
        level4_config = self.selection_config.get('level4', {})
        per_parent = level4_config.get('per_parent', 1)
        max_total = level4_config.get('max_total', 16)

        children_map = hierarchy.get('children_by_parent', {})
        id_col = 'sector_industry_id'
        level4_scores = scores[scores['level'] == 4].copy()
        selected = []

        for parent_id in selected_l3:
            children = set(children_map.get(int(parent_id), []))
            if not children:
                continue
            child_scores = level4_scores[level4_scores[id_col].astype(int).isin(children)]
            child_scores = child_scores.sort_values('S_trend', ascending=False)
            selected.extend(child_scores.head(per_parent)[id_col].tolist())

        if selected:
            selected_df = level4_scores[level4_scores[id_col].isin(selected)].copy()
            selected_df = selected_df.sort_values('S_trend', ascending=False).drop_duplicates(subset=[id_col])
            selected = selected_df.head(max_total)[id_col].tolist()

        print(
            f"Level 4筛选: 基于 {len(selected_l3)} 个L3父级，每父级选 {per_parent} 个，"
            f"总上限 {max_total}，实际选中 {len(selected)} 个"
        )
        return selected
