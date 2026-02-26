"""
补充机制执行器

卖出后立即补充评分最高的行业
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime


class SupplementEngine:
    """补充机制执行器"""
    
    def __init__(self, config: dict):
        """
        初始化执行器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.supplement_config = config.get('supplement', {})
        self.candidate_config = self.supplement_config.get('candidate_pool', {})
    
    def execute(self, sell_signals: List[Dict],
               portfolio: Dict,
               all_level4: List[str],
               trend_pool: List[str],
               scores: pd.Series,
               portfolio_value: float) -> Dict:
        """
        执行补充机制
        
        Args:
            sell_signals: 卖出信号列表
            portfolio: 当前持仓
            all_level4: 所有Level 4行业
            trend_pool: 趋势池
            scores: 各行业得分
            portfolio_value: 组合市值
        
        Returns:
            Dict: 补充结果
        """
        # Step 1: 计算释放的仓位
        released_weight = self._calculate_released_weight(
            sell_signals, portfolio
        )
        released_cash = released_weight * portfolio_value
        
        # Step 2: 更新候选池
        current_holdings = list(portfolio.get('positions', {}).keys())
        candidate_pool = self._update_candidate_pool(
            all_level4, current_holdings, trend_pool, scores
        )
        
        # Step 3: 选择补充行业
        new_sectors = self._select_supplement_sectors(candidate_pool, scores)
        
        # Step 4: 权重分配
        weights = self._allocate_weights(
            new_sectors, scores, released_cash, portfolio_value
        )
        
        return {
            'sold_sectors': [s['sector_industry_id'] for s in sell_signals],
            'new_sectors': new_sectors,
            'weights': weights,
            'released_cash': released_cash
        }
    
    def _calculate_released_weight(self, sell_signals: List[Dict],
                                   portfolio: Dict) -> float:
        """
        计算释放的仓位权重
        
        Args:
            sell_signals: 卖出信号
            portfolio: 当前持仓
        
        Returns:
            float: 释放的权重
        """
        positions = portfolio.get('positions', {})
        released_weight = 0.0
        
        for signal in sell_signals:
            sector_industry_id = signal['sector_industry_id']
            if sector_industry_id in positions:
                released_weight += positions[sector_industry_id].get('weight', 0)
        
        return released_weight
    
    def _update_candidate_pool(self, all_level4: List,
                              current_holdings: List,
                              trend_pool: List,
                              scores: pd.Series) -> List:
        """
        更新候选池
        
        候选池 = 所有Level 4 - 已持仓 - 趋势池
        过滤：评分 >= 65
        
        Args:
            all_level4: 所有Level 4行业
            current_holdings: 当前持仓
            trend_pool: 趋势池
            scores: 各行业得分
        
        Returns:
            List: 候选池
        """
        score_threshold = self.candidate_config.get('score_threshold', 65)
        
        # 排除已持仓和趋势池（需要处理类型不匹配问题）
        candidates = []
        for s in all_level4:
            # 将当前持仓和趋势池中的ID转为字符串进行比较
            s_str = str(s)
            if s_str not in [str(h) for h in current_holdings] and s_str not in [str(t) for t in trend_pool]:
                candidates.append(s)
        
        # 评分过滤
        filtered_candidates = []
        for sector_industry_id in candidates:
            # 尝试不同的类型匹配
            if sector_industry_id in scores.index:
                if scores[sector_industry_id] >= score_threshold:
                    filtered_candidates.append(sector_industry_id)
            elif str(sector_industry_id) in scores.index:
                if scores[str(sector_industry_id)] >= score_threshold:
                    filtered_candidates.append(sector_industry_id)
        
        return filtered_candidates
    
    def _select_supplement_sectors(self, candidate_pool: List,
                                   scores: pd.Series) -> List:
        """
        选择补充行业
        
        按评分降序，选前6个
        
        Args:
            candidate_pool: 候选池
            scores: 各行业得分
        
        Returns:
            List: 选中的行业
        """
        max_selections = self.candidate_config.get('max_selections', 6)
        
        # 获取候选行业的得分
        candidate_scores = []
        for sector_industry_id in candidate_pool:
            # 尝试不同的类型匹配
            if sector_industry_id in scores.index:
                candidate_scores.append((sector_industry_id, scores[sector_industry_id]))
            elif str(sector_industry_id) in scores.index:
                candidate_scores.append((sector_industry_id, scores[str(sector_industry_id)]))
        
        # 按得分降序排序
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前N个
        selected = [s[0] for s in candidate_scores[:max_selections]]
        
        return selected
    
    def _allocate_weights(self, new_sectors: List,
                         scores: pd.Series,
                         released_cash: float,
                         portfolio_value: float) -> Dict:
        """
        分配权重
        
        按评分加权分配
        
        Args:
            new_sectors: 新选中的行业
            scores: 各行业得分
            released_cash: 释放的现金
            portfolio_value: 组合市值
        
        Returns:
            Dict: 各行业的权重
        """
        if len(new_sectors) == 0 or released_cash <= 0:
            return {}
        
        # 获取新选中行业的得分
        sector_scores = []
        for sector_industry_id in new_sectors:
            # 尝试不同的类型匹配
            if sector_industry_id in scores.index:
                sector_scores.append((sector_industry_id, scores[sector_industry_id]))
            elif str(sector_industry_id) in scores.index:
                sector_scores.append((sector_industry_id, scores[str(sector_industry_id)]))
        
        if len(sector_scores) == 0:
            return {}
        
        # 计算总分
        total_score = sum(s[1] for s in sector_scores)
        
        if total_score <= 0:
            # 等权重分配
            weight_per_sector = released_cash / len(sector_scores) / portfolio_value
            return {str(s[0]): weight_per_sector for s in sector_scores}
        
        # 按评分加权分配
        weights = {}
        for sector_industry_id, score in sector_scores:
            cash_allocation = released_cash * (score / total_score)
            weights[str(sector_industry_id)] = cash_allocation / portfolio_value
        
        return weights
