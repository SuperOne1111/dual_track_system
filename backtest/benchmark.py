"""
基准构建器

构建Level 1行业市值加权指数作为基准
"""

import pandas as pd
import numpy as np
from typing import Dict


class BenchmarkBuilder:
    """基准构建器"""
    
    def __init__(self):
        self.benchmark = None
    
    def build_benchmark(self, level1_data: pd.DataFrame) -> pd.Series:
        """
        构建基准指数
        
        使用所有Level 1行业市值加权
        
        Args:
            level1_data: Level 1行业数据，包含 trade_date, sector_industry_id, close, total_market_cap
        
        Returns:
            Series: 基准指数序列
        """
        df = level1_data.copy()
        
        # 按日期分组计算市值加权收益率
        benchmark_returns = []
        
        for date in df['trade_date'].unique():
            day_data = df[df['trade_date'] == date]
            
            # 计算市值加权收益率
            total_cap = day_data['total_market_cap'].sum()
            
            if total_cap > 0:
                # 计算加权平均收益率
                weights = day_data['total_market_cap'] / total_cap
                daily_return = (day_data['r_1'] * weights).sum()
            else:
                daily_return = 0
            
            benchmark_returns.append({
                'trade_date': date,
                'daily_return': daily_return
            })
        
        benchmark_df = pd.DataFrame(benchmark_returns)
        benchmark_df = benchmark_df.sort_values('trade_date')
        
        # 计算累计净值
        benchmark_df['cumulative_return'] = (1 + benchmark_df['daily_return']).cumprod()
        benchmark_df['index_value'] = 100 * benchmark_df['cumulative_return']
        
        # 保存结果
        self.benchmark = benchmark_df.set_index('trade_date')['index_value']
        
        return self.benchmark
    
    def get_benchmark_returns(self) -> pd.Series:
        """
        获取基准收益率
        
        Returns:
            Series: 基准日收益率
        """
        if self.benchmark is None:
            raise ValueError("基准尚未构建，请先调用 build_benchmark()")
        
        returns = self.benchmark.pct_change().fillna(0)
        return returns
    
    def get_benchmark_metrics(self) -> Dict:
        """
        获取基准指标
        
        Returns:
            Dict: 基准指标
        """
        if self.benchmark is None:
            raise ValueError("基准尚未构建")
        
        returns = self.get_benchmark_returns()
        
        # 计算年化收益率
        total_return = self.benchmark.iloc[-1] / self.benchmark.iloc[0] - 1
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        
        # 计算年化波动率
        annualized_vol = returns.std() * np.sqrt(252)
        
        # 计算最大回撤
        cummax = self.benchmark.cummax()
        drawdown = (self.benchmark - cummax) / cummax
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'max_drawdown': max_drawdown
        }
