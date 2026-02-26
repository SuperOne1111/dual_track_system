"""
绩效指标计算器

计算通用指标和左侧专项指标
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class MetricsCalculator:
    """绩效指标计算器"""
    
    def __init__(self):
        pass
    
    def calculate_all_metrics(self, backtest_results: Dict,
                             benchmark_returns: pd.Series = None) -> Dict:
        """
        计算所有绩效指标
        
        Args:
            backtest_results: 回测结果
            benchmark_returns: 基准收益率
        
        Returns:
            Dict: 所有绩效指标
        """
        equity_curve = backtest_results['equity_curve']
        daily_returns = backtest_results['daily_returns']
        transactions = backtest_results.get('transactions', pd.DataFrame())
        
        metrics = {}
        
        # 通用指标
        metrics['general'] = self._calculate_general_metrics(
            equity_curve, daily_returns
        )
        
        # 左侧专项指标
        if len(transactions) > 0:
            left_transactions = transactions[transactions['type'] == 'left']
            metrics['left'] = self._calculate_left_metrics(left_transactions)
        
        # 相对基准指标
        if benchmark_returns is not None:
            metrics['relative'] = self._calculate_relative_metrics(
                daily_returns, benchmark_returns
            )
        
        return metrics
    
    def _calculate_general_metrics(self, equity_curve: pd.Series,
                                   daily_returns: pd.Series) -> Dict:
        """
        计算通用绩效指标
        
        Args:
            equity_curve: 净值曲线
            daily_returns: 日收益率
        
        Returns:
            Dict: 通用指标
        """
        metrics = {}
        
        # 年化收益率
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        n_days = len(daily_returns)
        n_years = n_days / 252
        metrics['annualized_return'] = (1 + total_return) ** (1 / n_years) - 1
        
        # 年化波动率
        metrics['annualized_volatility'] = daily_returns.std() * np.sqrt(252)
        
        # 夏普比率
        risk_free_rate = 0.03
        excess_return = metrics['annualized_return'] - risk_free_rate
        metrics['sharpe_ratio'] = excess_return / metrics['annualized_volatility'] \
            if metrics['annualized_volatility'] > 0 else 0
        
        # 最大回撤
        max_dd, dd_duration = self._calculate_max_drawdown(equity_curve)
        metrics['max_drawdown'] = max_dd
        metrics['drawdown_duration'] = dd_duration
        
        # 胜率（日收益率>0的比例）
        metrics['win_rate'] = (daily_returns > 0).sum() / len(daily_returns)
        
        # 盈亏比
        avg_gain = daily_returns[daily_returns > 0].mean()
        avg_loss = abs(daily_returns[daily_returns < 0].mean())
        metrics['profit_loss_ratio'] = avg_gain / avg_loss if avg_loss > 0 else 0
        
        # 换手率（简化计算）
        metrics['turnover'] = daily_returns.abs().mean() * 252
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """
        计算最大回撤
        
        Args:
            equity_curve: 净值曲线
        
        Returns:
            Tuple[float, int]: (最大回撤, 回撤持续天数)
        """
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        
        max_drawdown = drawdown.min()
        
        # 计算回撤持续天数
        max_dd_idx = drawdown.idxmin()
        peak_before_dd = equity_curve[:max_dd_idx].idxmax()
        drawdown_duration = (max_dd_idx - peak_before_dd).days
        
        return max_drawdown, drawdown_duration
    
    def _calculate_left_metrics(self, left_transactions: pd.DataFrame) -> Dict:
        """
        计算左侧专项指标
        
        Args:
            left_transactions: 左侧交易记录
        
        Returns:
            Dict: 左侧专项指标
        """
        metrics = {}
        
        if len(left_transactions) == 0:
            return metrics
        
        # 分离买入和卖出
        buys = left_transactions[left_transactions['action'] == 'buy']
        sells = left_transactions[left_transactions['action'] == 'sell']
        
        # 止跌成功率（卖出价格 >= 买入价格的比例）
        if len(sells) > 0:
            successful_stops = 0
            for _, sell in sells.iterrows():
                buy_record = buys[buys['sector_industry_id'] == sell['sector_industry_id']]
                if len(buy_record) > 0:
                    buy_price = buy_record['price'].iloc[0]
                    if sell['price'] >= buy_price:
                        successful_stops += 1
            
            metrics['stop_success_rate'] = successful_stops / len(sells)
        
        # 最大不利变动（MAE）
        # 简化：计算平均回撤
        if len(sells) > 0:
            mae_list = []
            for _, sell in sells.iterrows():
                buy_record = buys[buys['sector_industry_id'] == sell['sector_industry_id']]
                if len(buy_record) > 0:
                    buy_price = buy_record['price'].iloc[0]
                    mae = (sell['price'] - buy_price) / buy_price
                    mae_list.append(mae)
            
            if mae_list:
                metrics['median_mae'] = np.median(mae_list)
                metrics['avg_bounce'] = np.mean([m for m in mae_list if m > 0])
        
        # 假阳性率（止损卖出的比例）
        if len(sells) > 0:
            stop_loss_sells = sells[sells['reason'].isin(['atr_stop', 'fixed_stop'])]
            metrics['false_positive_rate'] = len(stop_loss_sells) / len(sells)
        
        return metrics
    
    def _calculate_relative_metrics(self, strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Dict:
        """
        计算相对基准指标
        
        Args:
            strategy_returns: 策略日收益率
            benchmark_returns: 基准日收益率
        
        Returns:
            Dict: 相对指标
        """
        metrics = {}
        
        # 对齐数据
        aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        aligned_data.columns = ['strategy', 'benchmark']
        
        if len(aligned_data) == 0:
            return metrics
        
        # Beta
        cov = aligned_data['strategy'].cov(aligned_data['benchmark'])
        var = aligned_data['benchmark'].var()
        metrics['beta'] = cov / var if var > 0 else 0
        
        # Alpha
        strategy_annual = aligned_data['strategy'].mean() * 252
        benchmark_annual = aligned_data['benchmark'].mean() * 252
        metrics['alpha'] = strategy_annual - metrics['beta'] * benchmark_annual
        
        # 信息比率
        tracking_error = (aligned_data['strategy'] - aligned_data['benchmark']).std() * np.sqrt(252)
        metrics['information_ratio'] = metrics['alpha'] / tracking_error if tracking_error > 0 else 0
        
        return metrics
