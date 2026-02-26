"""
报告生成器

生成Excel格式的绩效报告
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = 'output'):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.unallocated_reason_map = {
            'zero_target': '目标权重为零',
            'invalid_cap': '单名上限无效',
            'no_candidates': '无可用候选',
            'fully_allocated': '已完全分配',
            'cap_binding_or_insufficient_candidates': '单名上限约束/候选不足'
        }
    
    def generate_excel_report(self, metrics: Dict,
                             transactions: pd.DataFrame,
                             backtest_results: Dict,
                             output_path: str = None):
        """
        生成Excel报告
        
        Args:
            metrics: 绩效指标
            transactions: 交易记录
            backtest_results: 回测结果
            output_path: 输出路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                self.output_dir, f'performance_report_{timestamp}.xlsx'
            )
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. 概览表
            self._write_overview(metrics, writer)
            
            # 2. 通用指标
            self._write_general_metrics(metrics, writer)
            
            # 3. 左侧专项指标
            if 'left' in metrics:
                self._write_left_metrics(metrics, writer)
            
            # 4. 相对基准指标
            if 'relative' in metrics:
                self._write_relative_metrics(metrics, writer)
            
            # 5. 交易记录
            self._write_transactions(transactions, writer)
            
            # 6. 净值曲线
            self._write_equity_curve(backtest_results, writer)

            # 7. 现金占比
            self._write_cash_weights(backtest_results, writer)

            # 8. 未分配原因统计
            self._write_unallocated_stats(backtest_results, writer)
        
        print(f"报告已保存到 {output_path}")
        return output_path
    
    def _write_overview(self, metrics: Dict, writer):
        """写入概览"""
        general = metrics.get('general', {})
        
        overview = pd.DataFrame({
            '指标': ['年化收益率', '年化波动率', '夏普比率', '最大回撤', '胜率'],
            '数值': [
                f"{general.get('annualized_return', 0):.2%}",
                f"{general.get('annualized_volatility', 0):.2%}",
                f"{general.get('sharpe_ratio', 0):.2f}",
                f"{general.get('max_drawdown', 0):.2%}",
                f"{general.get('win_rate', 0):.2%}"
            ]
        })
        
        overview.to_excel(writer, sheet_name='概览', index=False)
    
    def _write_general_metrics(self, metrics: Dict, writer):
        """写入通用指标"""
        general = metrics.get('general', {})
        
        df = pd.DataFrame({
            '指标': list(general.keys()),
            '数值': [f"{v:.4f}" if isinstance(v, float) else str(v) 
                    for v in general.values()]
        })
        
        df.to_excel(writer, sheet_name='通用指标', index=False)
    
    def _write_left_metrics(self, metrics: Dict, writer):
        """写入左侧专项指标"""
        left = metrics.get('left', {})
        
        if not left:
            return
        
        df = pd.DataFrame({
            '指标': list(left.keys()),
            '数值': [f"{v:.4f}" if isinstance(v, float) else str(v) 
                    for v in left.values()]
        })
        
        df.to_excel(writer, sheet_name='左侧专项指标', index=False)
    
    def _write_relative_metrics(self, metrics: Dict, writer):
        """写入相对基准指标"""
        relative = metrics.get('relative', {})
        
        if not relative:
            return
        
        df = pd.DataFrame({
            '指标': list(relative.keys()),
            '数值': [f"{v:.4f}" if isinstance(v, float) else str(v) 
                    for v in relative.values()]
        })
        
        df.to_excel(writer, sheet_name='相对基准指标', index=False)
    
    def _write_transactions(self, transactions: pd.DataFrame, writer):
        """写入交易记录"""
        if len(transactions) == 0:
            return
        
        transactions.to_excel(writer, sheet_name='交易记录', index=False)
    
    def _write_equity_curve(self, backtest_results: Dict, writer):
        """写入净值曲线"""
        equity_curve = backtest_results.get('equity_curve')
        
        if equity_curve is None:
            return
        
        df = pd.DataFrame({
            '日期': equity_curve.index,
            '净值': equity_curve.values
        })
        
        df.to_excel(writer, sheet_name='净值曲线', index=False)

    def _write_cash_weights(self, backtest_results: Dict, writer):
        """写入每日现金占比"""
        cash_df = backtest_results.get('cash_weights')
        if cash_df is None or len(cash_df) == 0:
            return

        df = cash_df.copy()
        if 'date' in df.columns:
            df = df.rename(columns={'date': '日期'})
        if 'cash_weight' in df.columns:
            df = df.rename(columns={'cash_weight': '现金占比'})
        df.to_excel(writer, sheet_name='现金占比', index=False)

    def _write_unallocated_stats(self, backtest_results: Dict, writer):
        """写入未分配原因统计"""
        unalloc_df = backtest_results.get('unallocated')
        if unalloc_df is None or len(unalloc_df) == 0:
            return

        detail = unalloc_df.copy()
        if 'reason' in detail.columns:
            detail['reason'] = detail['reason'].map(
                lambda x: self.unallocated_reason_map.get(x, x)
            )
        detail = detail.rename(columns={
            'trade_date': '调仓日',
            'track': '轨道',
            'target_weight': '目标权重',
            'allocated_weight': '已分配权重',
            'remaining_weight': '未分配权重',
            'candidate_count': '候选数',
            'per_name_cap': '单名上限',
            'reason': '原因'
        })
        detail.to_excel(writer, sheet_name='未分配明细', index=False)

        summary = unalloc_df.groupby(['track', 'reason'], as_index=False).agg(
            unallocated_weight=('remaining_weight', 'sum'),
            rebalance_count=('remaining_weight', 'count')
        )
        summary['reason'] = summary['reason'].map(
            lambda x: self.unallocated_reason_map.get(x, x)
        )
        summary = summary.rename(columns={
            'track': '轨道',
            'reason': '原因',
            'unallocated_weight': '未分配权重合计',
            'rebalance_count': '出现次数'
        })
        summary.to_excel(writer, sheet_name='未分配原因统计', index=False)
