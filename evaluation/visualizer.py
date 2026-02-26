"""
可视化器

生成净值曲线、回撤曲线、月度收益热力图等
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict


class Visualizer:
    """可视化器"""
    
    def __init__(self, output_dir: str = 'output'):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_equity_curve(self, equity_curve: pd.Series,
                         benchmark: pd.Series = None,
                         save_path: str = None):
        """
        绘制净值曲线
        
        Args:
            equity_curve: 策略净值曲线
            benchmark: 基准净值曲线
            save_path: 保存路径
        """
        fig = go.Figure()
        
        # 策略曲线
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            name='策略',
            line=dict(color='blue', width=2)
        ))
        
        # 基准曲线
        if benchmark is not None:
            # 对齐并标准化
            aligned_benchmark = benchmark.reindex(equity_curve.index).ffill()
            aligned_benchmark = aligned_benchmark / aligned_benchmark.iloc[0]
            
            fig.add_trace(go.Scatter(
                x=aligned_benchmark.index,
                y=aligned_benchmark.values,
                name='基准',
                line=dict(color='gray', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title='净值曲线',
            xaxis_title='日期',
            yaxis_title='净值',
            legend=dict(x=0, y=1),
            template='plotly_white'
        )
        
        if save_path:
            fig.write_image(save_path)
            fig.write_html(save_path.replace('.png', '.html'))
        
        return fig
    
    def plot_drawdown(self, equity_curve: pd.Series,
                     save_path: str = None):
        """
        绘制回撤曲线
        
        Args:
            equity_curve: 净值曲线
            save_path: 保存路径
        """
        # 计算回撤
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='回撤',
            fill='tozeroy',
            line=dict(color='red', width=1)
        ))
        
        fig.update_layout(
            title='回撤曲线',
            xaxis_title='日期',
            yaxis_title='回撤 (%)',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_image(save_path)
            fig.write_html(save_path.replace('.png', '.html'))
        
        return fig
    
    def plot_monthly_returns_heatmap(self, daily_returns: pd.Series,
                                     save_path: str = None):
        """
        绘制月度收益热力图
        
        Args:
            daily_returns: 日收益率
            save_path: 保存路径
        """
        # 计算月度收益率
        monthly_returns = daily_returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        # 构建透视表
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        pivot_table = monthly_df.pivot(
            index='year',
            columns='month',
            values='return'
        )
        
        # 重命名月份
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                      '7月', '8月', '9月', '10月', '11月', '12月']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
        
        # 绘制热力图
        fig = px.imshow(
            pivot_table,
            labels=dict(x="月份", y="年份", color="收益率 (%)"),
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        
        fig.update_layout(
            title='月度收益率热力图',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def save_all_plots(self, backtest_results: Dict,
                      benchmark: pd.Series = None):
        """
        保存所有图表
        
        Args:
            backtest_results: 回测结果
            benchmark: 基准曲线
        """
        equity_curve = backtest_results['equity_curve']
        daily_returns = backtest_results['daily_returns']
        
        # 净值曲线
        self.plot_equity_curve(
            equity_curve, benchmark,
            os.path.join(self.output_dir, 'equity_curve.png')
        )
        
        # 回撤曲线
        self.plot_drawdown(
            equity_curve,
            os.path.join(self.output_dir, 'drawdown.png')
        )
        
        # 月度收益热力图
        self.plot_monthly_returns_heatmap(
            daily_returns,
            os.path.join(self.output_dir, 'monthly_returns.html')
        )
        
        print(f"图表已保存到 {self.output_dir}")
