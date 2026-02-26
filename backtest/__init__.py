"""
回测引擎模块

提供回测执行、基准构建和交易执行功能
"""

from .engine import BacktestEngine
from .benchmark import BenchmarkBuilder
from .executor import TradeExecutor
from .regime import MarketRegimeManager

__all__ = ['BacktestEngine', 'BenchmarkBuilder', 'TradeExecutor', 'MarketRegimeManager']
