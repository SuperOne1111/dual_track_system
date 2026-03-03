"""买入信号生成器测试专用的简化数据类

为了测试能够顺利运行，定义一个与真实MarketDataPoint兼容的简化版本。
"""

from datetime import date, datetime
from typing import Optional


class MarketDataPoint:
    """市场数据点类 - 测试版"""
    
    def __init__(
        self,
        trade_date: date,
        sector_industry_id: int,
        open: float,
        high: float,
        low: float,
        close: float,
        front_adj_close: float,
        turnover_rate: float,
        amount: float,
        total_market_cap: float,
        daily_mfv: float,
        ma_10: float,
        ma_20: float,
        ma_60: float,
        volatility_20: float,
        cmf_10: float,
        cmf_20: float,
        cmf_60: float,
        created_ts: Optional[datetime]
    ):
        self.trade_date = trade_date
        self.sector_industry_id = sector_industry_id
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.front_adj_close = front_adj_close
        self.turnover_rate = turnover_rate
        self.amount = amount
        self.total_market_cap = total_market_cap
        self.daily_mfv = daily_mfv
        self.ma_10 = ma_10
        self.ma_20 = ma_20
        self.ma_60 = ma_60
        self.volatility_20 = volatility_20
        self.cmf_10 = cmf_10
        self.cmf_20 = cmf_20
        self.cmf_60 = cmf_60
        self.created_ts = created_ts


class StateRepository:
    """状态仓库接口 - 测试版"""
    
    def load(self, key: str):
        """加载状态"""
        pass
    
    def save(self, key: str, data):
        """保存状态"""
        pass