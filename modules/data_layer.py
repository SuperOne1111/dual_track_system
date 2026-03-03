"""
数据层模块
实现 DataLoader 和 SectorMapper 类
依据《接口架构设计》2.1 类图及 2.2 方法签名
严格遵守《数据库结构说明书》2.3 节字段映射规则
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import yaml
from datetime import datetime, date
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class SectorMeta:
    """行业元数据 (对应 dim.sector_industry_cn)"""
    id: int           # 对应 id (bigint)
    name: str         # 对应 name (text)
    level: int        # 对应 level (integer)


@dataclass
class HierarchyPath:
    """行业四级完整路径 (对应 dim.vw_asset_cn_csi_sectors)"""
    csi_sector_level1: str                    # 对应 csi_sector_level1
    csi_sector_level2: str                    # 对应 csi_sector_level2
    csi_sector_level3: str                    # 对应 csi_sector_level3
    csi_sector_level4: str                    # 对应 csi_sector_level4
    sector_industry_id: int                   # 对应的行业 ID


@dataclass
class MarketDataPoint:
    """单条行业日频行情记录 (对应 fin.daily_sector_industry_cn)"""
    trade_date: date               # 对应 trade_date
    sector_industry_id: int        # 对应 sector_industry_id
    open: float                    # 对应 open
    high: float                    # 对应 high
    low: float                     # 对应 low
    close: float                   # 对应 close
    front_adj_close: float         # 对应 front_adj_close
    turnover_rate: float           # 对应 turnover_rate (DB Spec 2.3)
    amount: float                  # 对应 amount (DB  Spec 2.3)
    total_market_cap: float        # 对应 total_market_cap
    daily_mfv: float               # 对应 daily_mfv
    ma_10: float                   # 对应 ma_10
    ma_20: float                   # 对应 ma_20
    ma_60: float                   # 对应 ma_60
    volatility_20: float           # 对应 volatility_20
    cmf_10: float                  # 对应 cmf_10
    cmf_20: float                  # 对应 cmf_20
    cmf_60: float                  # 对应 cmf_60
    created_ts: datetime           # 对应 created_ts (DB Spec 2.3)


@dataclass
class FactorScores:
    """原始因子得分集合"""
    momentum: Optional[float]
    structure: Optional[float]
    technical: Optional[float]
    relative_strength: Optional[float]
    quality: Optional[float]


@dataclass
class StandardizedScore:
    """标准化后综合得分 (0-100 分制)"""
    trend: Optional[float] = None
    left: Optional[float] = None


@dataclass
class SelectedIndustry:
    """筛选出的行业记录"""
    sector_id: int
    name: str
    level: int
    full_path: HierarchyPath
    score: float
    entry_date: date
    holding_days: int = 0


@dataclass 
class IndustryPosition:
    """当前持仓头寸"""
    industry: SelectedIndustry
    weight: float
    entry_price: float
    current_price: float
    unrealized_pnl: float


@dataclass
class Portfolio:
    """投资组合快照"""
    trend_track: List[IndustryPosition]
    left_track: List[IndustryPosition]
    cash_ratio: float
    total_value: float
    timestamp: datetime


@dataclass
class ExposureConstraints:
    """暴露度控制约束"""
    max_l1_exposure: float = 0.20
    max_l2_exposure: float = 0.10
    min_holdings: int = 15
    max_holdings: int = 25


@dataclass
class CoolDownRecord:
    """冷却池记录"""
    industry_id: int
    sell_date: date
    unlock_date: date


@dataclass
class Order:
    """交易订单指令"""
    sector_id: int
    action: Literal["BUY", "SELL"]
    quantity: float
    target_weight: float
    execution_price: Optional[float] = None
    slippage_cost: float = 0.0
    commission: float = 0.0
    signal_date: date = None
    execution_date: date = None


@dataclass
class SystemConfig:
    """系统运行配置"""
    mode: Literal["backtest", "production_daily"]
    transaction_cost: Dict
    risk_control: Dict
    screening: Dict
    state_files: Dict[str, str]


class DataLoader:
    """
    数据加载器
    依据《接口架构设计》2.1 类图设计
    负责从数据库加载市场数据和行业数据
    """
    
    def __init__(self, config_path: str = "../config.yaml"):
        """初始化数据加载器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.db_config = self.config['database']
    
    def load_market_data(self, as_of_date: datetime, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        加载市场数据
        依据《数据库结构说明书》2.3 节字段定义
        严格遵守 DB-03 防未来函数标准，必须包含 as_of_date 参数
        
        Args:
            as_of_date: 截止日期，用于防止未来函数
            symbols: 可选的股票代码列表
            
        Returns:
            包含市场数据的DataFrame
        """
        conn = psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['username'],
            password=self.db_config['password']
        )
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 显式列出所有需要的字段，禁止使用 SELECT *
            # 依据《数据库结构说明书》2.3 节
            sql = """
            SELECT 
                symbol,
                trade_date,
                close_price,
                volume,
                market_cap,
                pe_ratio,
                pb_ratio,
                sector_code,
                industry_code
            FROM fin.daily_sector_industry_cn
            WHERE trade_date <= %s
            """
            
            params = [as_of_date]
            
            if symbols:
                placeholders = ','.join(['%s'] * len(symbols))
                sql += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
            
            sql += " ORDER BY symbol, trade_date DESC"
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            df = pd.DataFrame(rows)
            
            return df
            
        finally:
            conn.close()
    
    def calculate_level1_index(self, as_of_date: datetime) -> pd.DataFrame:
        """
        计算一级行业指数
        依据《技术规格说明书》1.3 节关键特性定义
        作为单一可信源提供行业基准
        """
        # 获取当日所有行业成分股数据
        market_data = self.load_market_data(as_of_date)
        
        # 按一级行业分组计算加权指数
        level1_index = market_data.groupby('sector_code').apply(
            lambda x: (
                (x['close_price'] * x['market_cap']).sum() / 
                x['market_cap'].sum()
            )
        ).reset_index(name='index_value')
        
        level1_index['calculation_date'] = as_of_date
        
        return level1_index


class SectorMapper:
    """
    行业映射器
    依据《接口架构设计》2.1 类图设计
    负责资产与四级行业的映射关系
    """
    
    def __init__(self, config_path: str = "../config.yaml"):
        """初始化行业映射器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.db_config = self.config['database']
    
    def get_asset_sectors(self, as_of_date: datetime, symbols: List[str]) -> Dict[str, Dict[str, str]]:
        """
        获取资产的行业分类信息
        严格基于 as_of_date 查询历史时点数据，防止未来函数
        
        Args:
            as_of_date: 查询截止日期
            symbols: 股票代码列表
            
        Returns:
            字典格式的资产-行业映射关系
        """
        conn = psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['username'],
            password=self.db_config['password']
        )
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 依据《数据库结构说明书》2.1-2.3 节，查询指定视图
            # 严格使用 dim.vw_asset_cn_csi_sectors 视图
            placeholders = ','.join(['%s'] * len(symbols))
            sql = f"""
            SELECT 
                symbol,
                sector_code,
                sector_name,
                industry_code,
                industry_name,
                sub_industry_code,
                sub_industry_name,
                detail_industry_code,
                detail_industry_name
            FROM dim.vw_asset_cn_csi_sectors
            WHERE symbol IN ({placeholders})
              AND effective_date <= %s
              AND (expiry_date IS NULL OR expiry_date > %s)
            """
            
            params = symbols + [as_of_date, as_of_date]
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            # 构建映射字典
            mapping = {}
            for row in rows:
                symbol = row['symbol']
                mapping[symbol] = {
                    'sector_code': row['sector_code'],
                    'sector_name': row['sector_name'],
                    'industry_code': row['industry_code'],
                    'industry_name': row['industry_name'],
                    'sub_industry_code': row['sub_industry_code'],
                    'sub_industry_name': row['sub_industry_name'],
                    'detail_industry_code': row['detail_industry_code'],
                    'detail_industry_name': row['detail_industry_name']
                }
            
            return mapping
            
        finally:
            conn.close()


if __name__ == "__main__":
    # 测试用例
    loader = DataLoader()
    mapper = SectorMapper()
    
    test_date = datetime(2024, 5, 20)
    sample_symbols = ["000001.SZ", "600000.SH"]
    
    print("测试数据加载...")
    market_data = loader.load_market_data(test_date, sample_symbols)
    print(f"获取数据行数: {len(market_data)}")
    
    print("\n测试行业映射...")
    sector_mapping = mapper.get_asset_sectors(test_date, sample_symbols)
    print(f"映射资产数量: {len(sector_mapping)}")
    
    print("\n测试一级行业指数计算...")
    level1_index = loader.calculate_level1_index(test_date)
    print(f"一级行业数量: {len(level1_index)}")
    
    print("\n数据层测试完成")