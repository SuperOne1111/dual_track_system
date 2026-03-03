"""买入信号生成器模块

依据《dual-track-api-design》2.1 节类图设计，实现买入信号生成器抽象基类及具体实现。
遵循 STRAT-01 双轨隔离要求，趋势跟踪与均值回归策略实现物理隔离。
遵循 DB-03 防未来函数原则，所有方法包含 as_of_date 参数。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import date
import pandas as pd

try:
    from modules.data_layer import MarketDataPoint
    from modules.state_manager import StateRepository
except ImportError:
    # 用于测试的简化版本
    from tests.test_data_classes import MarketDataPoint, StateRepository


class BuySignalGenerator(ABC):
    """买入信号生成器抽象基类
    
    依据《dual-track-api-design》2.1 节，定义买入信号生成的标准接口。
    通过多态注入实现 CONS-02 逻辑一致性标准。
    """
    
    def __init__(self, state_repo: StateRepository):
        """初始化买入信号生成器
        
        Args:
            state_repo: 状态仓库，用于读取必要配置和历史状态
        """
        self.state_repo = state_repo
    
    @abstractmethod
    def generate_buy_signals(
        self, 
        market_data: List[MarketDataPoint], 
        as_of_date: date,
        exclude_list: Optional[List[int]] = None
    ) -> List[Dict]:
        """生成买入信号
        
        Args:
            market_data: 市场数据点列表
            as_of_date: 截止日期，用于防未来函数
            exclude_list: 排除的行业ID列表
            
        Returns:
            买入信号列表，每个元素包含行业ID、得分等信息
        """
        pass


class TrendBuySignalGenerator(BuySignalGenerator):
    """趋势跟踪策略买入信号生成器
    
    实现趋势跟踪轨道的买入信号生成逻辑。
    遵循 STRAT-01 双轨隔离要求，与均值回归策略物理隔离。
    """
    
    def generate_buy_signals(
        self, 
        market_data: List[MarketDataPoint], 
        as_of_date: date,
        exclude_list: Optional[List[int]] = None
    ) -> List[Dict]:
        """生成趋势跟踪策略买入信号
        
        依据《dual-track-tech-spec》第3.3节趋势轨道特征，计算趋势得分，
        筛选出得分最高的行业作为买入候选。
        
        Args:
            market_data: 市场数据点列表
            as_of_date: 截止日期，用于防未来函数
            exclude_list: 排除的行业ID列表
            
        Returns:
            趋势跟踪策略买入信号列表
        """
        if exclude_list is None:
            exclude_list = []
            
        # 将市场数据转换为DataFrame便于处理
        df = pd.DataFrame([{
            'sector_industry_id': md.sector_industry_id,
            'trade_date': md.trade_date,
            'open': md.open,
            'high': md.high,
            'low': md.low,
            'close': md.close,
            'front_adj_close': md.front_adj_close,
            'total_market_cap': md.total_market_cap,
            'volatility_20': md.volatility_20,
            'ma_10': md.ma_10,
            'ma_20': md.ma_20,
            'ma_60': md.ma_60
        } for md in market_data])
        
        # 只处理指定日期的数据
        df = df[df['trade_date'] == as_of_date]
        
        # 排除指定行业
        df = df[~df['sector_industry_id'].isin(exclude_list)]
        
        # 计算趋势指标得分
        df = self._calculate_trend_scores(df)
        
        # 按得分排序，返回买入信号
        signals = []
        df_sorted = df.sort_values(by='trend_score', ascending=False)
        
        for _, row in df_sorted.iterrows():
            signals.append({
                'sector_id': int(row['sector_industry_id']),
                'score': float(row['trend_score']),
                'strategy': 'trend',
                'date': as_of_date,
                'reason': f"Trend score: {row['trend_score']:.2f}"
            })
        
        return signals
    
    def _calculate_trend_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势得分
        
        依据《dual-track-tech-spec》第3.3节趋势轨道特征，计算各类趋势因子得分。
        
        Args:
            df: 市场数据DataFrame
            
        Returns:
            添加了趋势得分的DataFrame
        """
        # 动量类因子 - 使用现有的移动平均线
        df['momentum_score'] = 0.30 * ((df['close'] / df['ma_20']) - 1).fillna(0)
        
        # 价格结构因子 - 避免除零错误
        df['price_structure_score'] = 0.15 * (
            (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-8)
        ).fillna(0)
        
        # 技术指标因子
        df['technical_score'] = 0.10 * (
            (df['close'] - df['ma_60']) / df['ma_60']
        ).fillna(0)
        
        # 相对强度因子
        # 这里简化处理，使用市值作为相对强度的一个代理指标
        df['relative_strength_score'] = 0.35 * (
            df['total_market_cap'] / df['total_market_cap'].mean()
        ).fillna(1)
        
        # 动量质量因子
        df['quality_score'] = 0.10 * (
            df['volatility_20'] / df['volatility_20'].mean()
        ).fillna(1)
        
        # 综合趋势得分 (0-100分制)
        # 先对各个因子进行标准化到0-100分制，然后加权求和
        factors = ['momentum_score', 'price_structure_score', 'technical_score', 
                  'relative_strength_score', 'quality_score']
        
        for factor in factors:
            # 标准化到0-100分制
            min_val = df[factor].min()
            max_val = df[factor].max()
            if max_val != min_val:
                df[f"{factor}_norm"] = 100 * (df[factor] - min_val) / (max_val - min_val)
            else:
                df[f"{factor}_norm"] = 50  # 如果所有值相同，则给中间分数
        
        # 计算综合得分
        df['trend_score'] = (
            df['momentum_score_norm'] * 0.30 +
            df['price_structure_score_norm'] * 0.15 +
            df['technical_score_norm'] * 0.10 +
            df['relative_strength_score_norm'] * 0.35 +
            df['quality_score_norm'] * 0.10
        )
        
        return df


class MeanReversionBuySignalGenerator(BuySignalGenerator):
    """均值回归策略买入信号生成器
    
    实现均值回归轨道的买入信号生成逻辑。
    遵循 STRAT-01 双轨隔离要求，与趋势跟踪策略物理隔离。
    """
    
    def generate_buy_signals(
        self, 
        market_data: List[MarketDataPoint], 
        as_of_date: date,
        exclude_list: Optional[List[int]] = None
    ) -> List[Dict]:
        """生成均值回归策略买入信号
        
        依据《dual-track-tech-spec》第3.4节左侧轨道特征，计算超跌得分，
        筛选出超跌且回升迹象明显的行业作为买入候选。
        
        Args:
            market_data: 市场数据点列表
            as_of_date: 截止日期，用于防未来函数
            exclude_list: 排除的行业ID列表
            
        Returns:
            均值回归策略买入信号列表
        """
        if exclude_list is None:
            exclude_list = []
            
        # 将市场数据转换为DataFrame便于处理
        df = pd.DataFrame([{
            'sector_industry_id': md.sector_industry_id,
            'trade_date': md.trade_date,
            'close': md.close,
            'front_adj_close': md.front_adj_close,
            'high': md.high,
            'low': md.low,
            'open': md.open,
            'turnover_rate': md.turnover_rate,
            'total_market_cap': md.total_market_cap,
            'daily_mfv': md.daily_mfv,
            'ma_10': md.ma_10,
            'ma_20': md.ma_20,
            'ma_60': md.ma_60,
            'volatility_20': md.volatility_20,
            'cmf_10': md.cmf_10
        } for md in market_data])
        
        # 只处理指定日期的数据
        df = df[df['trade_date'] == as_of_date]
        
        # 排除指定行业
        df = df[~df['sector_industry_id'].isin(exclude_list)]
        
        # 计算均值回归指标得分
        df = self._calculate_mean_reversion_scores(df)
        
        # 按得分排序，返回买入信号
        signals = []
        df_sorted = df.sort_values(by='mean_reversion_score', ascending=False)
        
        for _, row in df_sorted.iterrows():
            signals.append({
                'sector_id': int(row['sector_industry_id']),
                'score': float(row['mean_reversion_score']),
                'strategy': 'mean_reversion',
                'date': as_of_date,
                'reason': f"Mean reversion score: {row['mean_reversion_score']:.2f}"
            })
        
        return signals
    
    def _calculate_mean_reversion_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均值回归得分
        
        依据《dual-track-tech-spec》第3.4节左侧轨道特征，计算各类均值回归因子得分。
        
        Args:
            df: 市场数据DataFrame
            
        Returns:
            添加了均值回归得分的DataFrame
        """
        # 下跌减速因子 (下影线支撑)
        df['slowdown_score'] = 0.19 * (
            (df['open'] - df['low']) / (df['high'] - df['low'])
        ).fillna(0) * 30  # 乘以30是为了放大效应
        
        # 超跌因子 (乖离率)
        df['oversold_score'] = 0.30 * (
            (df['ma_60'] - df['close']) / df['ma_60']  # 注意这里是ma_60 - close，因为是超跌
        ).clip(lower=0).fillna(0)  # 只关心超跌的情况
        
        # 资金背离因子 - 避免除零错误
        avg_mfv = df['daily_mfv'].rolling(window=20).mean().replace(0, 1e-8)
        df['money_flow_divergence_score'] = 0.12 * (
            df['daily_mfv'] / avg_mfv
        ).fillna(1)
        
        # 波动收缩因子
        rolling_quantile = df['volatility_20'].rolling(window=60).quantile(0.7)
        df['vol_shrink_score'] = 0.11 * (
            (df['volatility_20'] < rolling_quantile).astype(int)
        ).fillna(0)
        
        # 成交量萎缩因子
        avg_turnover = df['turnover_rate'].rolling(window=20).mean().replace(0, 1e-8)
        df['volume_shrink_score'] = 0.37 * (
            (df['turnover_rate'] < avg_turnover * 0.8).astype(int)
        ).fillna(0)  # 在左侧轨道中，下影线支撑是最重要的
        
        # 综合均值回归得分 (0-100分制)
        factors = ['slowdown_score', 'oversold_score', 'money_flow_divergence_score', 
                  'vol_shrink_score', 'volume_shrink_score']
        
        for factor in factors:
            # 标准化到0-100分制
            min_val = df[factor].min()
            max_val = df[factor].max()
            if max_val != min_val:
                df[f"{factor}_norm"] = 100 * (df[factor] - min_val) / (max_val - min_val)
            else:
                df[f"{factor}_norm"] = 50  # 如果所有值相同，则给中间分数
        
        # 计算综合得分 (根据市场状态调整权重，这里使用默认的左侧轨道权重)
        df['mean_reversion_score'] = (
            df['slowdown_score_norm'] * 0.37 +  # 下影线支撑权重最高
            df['oversold_score_norm'] * 0.30 +
            df['money_flow_divergence_score_norm'] * 0.19 +
            df['vol_shrink_score_norm'] * 0.12 +
            df['volume_shrink_score_norm'] * 0.11
        )
        
        return df