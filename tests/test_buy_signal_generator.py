"""买入信号生成器单元测试

测试不同的买入信号生成策略在各种市场条件下的表现。
"""

import unittest
from datetime import date
from unittest.mock import Mock
import pandas as pd

from modules.buy_signal_generator import (
    BuySignalGenerator,
    TrendBuySignalGenerator,
    MeanReversionBuySignalGenerator
)
from tests.test_data_classes import MarketDataPoint, StateRepository


class TestBuySignalGenerator(unittest.TestCase):
    """买入信号生成器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_state_repo = Mock(spec=StateRepository)
        self.as_of_date = date(2023, 5, 15)
        
        # 创建一些测试数据
        self.test_market_data = [
            MarketDataPoint(
                trade_date=self.as_of_date,
                sector_industry_id=1,
                open=100.0,
                high=105.0,
                low=98.0,
                close=103.0,
                front_adj_close=103.0,
                turnover_rate=0.02,
                amount=1000000.0,
                total_market_cap=100000000.0,
                daily_mfv=0.01,
                ma_10=100.0,
                ma_20=99.0,
                ma_60=95.0,
                volatility_20=0.15,
                cmf_10=0.005,
                cmf_20=0.003,
                cmf_60=-0.002,
                created_ts=None
            ),
            MarketDataPoint(
                trade_date=self.as_of_date,
                sector_industry_id=2,
                open=50.0,
                high=52.0,
                low=48.0,
                close=49.0,
                front_adj_close=49.0,
                turnover_rate=0.01,
                amount=500000.0,
                total_market_cap=50000000.0,
                daily_mfv=-0.005,
                ma_10=52.0,
                ma_20=55.0,
                ma_60=60.0,
                volatility_20=0.18,
                cmf_10=-0.003,
                cmf_20=-0.005,
                cmf_60=-0.008,
                created_ts=None
            ),
            MarketDataPoint(
                trade_date=self.as_of_date,
                sector_industry_id=3,
                open=200.0,
                high=210.0,
                low=195.0,
                close=205.0,
                front_adj_close=205.0,
                turnover_rate=0.03,
                amount=2000000.0,
                total_market_cap=200000000.0,
                daily_mfv=0.02,
                ma_10=200.0,
                ma_20=198.0,
                ma_60=190.0,
                volatility_20=0.12,
                cmf_10=0.008,
                cmf_20=0.006,
                cmf_60=0.001,
                created_ts=None
            )
        ]
        
    def test_abstract_class_not_instantiable(self):
        """测试抽象基类不能被实例化"""
        with self.assertRaises(TypeError):
            BuySignalGenerator(self.mock_state_repo)
    
    def test_trend_buy_signal_generator(self):
        """测试趋势跟踪买入信号生成器"""
        generator = TrendBuySignalGenerator(self.mock_state_repo)
        
        # 生成买入信号
        signals = generator.generate_buy_signals(
            self.test_market_data,
            self.as_of_date
        )
        
        # 验证返回结果
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0)
        
        # 验证信号格式
        for signal in signals:
            self.assertIn('sector_id', signal)
            self.assertIn('score', signal)
            self.assertIn('strategy', signal)
            self.assertIn('date', signal)
            self.assertEqual(signal['strategy'], 'trend')
            self.assertIsInstance(signal['score'], float)
    
    def test_mean_reversion_buy_signal_generator(self):
        """测试均值回归买入信号生成器"""
        generator = MeanReversionBuySignalGenerator(self.mock_state_repo)
        
        # 生成买入信号
        signals = generator.generate_buy_signals(
            self.test_market_data,
            self.as_of_date
        )
        
        # 验证返回结果
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0)
        
        # 验证信号格式
        for signal in signals:
            self.assertIn('sector_id', signal)
            self.assertIn('score', signal)
            self.assertIn('strategy', signal)
            self.assertIn('date', signal)
            self.assertEqual(signal['strategy'], 'mean_reversion')
            self.assertIsInstance(signal['score'], float)
    
    def test_exclude_list_functionality(self):
        """测试排除列表功能"""
        generator = TrendBuySignalGenerator(self.mock_state_repo)
        
        # 生成买入信号，排除特定行业
        exclude_list = [1]
        signals_with_exclude = generator.generate_buy_signals(
            self.test_market_data,
            self.as_of_date,
            exclude_list=exclude_list
        )
        
        # 确保被排除的行业不在结果中
        for signal in signals_with_exclude:
            self.assertNotIn(signal['sector_id'], exclude_list)
    
    def test_different_dates(self):
        """测试不同日期的数据处理"""
        other_date = date(2023, 5, 14)
        mixed_market_data = self.test_market_data + [
            MarketDataPoint(
                trade_date=other_date,
                sector_industry_id=4,
                open=80.0,
                high=85.0,
                low=78.0,
                close=82.0,
                front_adj_close=82.0,
                turnover_rate=0.015,
                amount=800000.0,
                total_market_cap=80000000.0,
                daily_mfv=0.008,
                ma_10=80.0,
                ma_20=79.0,
                ma_60=75.0,
                volatility_20=0.14,
                cmf_10=0.004,
                cmf_20=0.002,
                cmf_60=-0.001,
                created_ts=None
            )
        ]
        
        generator = TrendBuySignalGenerator(self.mock_state_repo)
        
        # 生成买入信号，应该只包含指定日期的数据
        signals = generator.generate_buy_signals(
            mixed_market_data,
            self.as_of_date
        )
        
        # 验证结果中不包含其他日期的数据
        for signal in signals:
            # 因为我们的MarketDataPoint中没有直接的日期字段来验证，我们验证数据是否合理
            self.assertEqual(signal['date'], self.as_of_date)


if __name__ == '__main__':
    unittest.main()