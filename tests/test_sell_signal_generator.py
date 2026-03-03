"""
卖出信号生成器单元测试

测试不同卖出信号的检测功能，包括ATR止损、固定止损、持有期到期、评分跌破、异常波动
"""

import unittest
from datetime import date, timedelta
import pandas as pd
from unittest.mock import Mock, MagicMock

from modules.sell_signal_generator import (
    SellSignal,
    TrendSellSignalGenerator,
    LeftSellSignalGenerator,
    CoolDownManager
)
from dataclasses import dataclass
from modules.state_manager import StateRepository


@dataclass
class MockIndustry:
    sector_id: int
    name: str
    level: int
    entry_date: date
    holding_days: int = 0


@dataclass
class MockPosition:
    industry: MockIndustry
    weight: float
    entry_price: float
    current_price: float


class TestSellSignalGenerator(unittest.TestCase):
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'atr_stop_loss_multiplier': 0.5,  # 降低ATR倍数，使ATR止损不那么容易触发
            'fixed_stop_loss_pct': 0.08,  # 8%
            'score_exit_threshold': 60.0,
            'vol_spike_multiplier': 2.5,
            'left_holding_period_limit': 60  # 60天
        }
        
        # 创建测试用的持仓
        self.test_industry = MockIndustry(
            sector_id=123,
            name="Test Industry",
            level=4,
            entry_date=date.today() - timedelta(days=10),
            holding_days=10
        )
        
        self.test_position = MockPosition(
            industry=self.test_industry,
            weight=0.1,
            entry_price=100.0,
            current_price=95.0  # 有5%的亏损
        )
        
        # 创建测试用的市场数据
        self.market_data = pd.DataFrame({
            'sector_industry_id': [123, 456],
            'trade_date': [date.today(), date.today()],
            'front_adj_close': [95.0, 110.0],
            'atr_20': [3.0, 2.5]
        })
        
        # 评分数据
        self.scores = {123: 55.0, 456: 75.0}  # 123的分数低于阈值
    
    def test_trend_sell_signal_generator(self):
        """测试趋势轨道卖出信号生成器"""
        generator = TrendSellSignalGenerator(self.config)
        
        positions = [self.test_position]
        as_of_date = date.today()
        
        # 测试评分跌破信号
        signals = generator.evaluate_sell_signals(
            positions=positions,
            market_data=self.market_data,
            scores=self.scores,
            as_of_date=as_of_date
        )
        
        # 应该有一个评分跌破的信号
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, 'score_drop')
        self.assertIn('评分跌破阈值', signals[0].reason)
        self.assertEqual(signals[0].industry_id, 123)
    
    def test_trend_fixed_stop_loss(self):
        """测试趋势轨道固定止损"""
        # 创建一个亏损超过8%的持仓
        deep_loss_position = MockPosition(
            industry=self.test_industry,
            weight=0.1,
            entry_price=100.0,
            current_price=91.0  # 9%的亏损
        )
        
        generator = TrendSellSignalGenerator(self.config)
        
        signals = generator.evaluate_sell_signals(
            positions=[deep_loss_position],
            market_data=self.market_data,
            scores={123: 70.0},  # 分数高于阈值，不会触发评分跌破
            as_of_date=date.today()
        )
        
        # 应该有一个固定止损信号
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, 'fixed_stop_loss')
        self.assertIn("回撤超过8%", signals[0].reason)
    
    def test_left_sell_signal_generator_with_holding_period(self):
        """测试左侧轨道卖出信号生成器（包含持有期到期）"""
        # 创建一个持有期超过60天的持仓
        old_industry = MockIndustry(
            sector_id=124,
            name="Old Industry",
            level=4,
            entry_date=date.today() - timedelta(days=65),
            holding_days=65
        )
        
        old_position = MockPosition(
            industry=old_industry,
            weight=0.1,
            entry_price=100.0,
            current_price=105.0
        )
        
        generator = LeftSellSignalGenerator(self.config)
        
        signals = generator.evaluate_sell_signals(
            positions=[old_position],
            market_data=pd.DataFrame({
                'sector_industry_id': [124],
                'trade_date': [date.today()],
                'front_adj_close': [105.0],
                'atr_20': [2.0]
            }),
            scores={124: 70.0},
            as_of_date=date.today()
        )
        
        # 应该有一个持有期到期的信号（左侧轨道优先检查）
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, 'holding_period')
        self.assertIn("持有期到期", signals[0].reason)
    
    def test_signal_priority(self):
        """测试信号优先级 - 一旦触发一个信号就不继续检查其他条件"""
        # 创建一个同时满足多个条件的持仓
        bad_industry = MockIndustry(
            sector_id=125,
            name="Bad Industry",
            level=4,
            entry_date=date.today() - timedelta(days=65),  # 超过持有期
            holding_days=65
        )
        
        bad_position = MockPosition(
            industry=bad_industry,
            weight=0.1,
            entry_price=100.0,
            current_price=90.0  # 同时也超过止损线
        )
        
        generator = LeftSellSignalGenerator(self.config)
        
        signals = generator.evaluate_sell_signals(
            positions=[bad_position],
            market_data=pd.DataFrame({
                'sector_industry_id': [125],
                'trade_date': [date.today()],
                'front_adj_close': [90.0],
                'atr_20': [2.0]
            }),
            scores={125: 50.0},  # 评分也低于阈值
            as_of_date=date.today()
        )
        
        # 应该只有一个信号，且是持有期到期（优先级最高，对于左侧轨道）
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, 'holding_period')
    
    def test_no_signals_when_conditions_not_met(self):
        """测试条件不满足时不产生信号"""
        good_position = MockPosition(
            industry=self.test_industry,
            weight=0.1,
            entry_price=100.0,
            current_price=105.0  # 上涨5%
        )
        
        generator = TrendSellSignalGenerator(self.config)
        
        signals = generator.evaluate_sell_signals(
            positions=[good_position],
            market_data=pd.DataFrame({
                'sector_industry_id': [123],
                'trade_date': [date.today()],
                'front_adj_close': [105.0],
                'atr_20': [2.0]
            }),
            scores={123: 80.0},  # 分数很高
            as_of_date=date.today()
        )
        
        # 不应该有任何信号
        self.assertEqual(len(signals), 0)


class TestCoolDownManager(unittest.TestCase):
    
    def setUp(self):
        """测试前准备"""
        # 创建Mock的状态仓库
        self.mock_state_repo = Mock(spec=StateRepository)
        self.mock_state_repo.load = Mock(return_value={})
        self.mock_state_repo.save = Mock()
        
    def test_add_and_check_cool_down(self):
        """测试添加到冷却池和检查功能"""
        manager = CoolDownManager(self.mock_state_repo, cool_down_days=10)
        
        test_date = date.today()
        industry_id = 123
        
        # 添加到冷却池
        manager.add_to_cool_down(industry_id, test_date)
        
        # 检查是否在冷却期内（应该在）
        self.assertTrue(manager.is_in_cool_down(industry_id, test_date))
        
        # 检查一个很远的未来日期（应该不在冷却期）
        future_date = test_date + timedelta(days=15)
        self.assertFalse(manager.is_in_cool_down(industry_id, future_date))
    
    def test_prune_expired_entries(self):
        """测试清理过期条目"""
        manager = CoolDownManager(self.mock_state_repo, cool_down_days=10)
        
        test_date = date.today()
        industry_id1 = 123
        industry_id2 = 456
        
        # 添加两个行业到冷却池
        manager.add_to_cool_down(industry_id1, test_date)
        manager.add_to_cool_down(industry_id2, test_date)
        
        # 模拟当前日期已经过了冷却期
        current_date = test_date + timedelta(days=15)
        
        # 清理过期条目
        manager.prune_expired_entries(current_date)
        
        # 检查是否已清理
        self.assertEqual(len(manager.cool_down_records), 0)
    
    def test_load_initial_state(self):
        """测试初始状态加载"""
        initial_state = {
            123: {
                'sell_date': '2023-01-01',
                'unlock_date': '2023-01-15'
            }
        }
        
        self.mock_state_repo.load.return_value = initial_state
        
        manager = CoolDownManager(self.mock_state_repo, cool_down_days=10)
        
        # 检查是否正确加载了初始状态
        self.assertEqual(manager.cool_down_records, initial_state)


if __name__ == '__main__':
    unittest.main()