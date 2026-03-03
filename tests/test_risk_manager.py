"""风险管理模块单元测试 - 四级行业双轨筛选系统

测试RiskManager类的各种风险管理功能
"""

import unittest
from datetime import date, timedelta
from modules.risk_manager import (
    IndustryPosition,
    VolatilityRiskManager,
    CompositeRiskManager,
    create_risk_manager
)


class TestRiskManager(unittest.TestCase):
    """风险管理模块测试类"""
    
    def setUp(self):
        """测试初始化"""
        config = {
            'volatility_target': 0.15,
            'atr_multiplier': 1.5,
            'fixed_drawdown_threshold': 0.08,
            'score_exit_threshold': 60.0,
            'vol_spike_multiplier': 2.5,
            'max_holding_days': 60,
            'ewma_lambda': 0.94,
            'risk_limits': {
                'max_concentration': 0.20,
                'max_l1_exposure': 0.30,
                'max_l2_exposure': 0.15,
            }
        }
        self.risk_manager = create_risk_manager(config)
        self.today = date.today()
    
    def test_should_sell_atr_drawdown(self):
        """测试ATR动态止损功能"""
        # 创建一个亏损严重的持仓，且当前价格远低于入场价，ATR较小
        position = IndustryPosition(
            sector_id=1,
            name="Test_Sector",
            level=4,
            score=80.0,
            entry_date=self.today - timedelta(days=10),
            entry_price=100.0,
            current_price=80.0,  # 20%亏损
            weight=0.1,
            holding_days=10,
            atr_20=5.0  # ATR为5
        )
        
        # ATR止损阈值 = -1.5 * (5.0 / 100.0) = -0.075
        # 当前回撤 = (80-100)/100 = -0.2，小于-0.075，应触发止损
        result = self.risk_manager.should_sell_atr_drawdown(position)
        self.assertTrue(result, "ATR动态止损应被触发")
        
        # 创建一个轻微亏损但ATR很大的持仓
        position2 = IndustryPosition(
            sector_id=2,
            name="Test_Sector2",
            level=4,
            score=80.0,
            entry_date=self.today - timedelta(days=10),
            entry_price=100.0,
            current_price=95.0,  # 5%亏损
            weight=0.1,
            holding_days=10,
            atr_20=20.0  # ATR为20
        )
        
        # ATR止损阈值 = -1.5 * (20.0 / 100.0) = -0.3
        # 当前回撤 = (95-100)/100 = -0.05，大于-0.3，不应触发止损
        result2 = self.risk_manager.should_sell_atr_drawdown(position2)
        self.assertFalse(result2, "ATR动态止损不应被触发")
    
    def test_should_sell_fixed_drawdown(self):
        """测试固定比例止损功能"""
        # 创建一个亏损超过8%的持仓
        position = IndustryPosition(
            sector_id=1,
            name="Test_Sector",
            level=4,
            score=80.0,
            entry_date=self.today - timedelta(days=5),
            entry_price=100.0,
            current_price=91.0,  # 9%亏损
            weight=0.1,
            holding_days=5,
            atr_20=5.0
        )
        
        result = self.risk_manager.should_sell_fixed_drawdown(position)
        self.assertTrue(result, "固定比例止损应被触发")
        
        # 创建一个亏损少于8%的持仓
        position2 = IndustryPosition(
            sector_id=2,
            name="Test_Sector2",
            level=4,
            score=80.0,
            entry_date=self.today - timedelta(days=5),
            entry_price=100.0,
            current_price=93.0,  # 7%亏损
            weight=0.1,
            holding_days=5,
            atr_20=5.0
        )
        
        result2 = self.risk_manager.should_sell_fixed_drawdown(position2)
        self.assertFalse(result2, "固定比例止损不应被触发")
    
    def test_should_sell_on_score_drop(self):
        """测试评分跌破功能"""
        # 创建一个评分跌破60的持仓
        position = IndustryPosition(
            sector_id=1,
            name="Test_Sector",
            level=4,
            score=80.0,
            entry_date=self.today - timedelta(days=3),
            entry_price=100.0,
            current_price=105.0,
            weight=0.1,
            holding_days=3,
            atr_20=3.0,
            current_score=55.0  # 评分低于60
        )
        
        result = self.risk_manager.should_sell_on_score_drop(position)
        self.assertTrue(result, "评分跌破应被触发")
        
        # 创建一个评分高于60的持仓
        position2 = IndustryPosition(
            sector_id=2,
            name="Test_Sector2",
            level=4,
            score=80.0,
            entry_date=self.today - timedelta(days=3),
            entry_price=100.0,
            current_price=105.0,
            weight=0.1,
            holding_days=3,
            atr_20=3.0,
            current_score=70.0  # 评分高于60
        )
        
        result2 = self.risk_manager.should_sell_on_score_drop(position2)
        self.assertFalse(result2, "评分跌破不应被触发")
    
    def test_is_position_expired(self):
        """测试持仓过期功能"""
        # 创建一个超过60天的持仓
        position = IndustryPosition(
            sector_id=1,
            name="Test_Sector",
            level=4,
            score=80.0,
            entry_date=self.today - timedelta(days=65),
            entry_price=100.0,
            current_price=105.0,
            weight=0.1,
            holding_days=65,  # 超过60天
            atr_20=3.0,
            current_score=75.0
        )
        
        result = self.risk_manager.is_position_expired(position)
        self.assertTrue(result, "持仓过期应被触发")
        
        # 创建一个未超过60天的持仓
        position2 = IndustryPosition(
            sector_id=2,
            name="Test_Sector2",
            level=4,
            score=80.0,
            entry_date=self.today - timedelta(days=30),
            entry_price=100.0,
            current_price=105.0,
            weight=0.1,
            holding_days=30,  # 未超过60天
            atr_20=3.0,
            current_score=75.0
        )
        
        result2 = self.risk_manager.is_position_expired(position2)
        self.assertFalse(result2, "持仓过期不应被触发")
    
    def test_apply_volatility_target(self):
        """测试波动率目标控制功能"""
        positions = [
            IndustryPosition(
                sector_id=1,
                name="Test_Sector",
                level=4,
                score=80.0,
                entry_date=self.today - timedelta(days=10),
                entry_price=100.0,
                current_price=102.0,
                weight=0.1,
                holding_days=10,
                atr_20=3.0,
                current_score=75.0
            )
        ]
        
        # 测试高预测波动率的情况
        high_vol = 0.20  # 20%预测波动率
        target_pos = self.risk_manager.apply_volatility_target(positions, high_vol, self.today)
        expected_target = min(1.0, 0.15 / 0.20)  # 0.15是目标波动率
        self.assertEqual(target_pos, expected_target, "波动率目标控制计算应正确")
        
        # 测试低预测波动率的情况
        low_vol = 0.10  # 10%预测波动率
        target_pos2 = self.risk_manager.apply_volatility_target(positions, low_vol, self.today)
        expected_target2 = min(1.0, 0.15 / 0.10)  # 应该是1.0（满仓）
        self.assertEqual(target_pos2, expected_target2, "波动率目标控制计算应正确")
        
        # 测试零波动率的情况
        zero_vol = 0.0
        target_pos3 = self.risk_manager.apply_volatility_target(positions, zero_vol, self.today)
        self.assertEqual(target_pos3, 1.0, "零波动率时应返回满仓")
    
    def test_monitor_portfolio_risk(self):
        """测试组合风险监控功能"""
        positions = [
            IndustryPosition(
                sector_id=1,
                name="A_Test_Sector",  # 前两个字母为"A_"，代表某个L1
                level=4,
                score=80.0,
                entry_date=self.today - timedelta(days=10),
                entry_price=100.0,
                current_price=102.0,
                weight=0.1,
                holding_days=10,
                atr_20=3.0,
                current_score=75.0
            ),
            IndustryPosition(
                sector_id=2,
                name="A_Another_Sector",  # 同样属于"A_" L1
                level=4,
                score=75.0,
                entry_date=self.today - timedelta(days=5),
                entry_price=95.0,
                current_price=97.0,
                weight=0.15,  # 更高的权重
                holding_days=5,
                atr_20=2.5,
                current_score=70.0
            )
        ]
        
        risk_metrics = self.risk_manager.monitor_portfolio_risk(positions, self.today)
        
        # 检查返回的指标是否包含预期的键
        self.assertIn('concentration_risk', risk_metrics)
        self.assertIn('exposure_by_level1', risk_metrics)
        self.assertIn('exposure_by_level2', risk_metrics)
        self.assertIn('avg_holding_days', risk_metrics)
        self.assertIn('total_positions', risk_metrics)
        
        # 检查集中度风险计算
        expected_avg_days = (10 + 5) / 2  # 平均持仓天数
        self.assertEqual(risk_metrics['avg_holding_days'], expected_avg_days)
        
        # 检查总持仓数
        self.assertEqual(risk_metrics['total_positions'], 2)
        
        # 检查L1暴露度
        l1_exposure = risk_metrics['exposure_by_level1']
        expected_a_exposure = 0.1 + 0.15  # 两个持仓的权重之和
        self.assertEqual(l1_exposure.get('L1_A_'), expected_a_exposure)
    
    def test_check_position_risk(self):
        """测试持仓风险检查功能"""
        # 创建一个触发ATR止损的持仓
        position = IndustryPosition(
            sector_id=1,
            name="Test_Sector",
            level=4,
            score=80.0,
            entry_date=self.today - timedelta(days=10),
            entry_price=100.0,
            current_price=80.0,  # 20%亏损
            weight=0.1,
            holding_days=10,
            atr_20=5.0  # ATR为5
        )
        
        should_sell, risk_type = self.risk_manager.check_position_risk(position)
        self.assertTrue(should_sell, "风险检查应识别出需要卖出的持仓")
        self.assertEqual(risk_type, "ATR_DRAWDOWN", "风险类型应为ATR_DRAWDOWN")


if __name__ == '__main__':
    unittest.main()