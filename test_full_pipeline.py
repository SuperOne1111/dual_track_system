#!/usr/bin/env python3
"""
完整流程测试脚本

测试整个系统的各个模块
"""

import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# -------------------------
# 环境与日志配置
# -------------------------
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config


def test_config():
    """测试配置加载"""
    print("\n" + "=" * 60)
    print("测试配置加载")
    print("=" * 60)
    
    try:
        config = load_config('config.yaml')
        print("配置加载成功！")
        print(f"回测区间: {config['backtest']['start_date']} 至 {config['backtest']['end_date']}")
        print(f"初始资金: {config['backtest']['initial_capital']:,}")
        return True
    except Exception as e:
        print(f"配置加载失败: {e}")
        return False


def test_data_modules():
    """测试数据层模块"""
    print("\n" + "=" * 60)
    print("测试数据层模块")
    print("=" * 60)
    
    try:
        from data import DataLoader, LifecycleBuilder, DataPreprocessor
        print("数据层模块导入成功！")
        
        loader = DataLoader()
        print("数据加载器初始化成功")
        
        return True
    except Exception as e:
        print(f"数据层模块测试失败: {e}")
        return False


def test_feature_modules():
    """测试特征层模块"""
    print("\n" + "=" * 60)
    print("测试特征层模块")
    print("=" * 60)
    
    try:
        from features import TrendFeatureCalculator, LeftFeatureCalculator
        print("特征层模块导入成功！")
        
        config = load_config('config.yaml')
        
        # 测试趋势特征计算器
        trend_calc = TrendFeatureCalculator(config)
        print(f"趋势特征计算器初始化成功")
        
        # 测试左侧特征计算器
        left_calc = LeftFeatureCalculator(config)
        print(f"左侧特征计算器初始化成功")
        
        return True
    except Exception as e:
        print(f"特征层模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_selection_modules():
    """测试筛选层模块"""
    print("\n" + "=" * 60)
    print("测试筛选层模块")
    print("=" * 60)
    
    try:
        from selection import TrendSelector, LeftSelector
        print("筛选层模块导入成功！")
        
        config = load_config('config.yaml')
        
        # 测试趋势筛选器
        trend_selector = TrendSelector(config)
        print(f"趋势筛选器初始化成功")
        
        # 测试左侧筛选器
        left_selector = LeftSelector(config)
        print(f"左侧筛选器初始化成功")
        
        return True
    except Exception as e:
        print(f"筛选层模块测试失败: {e}")
        return False


def test_rebalance_modules():
    """测试动态调仓模块"""
    print("\n" + "=" * 60)
    print("测试动态调仓模块")
    print("=" * 60)
    
    try:
        from rebalance import SellMonitor, SupplementEngine, PositionManager
        print("动态调仓模块导入成功！")
        
        config = load_config('config.yaml')
        
        # 测试卖出监控器
        sell_monitor = SellMonitor(config)
        print(f"卖出监控器初始化成功")
        
        # 测试补充引擎
        supplement_engine = SupplementEngine(config)
        print(f"补充引擎初始化成功")
        
        # 测试持仓管理器
        position_manager = PositionManager(config)
        print(f"持仓管理器初始化成功")
        
        return True
    except Exception as e:
        print(f"动态调仓模块测试失败: {e}")
        return False


def test_backtest_modules():
    """测试回测引擎模块"""
    print("\n" + "=" * 60)
    print("测试回测引擎模块")
    print("=" * 60)
    
    try:
        from backtest import BacktestEngine, BenchmarkBuilder, TradeExecutor
        print("回测引擎模块导入成功！")
        
        config = load_config('config.yaml')
        
        # 测试回测引擎
        engine = BacktestEngine(config)
        print(f"回测引擎初始化成功")
        
        # 测试基准构建器
        benchmark_builder = BenchmarkBuilder()
        print(f"基准构建器初始化成功")
        
        # 测试交易执行器
        config = load_config('config.yaml')
        executor = TradeExecutor(config)
        print(f"交易执行器初始化成功")
        
        return True
    except Exception as e:
        print(f"回测引擎模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_modules():
    """测试绩效评估模块"""
    print("\n" + "=" * 60)
    print("测试绩效评估模块")
    print("=" * 60)
    
    try:
        from evaluation import MetricsCalculator, Visualizer, ReportGenerator
        print("绩效评估模块导入成功！")
        
        # 测试指标计算器
        metrics_calc = MetricsCalculator()
        print(f"指标计算器初始化成功")
        
        # 测试可视化器
        visualizer = Visualizer('output')
        print(f"可视化器初始化成功")
        
        # 测试报告生成器
        report_gen = ReportGenerator('output')
        print(f"报告生成器初始化成功")
        
        return True
    except Exception as e:
        print(f"绩效评估模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils_modules():
    """测试工具模块"""
    print("\n" + "=" * 60)
    print("测试工具模块")
    print("=" * 60)
    
    try:
        from utils import setup_logger, date_range, load_config
        print("工具模块导入成功！")
        
        # 测试日志设置
        logger = setup_logger('test')
        print(f"日志设置成功")
        
        return True
    except Exception as e:
        print(f"工具模块测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("四级行业双轨筛选系统 - 完整流程测试")
    print("=" * 60)

    if not os.getenv("PG_CONN_STRING"):
        print("失败: 未设置 PG_CONN_STRING。根据测试策略，本测试必须连接真实PostgreSQL。")
        return False
    
    tests = [
        ("配置加载", test_config),
        ("数据层模块", test_data_modules),
        ("特征层模块", test_feature_modules),
        ("筛选层模块", test_selection_modules),
        ("动态调仓模块", test_rebalance_modules),
        ("回测引擎模块", test_backtest_modules),
        ("绩效评估模块", test_evaluation_modules),
        ("工具模块", test_utils_modules),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n{name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印测试结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "通过" if result else "失败"
        symbol = "[PASS]" if result else "[FAIL]"
        print(f"{symbol} {name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"总计: {len(results)} 项测试, {passed} 项通过, {failed} 项失败")
    print("=" * 60)
    
    if failed == 0:
        print("\n所有测试通过！项目可以正常运行。")
        print("注意: 数据层模块需要安装psycopg2并配置PostgreSQL数据库")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
