#!/usr/bin/env python3
"""
四级行业双轨筛选系统 - 主入口

执行完整回测流程
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logger, load_config, build_hierarchy_mapping
from data import DataLoader, LifecycleBuilder, DataPreprocessor
from features import TrendFeatureCalculator, LeftFeatureCalculator
from selection import TrendSelector, LeftSelector
from backtest import BacktestEngine, BenchmarkBuilder
from evaluation import MetricsCalculator, Visualizer, ReportGenerator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='四级行业双轨筛选系统'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='输出目录'
    )
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='测试数据库连接'
    )
    
    return parser.parse_args()


def test_database_connection(config: dict):
    """测试数据库连接"""
    print("测试数据库连接...")
    loader = DataLoader()
    success = loader.test_connection()
    
    if success:
        print("数据库连接测试通过！")
        # 尝试加载一些数据
        try:
            metadata = loader.load_sector_metadata()
            print(f"成功加载 {len(metadata)} 条行业元数据")
        except Exception as e:
            print(f"加载数据失败: {e}")
    else:
        print("数据库连接测试失败，请检查配置")
    
    return success


def run_backtest(config: dict, output_dir: str):
    """
    执行回测
    
    Args:
        config: 配置字典
        output_dir: 输出目录
    """
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("四级行业双轨筛选系统 - 回测开始")
    logger.info("=" * 60)
    
    # 1. 加载数据
    logger.info("【1/7】加载数据...")
    data_config = config.get('data', {})
    start_date = data_config.get('start_date')
    end_date = data_config.get('end_date')
    
    loader = DataLoader()
    raw_data = loader.load_all_data(start_date, end_date)
    
    sector_metadata = raw_data['sector_metadata']
    sector_hierarchy = raw_data['sector_hierarchy']
    daily_data = raw_data['daily_data']
    
    logger.info(f"数据加载完成: {len(daily_data)} 条记录")
    
    # 2. 构建生命周期表
    logger.info("【2/7】构建行业生命周期表...")
    lifecycle_builder = LifecycleBuilder(daily_data)
    lifecycle_table = lifecycle_builder.build_lifecycle_table()
    
    # 3. 数据预处理
    logger.info("【3/7】数据预处理...")
    preprocessor = DataPreprocessor(daily_data)
    processed_data = preprocessor.preprocess(
        coverage_threshold=data_config.get('validation', {}).get('coverage_threshold', 0.90),
        missing_threshold=data_config.get('validation', {}).get('missing_threshold', 0.05)
    )
    
    # 添加层级信息
    processed_data = processed_data.merge(
        sector_metadata[['id', 'name', 'level']],
        left_on='sector_industry_id',
        right_on='id',
        how='left'
    )
    
    logger.info(f"预处理完成: {len(processed_data)} 条记录")
    
    # 4. 初始化回测引擎
    logger.info("【4/7】初始化回测引擎...")
    hierarchy_mapping = build_hierarchy_mapping(sector_hierarchy, sector_metadata)
    engine = BacktestEngine(config)
    engine.initialize(processed_data, hierarchy_mapping, lifecycle_table)
    
    # 5. 执行回测
    logger.info("【5/7】执行回测...")
    results = engine.run_backtest()
    
    logger.info(f"回测完成，最终净值: {results['final_nav']:.4f}")
    logger.info(f"总收益率: {results['total_return']:.2%}")
    
    # 6. 构建基准
    logger.info("【6/7】构建基准...")
    level1_data = processed_data[processed_data['level'] == 1].copy()
    benchmark_builder = BenchmarkBuilder()
    benchmark = benchmark_builder.build_benchmark(level1_data)
    
    # 7. 绩效评估
    logger.info("【7/7】绩效评估...")
    
    # 计算指标
    metrics_calc = MetricsCalculator()
    benchmark_returns = benchmark_builder.get_benchmark_returns()
    metrics = metrics_calc.calculate_all_metrics(results, benchmark_returns)
    
    # 打印指标
    logger.info("\n" + "=" * 60)
    logger.info("绩效指标")
    logger.info("=" * 60)
    
    general = metrics.get('general', {})
    logger.info(f"年化收益率: {general.get('annualized_return', 0):.2%}")
    logger.info(f"年化波动率: {general.get('annualized_volatility', 0):.2%}")
    logger.info(f"夏普比率: {general.get('sharpe_ratio', 0):.2f}")
    logger.info(f"最大回撤: {general.get('max_drawdown', 0):.2%}")
    logger.info(f"胜率: {general.get('win_rate', 0):.2%}")
    
    if 'relative' in metrics:
        relative = metrics.get('relative', {})
        logger.info(f"Alpha: {relative.get('alpha', 0):.4f}")
        logger.info(f"Beta: {relative.get('beta', 0):.4f}")
        logger.info(f"信息比率: {relative.get('information_ratio', 0):.2f}")
    
    # 生成可视化
    # visualizer = Visualizer(output_dir)
    # visualizer.save_all_plots(results, benchmark)
    
    # 生成报告
    report_gen = ReportGenerator(output_dir)
    report_path = report_gen.generate_excel_report(
        metrics,
        results.get('transactions', pd.DataFrame()),
        results
    )
    
    logger.info(f"\n报告已生成: {report_path}")
    logger.info("=" * 60)
    logger.info("回测完成！")
    logger.info("=" * 60)
    
    return results, metrics


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    print(f"已加载配置: {args.config}")
    
    # 测试数据库连接
    if args.test_connection:
        success = test_database_connection(config)
        sys.exit(0 if success else 1)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 执行回测
    try:
        results, metrics = run_backtest(config, args.output)
        print("\n回测执行成功！")
        print(f"输出目录: {os.path.abspath(args.output)}")
    except Exception as e:
        print(f"\n回测执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
