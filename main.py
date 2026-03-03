"""
主执行流程脚本
实现完整的调仓逻辑，整合所有模块
依据《技术规格说明书》v1.0 第1.2节部署模式和《接口架构设计》v1.2整体架构
"""

import yaml
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
import sys

from modules.data_layer import DataLoader, SectorMapper
from modules.buy_signal_generator import BuySignalGenerator, TrendBuySignalGenerator, MeanReversionBuySignalGenerator
from modules.sell_signal_generator import SellSignalGenerator, TrendSellSignalGenerator, LeftSellSignalGenerator, CoolDownManager
from modules.risk_manager import RiskManager, CompositeRiskManager
from modules.state_manager import StateRepository, create_state_repository


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def initialize_components(config: Dict[str, Any]) -> tuple:
    """
    初始化所有组件
    返回各个模块实例
    """
    # 初始化数据层
    data_loader = DataLoader()  # 使用默认配置路径
    
    # 初始化状态管理器 - 通过工厂方法实现依赖注入，满足CONS-02逻辑一致性标准
    state_repo: StateRepository = create_state_repository(config['system']['mode'])
    
    # 初始化买入信号生成器（双轨隔离 - STRAT-01）
    buy_signal_generator_trend: BuySignalGenerator = TrendBuySignalGenerator()
    buy_signal_generator_left: BuySignalGenerator = MeanReversionBuySignalGenerator()
    
    # 初始化卖出信号生成器（双轨隔离 - STRAT-01）
    sell_signal_generator_trend: SellSignalGenerator = TrendSellSignalGenerator()
    sell_signal_generator_left: SellSignalGenerator = LeftSellSignalGenerator()
    
    # 初始化冷却期管理器
    cool_down_manager = CoolDownManager(config['screening']['cool_down']['days'])
    
    # 初始化风险管理器
    risk_manager: RiskManager = CompositeRiskManager(
        lookback_days=config['factors']['volatility_control']['lookback_days'],
        target_volatility=config['factors']['volatility_control']['target_volatility']
    )
    
    return (
        data_loader, state_repo,
        buy_signal_generator_trend, buy_signal_generator_left,
        sell_signal_generator_trend, sell_signal_generator_left,
        cool_down_manager, risk_manager
    )


def run_rebalancing_cycle(
    current_date: date,
    data_loader: DataLoader,
    state_repo: StateRepository,
    buy_signal_generator_trend: BuySignalGenerator,
    buy_signal_generator_left: BuySignalGenerator,
    sell_signal_generator_trend: SellSignalGenerator,
    sell_signal_generator_left: SellSignalGenerator,
    cool_down_manager: CoolDownManager,
    risk_manager: RiskManager,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    执行一次完整的调仓循环
    依据《技术规格说明书》v1.0 第5节动态调仓与风控和《接口架构设计》v1.2整体架构
    """
    logger = logging.getLogger(__name__)
    
    # 1. 加载市场数据 - 防未来函数(DB-03)
    # 使用前一天的数据进行分析，确保T日收盘后分析不会使用T日收盘价
    analysis_date = current_date - timedelta(days=1)
    while analysis_date.weekday() >= 5:  # 跳过周末
        analysis_date -= timedelta(days=1)
    
    # 加载过去一段时间的数据用于分析
    start_date = analysis_date - timedelta(days=250)  # 使用250天的历史数据
    market_data = data_loader.load_market_data(
        start_date=start_date,
        end_date=analysis_date,
        as_of_date=analysis_date  # 防未来函数(DB-03)
    )
    
    # 2. 获取当前持仓状态
    current_portfolio = state_repo.load('portfolio_snapshot')
    if current_portfolio is None:
        # 如果没有历史状态，初始化空投资组合
        current_portfolio = {
            'holdings': [],
            'cash_ratio': 1.0,  # 假设初始现金比率为1
            'total_value': 1.0,
            'timestamp': analysis_date.isoformat()
        }
    
    # 3. 更新冷却期状态，移除过期的行业
    cool_down_manager.prune_expired_entries(current_date)
    
    # 4. 生成买入信号（双轨独立）
    # 趋势轨道买入信号
    trend_buy_signals = buy_signal_generator_trend.generate_signals(
        market_data=market_data,
        current_holdings=current_portfolio.get('holdings', []),
        as_of_date=analysis_date
    )
    
    # 左侧轨道买入信号
    left_buy_signals = buy_signal_generator_left.generate_signals(
        market_data=market_data,
        current_holdings=current_portfolio.get('holdings', []),
        as_of_date=analysis_date
    )
    
    # 5. 生成卖出信号（双轨独立）
    # 趋势轨道卖出信号
    trend_sell_signals = sell_signal_generator_trend.check_sell_signals(
        current_holdings=current_portfolio.get('holdings', []),
        market_data=market_data,
        as_of_date=analysis_date
    )
    
    # 左侧轨道卖出信号
    left_sell_signals = sell_signal_generator_left.check_sell_signals(
        current_holdings=current_portfolio.get('holdings', []),
        market_data=market_data,
        as_of_date=analysis_date
    )
    
    # 6. 合并卖出信号
    all_sell_signals = trend_sell_signals + left_sell_signals
    
    # 7. 应用风险管理
    trend_buy_signals = risk_manager.apply_risk_controls(
        signals=trend_buy_signals,
        current_portfolio=current_portfolio,
        market_data=market_data,
        as_of_date=analysis_date
    )
    
    left_buy_signals = risk_manager.apply_risk_controls(
        signals=left_buy_signals,
        current_portfolio=current_portfolio,
        market_data=market_data,
        as_of_date=analysis_date
    )
    
    # 8. 如果有卖出信号，则执行卖出并重新分配资金
    if all_sell_signals:
        logger.info(f"Sell signals triggered: {len(all_sell_signals)}")
        
        # 计算释放的资金
        released_weights = 0.0
        updated_holdings = []
        
        for holding in current_portfolio.get('holdings', []):
            should_sell = any(
                sell_signal.sector_industry_id == holding['sector_industry_id'] 
                for sell_signal in all_sell_signals
            )
            
            if should_sell:
                # 累加释放的权重
                released_weights += holding['weight']
                # 将该行业加入冷却期
                cool_down_manager.add_to_cool_down(holding['sector_industry_id'], current_date)
                logger.info(f"Selling industry {holding['sector_industry_id']} due to sell signal")
            else:
                updated_holdings.append(holding)
        
        # 9. 使用释放的资金购买新行业
        if released_weights > 0 and (trend_buy_signals or left_buy_signals):
            # 按照卖出信号的轨道，优先购买对应轨道的新行业
            trend_replacements = trend_buy_signals[:len([s for s in all_sell_signals if hasattr(s, 'track') and s.track == 'trend'])]
            left_replacements = left_buy_signals[:len([s for s in all_sell_signals if hasattr(s, 'track') and s.track == 'left'])]
            
            # 分配释放的资金给新选中的行业
            if trend_replacements:
                trend_weight_per_industry = released_weights * config['portfolio']['target_weights']['trend_track'] / len(trend_replacements)
                for signal in trend_replacements:
                    updated_holdings.append({
                        'sector_industry_id': signal.sector_industry_id,
                        'name': getattr(signal, 'name', f'Industry_{signal.sector_industry_id}'),
                        'weight': trend_weight_per_industry,
                        'entry_date': current_date.isoformat(),
                        'track': 'trend'
                    })
            
            if left_replacements:
                left_weight_per_industry = released_weights * config['portfolio']['target_weights']['left_track'] / len(left_replacements)
                for signal in left_replacements:
                    updated_holdings.append({
                        'sector_industry_id': signal.sector_industry_id,
                        'name': getattr(signal, 'name', f'Industry_{signal.sector_industry_id}'),
                        'weight': left_weight_per_industry,
                        'entry_date': current_date.isoformat(),
                        'track': 'left'
                    })
        
        # 10. 更新投资组合
        final_portfolio = {
            'date': current_date.isoformat(),
            'holdings': updated_holdings,
            'cash_ratio': current_portfolio.get('cash_ratio', 0.0),
            'total_value': current_portfolio.get('total_value', 1.0),
            'buy_signals_generated': len(trend_buy_signals) + len(left_buy_signals),
            'sell_signals_triggered': len(all_sell_signals)
        }
    else:
        # 没有卖出信号，保持原有持仓
        logger.info("No sell signals triggered, maintaining current holdings")
        final_portfolio = current_portfolio
        final_portfolio['date'] = current_date.isoformat()
        final_portfolio['buy_signals_generated'] = len(trend_buy_signals) + len(left_buy_signals)
        final_portfolio['sell_signals_triggered'] = len(all_sell_signals)
    
    logger.info(f"Rebalancing cycle completed for {current_date}")
    logger.info(f"Total holdings: {len(final_portfolio.get('holdings', []))}, "
                f"Cash ratio: {final_portfolio.get('cash_ratio', 0):.4f}")
    
    return final_portfolio


def main():
    """主执行函数 - RUN-01 脚本化运行"""
    print("Starting Dual Track Screening System...")
    
    # 1. 加载配置
    config = load_config()
    
    # 2. 设置日志
    setup_logging(config['system']['log_level'])
    logger = logging.getLogger(__name__)
    
    # 3. 初始化所有组件
    components = initialize_components(config)
    (
        data_loader, state_repo,
        buy_signal_generator_trend, buy_signal_generator_left,
        sell_signal_generator_trend, sell_signal_generator_left,
        cool_down_manager, risk_manager
    ) = components
    
    # 4. 获取执行日期（这里简化为今天的日期，实际应用中可能从参数获取）
    execution_date = date.today()
    logger.info(f"Running rebalancing for date: {execution_date}")
    
    # 5. 执行调仓循环
    try:
        portfolio_result = run_rebalancing_cycle(
            current_date=execution_date,
            data_loader=data_loader,
            state_repo=state_repo,
            buy_signal_generator_trend=buy_signal_generator_trend,
            buy_signal_generator_left=buy_signal_generator_left,
            sell_signal_generator_trend=sell_signal_generator_trend,
            sell_signal_generator_left=sell_signal_generator_left,
            cool_down_manager=cool_down_manager,
            risk_manager=risk_manager,
            config=config
        )
        
        # 6. 保存最终状态（使用原子写入 - RUN-02）
        state_repo.save('portfolio_snapshot', portfolio_result)
        
        # 保存冷却期状态
        cool_down_state = cool_down_manager.get_state()
        state_repo.save('cool_down_state', cool_down_state)
        
        logger.info("Portfolio rebalancing completed successfully")
        print(f"Final portfolio value: {portfolio_result['total_value']:.4f}")
        print(f"Holdings count: {len(portfolio_result.get('holdings', []))}")
        
    except Exception as e:
        logger.error(f"Error during rebalancing: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()