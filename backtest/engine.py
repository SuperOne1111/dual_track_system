"""
回测引擎

执行完整回测流程（生命周期切片 + 月度调仓 + 动态补充 + T+1执行）
"""

import copy
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

try:
    from ..features import TrendFeatureCalculator, LeftFeatureCalculator
    from ..selection import TrendSelector, LeftSelector
    from ..rebalance import SellMonitor, SupplementEngine, PositionManager
    from .benchmark import BenchmarkBuilder
    from .executor import TradeExecutor
    from .regime import MarketRegimeManager
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from features import TrendFeatureCalculator, LeftFeatureCalculator
    from selection import TrendSelector, LeftSelector
    from rebalance import SellMonitor, SupplementEngine, PositionManager
    from backtest.benchmark import BenchmarkBuilder
    from backtest.executor import TradeExecutor
    from backtest.regime import MarketRegimeManager


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: dict):
        self.base_config = copy.deepcopy(config)
        self.config = copy.deepcopy(config)
        self.backtest_config = self.config.get('backtest', {})
        self.rebalance_cfg = self.backtest_config.get('rebalance', {})

        self.trend_calc = TrendFeatureCalculator(self.config)
        self.left_calc = LeftFeatureCalculator(self.config)
        self.trend_selector = TrendSelector(self.config)
        self.left_selector = LeftSelector(self.config)
        self.sell_monitor = SellMonitor(self.config)
        self.supplement_engine = SupplementEngine(self.config)
        self.position_manager = PositionManager(self.config)
        self.executor = TradeExecutor(self.config)
        self.benchmark_builder = BenchmarkBuilder()
        self.regime_manager = MarketRegimeManager(self.base_config)

        self.current_regime_state = 'init'
        self.current_regime_flags = {}

        self.equity_curve = []
        self.daily_returns = []
        self.daily_cash_weights = []
        self.pending_rebalance_orders = []
        self.pending_buy_orders = []
        self.pending_cost_ratio = 0.0
        self.unallocated_records = []
        self.regime_records = []
        self.trade_dates = []
        self.date_to_next = {}

    def initialize(self,
                   data: pd.DataFrame,
                   hierarchy_mapping: Dict,
                   lifecycle_table: pd.DataFrame):
        """
        初始化回测

        Args:
            data: 预处理后的数据
            hierarchy_mapping: 行业层级映射
            lifecycle_table: 生命周期表
        """
        self.data = data.copy().sort_values(['trade_date', 'sector_industry_id'])
        self.hierarchy = hierarchy_mapping
        self.lifecycle_table = lifecycle_table.copy()

        self.initial_capital = self.backtest_config.get('initial_capital', 1000000)
        self.start_date = pd.to_datetime(self.backtest_config.get('start_date'))
        self.end_date = pd.to_datetime(self.backtest_config.get('end_date'))
        self.execution_delay = int(self.rebalance_cfg.get('execution_delay', 1))

        self.current_nav = 1.0
        self.equity_curve = [{
            'date': self.start_date,
            'nav': 1.0,
            'portfolio_value': self.initial_capital
        }]
        self.daily_returns = []
        self.daily_cash_weights = []
        self.pending_rebalance_orders = []
        self.pending_buy_orders = []
        self.pending_cost_ratio = 0.0
        self.unallocated_records = []
        self.regime_records = []
        self.current_regime_state = 'init'
        self.current_regime_flags = {}
        self.trade_dates = []
        self.date_to_next = {}
        self._sync_modules_with_config(self.config)

        print("回测初始化完成")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"回测区间: {self.start_date.date()} 至 {self.end_date.date()}")

    def run_backtest(self) -> Dict:
        trade_dates = self.data[
            (self.data['trade_date'] >= self.start_date) &
            (self.data['trade_date'] <= self.end_date)
        ]['trade_date'].drop_duplicates().sort_values().tolist()
        self.trade_dates = trade_dates
        self.date_to_next = {
            trade_dates[i]: (trade_dates[i + 1] if i + 1 < len(trade_dates) else None)
            for i in range(len(trade_dates))
        }

        self.regime_manager.prepare(self.data, trade_dates)

        rebalance_dates = set(self._get_rebalance_dates(trade_dates))
        print(f"总交易日: {len(trade_dates)}")
        print(f"调仓信号日: {len(rebalance_dates)}")

        for i, trade_date in enumerate(trade_dates):
            self._apply_regime_for_date(trade_date)
            history_data, day_data = self._get_data_slice_at_date(trade_date)
            if day_data.empty:
                continue

            # 1) 当日收益仅按开盘前持仓计算（防同日换仓吃到当日收益）
            positions_for_return = self.position_manager.get_portfolio().get('positions', {}).copy()
            self._calculate_daily_nav(trade_date, day_data, positions_for_return)

            # 2) 执行到期买单（下一交易日开盘价）
            self._execute_pending_buys(trade_date, day_data)

            # 3) 执行到期再平衡（卖出执行，买入挂到下一交易日开盘）
            self._execute_pending_rebalance(trade_date, day_data)

            if trade_date in rebalance_dates:
                current_positions = self.position_manager.get_portfolio().get('positions', {})
                # 改为事件驱动：仅在空仓时进行建仓，不再每月强制换仓
                if not current_positions:
                    self._schedule_monthly_rebalance(trade_date, history_data, day_data, trade_dates)

            # 4) 日监控：卖出后双轨重筛，买入挂到下一交易日开盘执行
            self._run_daily_monitor(trade_date, history_data, day_data)

            if (i + 1) % 60 == 0:
                print(f"进度: {i+1}/{len(trade_dates)} 日, 当前净值: {self.current_nav:.4f}")

        return self._build_results()

    def _get_rebalance_dates(self, trade_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
        df = pd.DataFrame({'trade_date': pd.to_datetime(trade_dates)})
        df['year_month'] = df['trade_date'].dt.to_period('M')
        return df.groupby('year_month')['trade_date'].max().tolist()

    def _sync_modules_with_config(self, cfg: Dict):
        """Sync config-dependent module attributes for dynamic regime overrides."""
        self.trend_calc.config = cfg
        self.trend_calc.score_calc.config = cfg
        self.trend_calc.score_calc.weights = cfg.get('trend', {}).get('weights', {})

        self.left_calc.config = cfg
        self.left_calc.score_calc.config = cfg
        self.left_calc.score_calc.weights = cfg.get('left', {}).get('score_weights', {})

        self.trend_selector.config = cfg
        self.trend_selector.selection_config = cfg.get('selection', {}).get('trend', {})

        self.left_selector.config = cfg
        self.left_selector.left_config = cfg.get('left', {})
        self.left_selector.selection_config = self.left_selector.left_config.get('selection', {})

        self.sell_monitor.config = cfg
        self.sell_monitor.risk_config = cfg.get('risk_control', {})
        self.sell_monitor.stop_loss_config = self.sell_monitor.risk_config.get('stop_loss', {})

        self.supplement_engine.config = cfg
        self.supplement_engine.supplement_config = cfg.get('supplement', {})
        self.supplement_engine.candidate_config = self.supplement_engine.supplement_config.get('candidate_pool', {})

        self.position_manager.config = cfg
        self.position_manager.risk_config = cfg.get('risk_control', {})
        self.position_manager.position_config = self.position_manager.risk_config.get('position', {})

        self.executor.config = cfg
        self.executor.transaction_cost = cfg.get('backtest', {}).get('transaction_cost', {})

        self.backtest_config = cfg.get('backtest', {})
        self.rebalance_cfg = self.backtest_config.get('rebalance', {})
        self.execution_delay = int(self.rebalance_cfg.get('execution_delay', 1))

    def _apply_regime_for_date(self, trade_date: pd.Timestamp):
        effective_cfg, state, flags = self.regime_manager.get_effective_config(trade_date)
        self.config = effective_cfg
        self._sync_modules_with_config(effective_cfg)

        if state != self.current_regime_state:
            print(
                f"Regime switch @ {pd.to_datetime(trade_date).date()}: "
                f"{self.current_regime_state} -> {state}; flags={flags}"
            )
            self.current_regime_state = state
            self.current_regime_flags = flags

        self.regime_records.append({
            'date': pd.to_datetime(trade_date),
            'state': state,
            **{f'flag_{k}': bool(v) for k, v in flags.items()}
        })

    def _get_active_lifecycle_at_date(self, trade_date: pd.Timestamp) -> pd.DataFrame:
        t = pd.to_datetime(trade_date)
        active = self.lifecycle_table[
            (self.lifecycle_table['start_date'] <= t) &
            (self.lifecycle_table['end_date'] >= t)
        ][['sector_industry_id', 'start_date']].copy()
        return active

    def _get_data_slice_at_date(self, trade_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        active = self._get_active_lifecycle_at_date(trade_date)
        if active.empty:
            return pd.DataFrame(), pd.DataFrame()

        merged = self.data.merge(active, on='sector_industry_id', how='inner')
        history = merged[
            (merged['trade_date'] <= trade_date) &
            (merged['trade_date'] >= merged['start_date'])
        ].copy()
        day = history[history['trade_date'] == trade_date].copy()
        history = history.drop(columns=['start_date'])
        day = day.drop(columns=['start_date'])
        return history, day

    def _schedule_monthly_rebalance(self, trade_date: pd.Timestamp,
                                    history_data: pd.DataFrame,
                                    day_data: pd.DataFrame,
                                    trade_dates: List[pd.Timestamp]):
        trend_snapshot = self.trend_calc.calculate_snapshot(history_data, trade_date, self.hierarchy)
        trend_snapshot['S_trend'] = self.trend_calc.calculate_score(trend_snapshot)

        trend_selection = self.trend_selector.select(trend_snapshot, self.hierarchy, trade_date)
        trend_pool = trend_selection['trend_pool']

        left_snapshot = self.left_calc.calculate_snapshot(history_data, trade_date, self.hierarchy)
        left_snapshot['S_recovery'] = self.left_calc.calculate_score(left_snapshot)
        returns = day_data.set_index('sector_industry_id')['r_30']
        left_selection = self.left_selector.select(left_snapshot, trend_pool, returns, trade_date)
        left_pool = left_selection['left_pool']

        target_weights, target_types, score_map, cash_reserve, unallocated_details = self._build_target_portfolio(
            trend_pool, left_pool, trend_snapshot, left_snapshot
        )

        for detail in unallocated_details:
            if detail.get('remaining_weight', 0.0) > 0:
                self.unallocated_records.append({
                    'trade_date': trade_date,
                    'source': 'monthly_rebalance',
                    **detail
                })

        execute_date = self._get_execution_date(trade_date, trade_dates)
        self.pending_rebalance_orders.append({
            'signal_date': trade_date,
            'execute_date': execute_date,
            'target_weights': target_weights,
            'target_types': target_types,
            'score_map': score_map,
            'cash_reserve': cash_reserve
        })

    def _get_execution_date(self, signal_date: pd.Timestamp, trade_dates: List[pd.Timestamp]) -> pd.Timestamp:
        idx = trade_dates.index(signal_date)
        target_idx = min(idx + self.execution_delay, len(trade_dates) - 1)
        return trade_dates[target_idx]

    def _build_target_portfolio(self, trend_pool: List, left_pool: List,
                                trend_scores: pd.DataFrame, left_scores: pd.DataFrame):
        allocation = self.backtest_config.get('allocation', {})
        trend_weight = allocation.get('trend', 0.85)
        left_weight = allocation.get('left', 0.15)
        position_cfg = self.config.get('risk_control', {}).get('position', {})
        trend_cap = position_cfg.get('max_single_sector', 0.10)
        left_cap = position_cfg.get('max_left_single', 0.02)

        target_weights = {}
        target_types = {}
        score_map = {}

        if 'S_trend' in trend_scores.columns:
            for sid, score in trend_scores.set_index('sector_industry_id')['S_trend'].items():
                score_map[str(sid)] = score
        if 'S_recovery' in left_scores.columns:
            for sid, score in left_scores.set_index('sector_industry_id')['S_recovery'].items():
                score_map[str(sid)] = score

        trend_score_map = {str(s): float(score_map.get(str(s), 0.0)) for s in trend_pool}
        left_score_map = {str(s): float(score_map.get(str(s), 0.0)) for s in left_pool}

        trend_weights, trend_reserve, trend_detail = self._allocate_with_caps(
            [str(s) for s in trend_pool], trend_weight, trend_cap, trend_score_map
        )
        left_weights, left_reserve, left_detail = self._allocate_with_caps(
            [str(s) for s in left_pool], left_weight, left_cap, left_score_map
        )

        for sid, w in trend_weights.items():
            if w > 0:
                target_weights[sid] = w
                target_types[sid] = 'trend'

        for sid, w in left_weights.items():
            if w > 0:
                target_weights[sid] = w
                target_types[sid] = 'left'

        cash_reserve = trend_reserve + left_reserve
        unallocated_details = [
            {'track': 'trend', **trend_detail},
            {'track': 'left', **left_detail}
        ]
        return target_weights, target_types, score_map, cash_reserve, unallocated_details

    def _allocate_with_caps(self, candidate_ids: List[str], target_total: float,
                            per_name_cap: float, scores: Dict[str, float]) -> Tuple[Dict[str, float], float, Dict]:
        """
        上限截断 + 评分再分配；无法分配部分以现金保留。
        """
        if target_total <= 0:
            return {}, 0.0, {
                'target_weight': target_total,
                'allocated_weight': 0.0,
                'remaining_weight': 0.0,
                'candidate_count': len(candidate_ids),
                'per_name_cap': per_name_cap,
                'reason': 'zero_target'
            }
        if per_name_cap <= 0:
            return {}, target_total, {
                'target_weight': target_total,
                'allocated_weight': 0.0,
                'remaining_weight': target_total,
                'candidate_count': len(candidate_ids),
                'per_name_cap': per_name_cap,
                'reason': 'invalid_cap'
            }
        if not candidate_ids:
            return {}, target_total, {
                'target_weight': target_total,
                'allocated_weight': 0.0,
                'remaining_weight': target_total,
                'candidate_count': 0,
                'per_name_cap': per_name_cap,
                'reason': 'no_candidates'
            }

        ids = list(dict.fromkeys(candidate_ids))
        n = len(ids)
        equal_w = target_total / n
        weights = {sid: min(equal_w, per_name_cap) for sid in ids}
        allocated = sum(weights.values())
        remaining = max(target_total - allocated, 0.0)

        if remaining <= 1e-12:
            allocated = sum(weights.values())
            return weights, 0.0, {
                'target_weight': target_total,
                'allocated_weight': allocated,
                'remaining_weight': 0.0,
                'candidate_count': len(ids),
                'per_name_cap': per_name_cap,
                'reason': 'fully_allocated'
            }

        for _ in range(10):
            headroom = {sid: max(per_name_cap - w, 0.0) for sid, w in weights.items()}
            open_ids = [sid for sid, h in headroom.items() if h > 1e-12]
            if not open_ids or remaining <= 1e-12:
                break

            raw_scores = {sid: max(float(scores.get(sid, 0.0)), 0.0) for sid in open_ids}
            score_sum = sum(raw_scores.values())
            if score_sum <= 1e-12:
                base_alloc = {sid: remaining / len(open_ids) for sid in open_ids}
            else:
                base_alloc = {sid: remaining * (raw_scores[sid] / score_sum) for sid in open_ids}

            used = 0.0
            for sid in open_ids:
                add_w = min(base_alloc[sid], headroom[sid])
                if add_w > 0:
                    weights[sid] += add_w
                    used += add_w

            if used <= 1e-12:
                break
            remaining = max(remaining - used, 0.0)

        allocated = sum(weights.values())
        reason = 'cap_binding_or_insufficient_candidates' if remaining > 1e-12 else 'fully_allocated'
        return weights, remaining, {
            'target_weight': target_total,
            'allocated_weight': allocated,
            'remaining_weight': remaining,
            'candidate_count': len(ids),
            'per_name_cap': per_name_cap,
            'reason': reason
        }

    def _allocate_incremental_with_caps(self, candidate_ids: List[str], target_total: float,
                                        per_name_cap: float, scores: Dict[str, float],
                                        existing_weights: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """
        在已有分配基础上继续分配，不突破单标的上限。
        返回新增分配和未分配剩余。
        """
        if target_total <= 0 or per_name_cap <= 0 or not candidate_ids:
            return {}, max(float(target_total), 0.0)

        ids = list(dict.fromkeys(candidate_ids))
        existing = {sid: max(float(existing_weights.get(sid, 0.0)), 0.0) for sid in ids}
        add_weights = {sid: 0.0 for sid in ids}
        remaining = max(float(target_total), 0.0)

        for _ in range(10):
            headroom = {
                sid: max(per_name_cap - existing[sid] - add_weights[sid], 0.0)
                for sid in ids
            }
            open_ids = [sid for sid, h in headroom.items() if h > 1e-12]
            if not open_ids or remaining <= 1e-12:
                break

            raw_scores = {sid: max(float(scores.get(sid, 0.0)), 0.0) for sid in open_ids}
            score_sum = sum(raw_scores.values())
            if score_sum <= 1e-12:
                base_alloc = {sid: remaining / len(open_ids) for sid in open_ids}
            else:
                base_alloc = {sid: remaining * (raw_scores[sid] / score_sum) for sid in open_ids}

            used = 0.0
            for sid in open_ids:
                add_w = min(base_alloc[sid], headroom[sid])
                if add_w > 0:
                    add_weights[sid] += add_w
                    used += add_w

            if used <= 1e-12:
                break
            remaining = max(remaining - used, 0.0)

        add_weights = {sid: w for sid, w in add_weights.items() if w > 1e-12}
        return add_weights, remaining

    def _execute_pending_rebalance(self, trade_date: pd.Timestamp, day_data: pd.DataFrame):
        today_orders = [o for o in self.pending_rebalance_orders if o['execute_date'] == trade_date]
        if not today_orders:
            return

        current_positions = self.position_manager.get_portfolio().get('positions', {})
        prices = day_data.set_index('sector_industry_id')['close'].to_dict()
        portfolio_value = self.current_nav * self.initial_capital

        for order in today_orders:
            trades = self.executor.execute_rebalance(
                current_positions=current_positions,
                target_weights=order['target_weights'],
                trade_date=trade_date,
                prices={str(k): v for k, v in prices.items()},
                portfolio_value=portfolio_value
            )

            sell_trades = [t for t in trades if t.get('action') == 'sell']
            buy_orders = [t for t in trades if t.get('action') == 'buy']

            executed_sell_trades = self._apply_sell_trades(sell_trades)
            total_cost = self.executor.calculate_total_cost(executed_sell_trades)
            if portfolio_value > 0:
                self.pending_cost_ratio += total_cost / portfolio_value

            self._queue_buy_orders(
                signal_date=trade_date,
                buy_orders=buy_orders,
                target_types=order['target_types'],
                score_map=order['score_map'],
                source='rebalance'
            )
            current_positions = self.position_manager.get_portfolio().get('positions', {})

        self.pending_rebalance_orders = [o for o in self.pending_rebalance_orders if o['execute_date'] != trade_date]

    def _apply_sell_trades(self, trades: List[Dict]):
        executed_trades = []
        for trade in trades:
            sid = str(trade['sector_industry_id'])
            if trade.get('action') != 'sell':
                continue
            price = trade['price']
            if sid not in self.position_manager.get_portfolio().get('positions', {}):
                continue
            self.position_manager.remove_position(sid, trade['date'], price, reason='rebalance')
            executed_trades.append(trade)
        return executed_trades

    def _queue_buy_orders(self, signal_date: pd.Timestamp, buy_orders: List[Dict],
                          target_types: Dict[str, str], score_map: Dict[str, float],
                          source: str):
        next_date = self.date_to_next.get(signal_date)
        if next_date is None or not buy_orders:
            return

        min_order_weight = float(self.rebalance_cfg.get('min_order_weight', 0.0005))
        for order in buy_orders:
            sid = str(order.get('sector_industry_id'))
            weight = float(order.get('weight', 0.0))
            if weight <= max(1e-12, min_order_weight):
                continue
            self.pending_buy_orders.append({
                'execute_date': next_date,
                'sector_industry_id': sid,
                'weight': weight,
                'type': target_types.get(sid, 'left'),
                'score': float(score_map.get(sid, 0.0)),
                'source': source,
                'signal_date': signal_date
            })

    def _execute_pending_buys(self, trade_date: pd.Timestamp, day_data: pd.DataFrame):
        today_orders = [o for o in self.pending_buy_orders if o['execute_date'] == trade_date]
        if not today_orders:
            return

        remaining_orders = [o for o in self.pending_buy_orders if o['execute_date'] != trade_date]
        prices = day_data.set_index('sector_industry_id')['open'].to_dict()
        portfolio_value = self.current_nav * self.initial_capital
        buy_trades = []
        min_order_weight = float(self.rebalance_cfg.get('min_order_weight', 0.0005))

        for order in today_orders:
            sid = order['sector_industry_id']
            try:
                sid_int = int(float(sid))
            except (TypeError, ValueError):
                continue
            if sid_int not in prices:
                continue
            weight = float(order['weight'])
            if weight <= max(1e-12, min_order_weight):
                continue
            ptype = order['type']
            if not self.position_manager.check_position_limits(sid, weight, ptype):
                continue

            trade = self.executor.execute_buy(sid, weight, trade_date, float(prices[sid_int]), portfolio_value)
            buy_trades.append(trade)
            try:
                self.position_manager.add_position(
                    sid, weight, trade_date, float(prices[sid_int]), ptype, float(order['score']), enforce_limits=True
                )
            except ValueError:
                continue

        total_buy_cost = self.executor.calculate_total_cost(buy_trades)
        if portfolio_value > 0:
            self.pending_cost_ratio += total_buy_cost / portfolio_value

        self.pending_buy_orders = remaining_orders

    def _run_daily_monitor(self, trade_date: pd.Timestamp, history_data: pd.DataFrame, day_data: pd.DataFrame):
        if self.position_manager.get_portfolio().get('positions', {}) == {}:
            return

        trend_snapshot = self.trend_calc.calculate_snapshot(history_data, trade_date, self.hierarchy)
        trend_snapshot['S_trend'] = self.trend_calc.calculate_score(trend_snapshot)
        left_snapshot = self.left_calc.calculate_snapshot(history_data, trade_date, self.hierarchy)
        left_snapshot['S_recovery'] = self.left_calc.calculate_score(left_snapshot)

        score_updates = {}
        if 'S_trend' in trend_snapshot.columns:
            for sid, score in trend_snapshot.set_index('sector_industry_id')['S_trend'].items():
                score_updates[str(sid)] = float(score)
        if 'S_recovery' in left_snapshot.columns:
            for sid, score in left_snapshot.set_index('sector_industry_id')['S_recovery'].items():
                score_updates[str(sid)] = float(score)
        self.position_manager.update_scores(score_updates)

        portfolio = self.position_manager.get_portfolio()
        sell_signals = self.sell_monitor.get_sell_signals(
            portfolio, day_data, trade_date, current_scores=score_updates
        )
        if not sell_signals:
            return

        portfolio_value = self.current_nav * self.initial_capital
        sell_trades = []
        sold_ids = set()
        for signal in sell_signals:
            sid = str(signal['sector_industry_id'])
            pos = portfolio['positions'].get(sid)
            if not pos:
                continue
            sold_ids.add(sid)
            trade = self.executor.execute_sell(
                sid, pos['weight'], trade_date, signal['current_price'], portfolio_value
            )
            sell_trades.append(trade)
            self.position_manager.remove_position(sid, trade_date, signal['current_price'], signal['signal_type'])

        total_cost = self.executor.calculate_total_cost(sell_trades)
        if portfolio_value > 0:
            self.pending_cost_ratio += total_cost / portfolio_value

        # 卖出后当日重新筛选：趋势 + 左侧双轨补充
        returns = day_data.set_index('sector_industry_id')['r_30']
        trend_selection = self.trend_selector.select(trend_snapshot, self.hierarchy, trade_date)
        trend_pool_raw = trend_selection['trend_pool']
        left_selection = self.left_selector.select(left_snapshot, trend_pool_raw, returns, trade_date)

        trend_pool = [str(s) for s in trend_pool_raw]
        left_pool = [str(s) for s in left_selection['left_pool']]

        current_portfolio = self.position_manager.get_portfolio()
        current_positions = current_portfolio.get('positions', {})
        current_holdings = set(current_positions.keys())

        position_cfg = self.config.get('risk_control', {}).get('position', {})
        trend_cap = position_cfg.get('max_single_sector', 0.10)
        left_cap = position_cfg.get('max_left_single', 0.02)
        allocation_cfg = self.backtest_config.get('allocation', {})
        trend_track_cap = float(allocation_cfg.get('trend', 0.85))
        left_track_cap = float(allocation_cfg.get('left', 0.15))
        tolerance_band = float(self.rebalance_cfg.get('tolerance_band', 0.03))
        residual_policy = str(self.rebalance_cfg.get('residual_policy', 'cash_only')).lower()
        min_order_weight = float(self.rebalance_cfg.get('min_order_weight', 0.0005))

        current_trend_weight = float(current_portfolio.get('total_trend_weight', 0.0))
        current_left_weight = float(current_portfolio.get('total_left_weight', 0.0))
        current_total_weight = float(current_portfolio.get('total_weight', 0.0))
        trend_need = max(trend_track_cap - current_trend_weight - tolerance_band, 0.0)
        left_need = max(left_track_cap - current_left_weight - tolerance_band, 0.0)
        total_need = trend_need + left_need
        if total_need <= 1e-12:
            return

        # 目标仓位缺口驱动：仅在卖出事件触发时，使用当前可用现金向目标靠拢
        available_cash = max(0.0, 1.0 - current_total_weight)
        buy_budget = min(available_cash, total_need)
        if buy_budget <= max(1e-12, min_order_weight):
            return

        trend_score_map = {}
        if 'S_trend' in trend_snapshot.columns:
            trend_score_map = {
                str(k): float(v)
                for k, v in trend_snapshot.set_index('sector_industry_id')['S_trend'].to_dict().items()
            }
        left_score_map = {}
        if 'S_recovery' in left_snapshot.columns:
            left_score_map = {
                str(k): float(v)
                for k, v in left_snapshot.set_index('sector_industry_id')['S_recovery'].to_dict().items()
            }

        # 禁止当日卖出的同一标的被当日回补
        trend_candidates = [sid for sid in trend_pool if sid not in current_holdings and sid not in sold_ids]
        left_candidates = [sid for sid in left_pool if sid not in current_holdings and sid not in sold_ids]

        if trend_candidates and left_candidates:
            trend_target = buy_budget * (trend_need / total_need)
            left_target = buy_budget - trend_target
        elif trend_candidates:
            trend_target = buy_budget
            left_target = 0.0
        elif left_candidates:
            trend_target = 0.0
            left_target = buy_budget
        else:
            trend_target = buy_budget * (trend_need / total_need)
            left_target = buy_budget - trend_target

        trend_alloc, trend_reserve, _ = self._allocate_with_caps(
            trend_candidates, trend_target, trend_cap, trend_score_map
        )
        left_alloc, left_reserve, _ = self._allocate_with_caps(
            left_candidates, left_target, left_cap, left_score_map
        )

        if residual_policy == 'cross_track_first' and left_reserve > 1e-12 and trend_candidates:
            trend_cap_room = max(0.0, trend_track_cap - current_trend_weight - sum(trend_alloc.values()))
            cross_budget = min(left_reserve, trend_cap_room)
            if cross_budget > 1e-12:
                extra_trend_alloc, extra_reserve = self._allocate_incremental_with_caps(
                    trend_candidates, cross_budget, trend_cap, trend_score_map, trend_alloc
                )
                if extra_trend_alloc:
                    for sid, w in extra_trend_alloc.items():
                        trend_alloc[sid] = trend_alloc.get(sid, 0.0) + w
                    moved = sum(extra_trend_alloc.values())
                    left_reserve = max(left_reserve - moved, 0.0)
                trend_reserve += extra_reserve

        trend_allocated = sum(trend_alloc.values())
        left_allocated = sum(left_alloc.values())
        trend_remaining = max(trend_target - trend_allocated, 0.0)
        left_remaining = max(left_target - left_allocated, 0.0)

        def _detail(track: str, target_weight: float, allocated_weight: float,
                    remaining_weight: float, candidate_count: int, per_name_cap: float) -> Dict:
            if target_weight <= 1e-12:
                reason = 'zero_target'
            elif candidate_count <= 0:
                reason = 'no_candidates'
            elif remaining_weight > 1e-12:
                reason = 'cap_binding_or_insufficient_candidates'
            else:
                reason = 'fully_allocated'
            return {
                'track': track,
                'target_weight': target_weight,
                'allocated_weight': allocated_weight,
                'remaining_weight': remaining_weight,
                'candidate_count': candidate_count,
                'per_name_cap': per_name_cap,
                'reason': reason
            }

        trend_detail = _detail(
            'trend', trend_target, trend_allocated, trend_remaining, len(set(trend_candidates)), trend_cap
        )
        left_detail = _detail(
            'left', left_target, left_allocated, left_remaining, len(set(left_candidates)), left_cap
        )

        for detail in (
            trend_detail,
            left_detail,
        ):
            if detail.get('remaining_weight', 0.0) > 0:
                self.unallocated_records.append({
                    'trade_date': trade_date,
                    'source': 'sell_replenish',
                    **detail
                })

        buy_orders = []
        for sid, weight in trend_alloc.items():
            if weight <= max(1e-12, min_order_weight):
                continue
            if not self.position_manager.check_position_limits(sid, weight, 'trend'):
                continue
            score = float(trend_score_map.get(sid, 0.0))
            buy_orders.append({
                'sector_industry_id': sid,
                'weight': weight,
                'type': 'trend',
                'score': score
            })

        for sid, weight in left_alloc.items():
            if weight <= max(1e-12, min_order_weight):
                continue
            if not self.position_manager.check_position_limits(sid, weight, 'left'):
                continue
            score = float(left_score_map.get(sid, 0.0))
            buy_orders.append({
                'sector_industry_id': sid,
                'weight': weight,
                'type': 'left',
                'score': score
            })

        # 买单统一挂到下一交易日开盘执行
        next_date = self.date_to_next.get(trade_date)
        if next_date is not None:
            for order in buy_orders:
                self.pending_buy_orders.append({
                    'execute_date': next_date,
                    'sector_industry_id': str(order['sector_industry_id']),
                    'weight': float(order['weight']),
                    'type': order['type'],
                    'score': float(order['score']),
                    'source': 'sell_replenish',
                    'signal_date': trade_date
                })

    def _calculate_daily_nav(self, trade_date: pd.Timestamp, day_data: pd.DataFrame,
                             positions_snapshot: Optional[Dict[str, Dict]] = None):
        positions = positions_snapshot or self.position_manager.get_portfolio().get('positions', {})
        daily_return = 0.0

        if positions:
            for sid, position in positions.items():
                try:
                    sid_int = int(float(sid))
                except (TypeError, ValueError):
                    continue
                sector_data = day_data[day_data['sector_industry_id'] == sid_int]
                if sector_data.empty:
                    continue
                r1 = sector_data['r_1'].iloc[0]
                if pd.notna(r1):
                    daily_return += r1 * position['weight']

        daily_return -= self.pending_cost_ratio
        self.pending_cost_ratio = 0.0

        self.current_nav *= (1 + daily_return)
        self.equity_curve.append({
            'date': trade_date,
            'nav': self.current_nav,
            'daily_return': daily_return,
            'portfolio_value': self.current_nav * self.initial_capital
        })
        self.daily_returns.append({'date': trade_date, 'return': daily_return})
        total_weight = sum(float(pos.get('weight', 0.0)) for pos in positions.values())
        cash_weight = max(0.0, 1.0 - total_weight)
        self.daily_cash_weights.append({'date': trade_date, 'cash_weight': cash_weight})

    def _build_results(self) -> Dict:
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        returns_df = pd.DataFrame(self.daily_returns).set_index('date') if self.daily_returns else pd.DataFrame(columns=['return'])
        trade_history = self.position_manager.get_trade_history()

        return {
            'equity_curve': equity_df['nav'],
            'portfolio_value': equity_df['portfolio_value'],
            'daily_returns': returns_df['return'] if not returns_df.empty else pd.Series(dtype=float),
            'transactions': trade_history,
            'cash_weights': pd.DataFrame(self.daily_cash_weights),
            'unallocated': pd.DataFrame(self.unallocated_records),
            'regime_states': pd.DataFrame(self.regime_records),
            'final_nav': self.current_nav,
            'total_return': self.current_nav - 1
        }
