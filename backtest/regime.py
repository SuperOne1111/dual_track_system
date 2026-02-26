"""
Market regime manager for state-aware parameter activation.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class MarketRegimeManager:
    """Classify market regimes and produce effective config by date."""

    LOOKBACK_BY_SEGMENTATION = {
        'quarter': 63,
        'quarterly': 63,
        'half_year': 126,
        'semiannual': 126,
        'year': 252,
        'annual': 252,
    }

    DEFAULT_FLAGS = {
        'left_track_active': True,
        'score_exit_active': True,
        'aggressive_trend_expand_active': False,
        'strict_left_filter_active': False,
        'cross_track_first_active': True,
    }

    def __init__(self, config: Dict):
        self.base_config = copy.deepcopy(config)
        self.regime_config = copy.deepcopy(config.get('regime', {}))
        self.enabled = bool(self.regime_config.get('enabled', False))

        segmentation = str(self.regime_config.get('segmentation', 'quarter')).lower()
        self.segmentation = segmentation
        default_lookback = self.LOOKBACK_BY_SEGMENTATION.get(segmentation, 63)
        self.lookback_days = int(self.regime_config.get('lookback_days', default_lookback))

        self.update_frequency = str(self.regime_config.get('update_frequency', 'monthly')).lower()
        self.min_history_days = int(self.regime_config.get('min_history_days', self.lookback_days))
        self.min_regime_days = int(self.regime_config.get('min_regime_days', 20))

        thresholds = self.regime_config.get('thresholds', {})
        self.bull_return = float(thresholds.get('bull_return', 0.05))
        self.bear_return = float(thresholds.get('bear_return', -0.05))
        self.high_vol_quantile = float(thresholds.get('high_vol_quantile', 0.70))
        self.breadth_bull = float(thresholds.get('breadth_bull', 0.55))
        self.breadth_bear = float(thresholds.get('breadth_bear', 0.45))

        self.flag_overrides = self.regime_config.get('flag_overrides', {})
        self.states_config = self.regime_config.get('states', {})

        self.default_flags = copy.deepcopy(self.DEFAULT_FLAGS)
        self.default_flags.update(self.regime_config.get('active_flags_defaults', {}))

        self.regime_by_date: Dict[pd.Timestamp, str] = {}
        self.features_by_date: pd.DataFrame = pd.DataFrame()

    def prepare(self, data: pd.DataFrame, trade_dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, str]:
        trade_dates = [pd.to_datetime(d) for d in trade_dates]
        if not trade_dates:
            self.regime_by_date = {}
            self.features_by_date = pd.DataFrame()
            return self.regime_by_date

        if not self.enabled:
            self.regime_by_date = {d: 'disabled' for d in trade_dates}
            self.features_by_date = pd.DataFrame(index=pd.Index(trade_dates, name='trade_date'))
            return self.regime_by_date

        features = self._build_market_features(data, trade_dates)
        regimes = self._classify_regimes(features, trade_dates)

        self.features_by_date = features
        self.regime_by_date = regimes
        return regimes

    def get_effective_config(self, trade_date: pd.Timestamp) -> Tuple[Dict, str, Dict]:
        base = copy.deepcopy(self.base_config)
        t = pd.to_datetime(trade_date)

        if not self.enabled:
            return base, 'disabled', copy.deepcopy(self.default_flags)

        state = self.regime_by_date.get(t, 'range')
        state_cfg = self.states_config.get(state, {})

        flags = copy.deepcopy(self.default_flags)
        flags.update(state_cfg.get('active_flags', {}))

        if bool(state_cfg.get('enabled', True)):
            overrides = state_cfg.get('overrides', {})
            self._deep_merge(base, overrides)

        self._apply_activation_flags(base, flags)
        return base, state, flags

    def _build_market_features(self, data: pd.DataFrame, trade_dates: List[pd.Timestamp]) -> pd.DataFrame:
        df = data.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        market_universe = df[df.get('level', 4) == 1].copy()
        if market_universe.empty:
            market_universe = df.copy()

        if 'total_market_cap' in market_universe.columns and market_universe['total_market_cap'].notna().any():
            day_total = market_universe.groupby('trade_date')['total_market_cap'].transform('sum').replace(0, np.nan)
            market_universe['_w'] = (market_universe['total_market_cap'] / day_total).fillna(0)
            market_return = (market_universe['r_1'] * market_universe['_w']).groupby(market_universe['trade_date']).sum()
        else:
            market_return = market_universe.groupby('trade_date')['r_1'].mean()

        breadth_universe = df[df.get('level', 4) == 4].copy()
        if breadth_universe.empty:
            breadth_universe = df.copy()

        if {'close', 'ma_60'}.issubset(breadth_universe.columns):
            breadth_universe['_above_ma60'] = (breadth_universe['close'] > breadth_universe['ma_60']).astype(float)
            breadth = breadth_universe.groupby('trade_date')['_above_ma60'].mean()
        elif 'r_30' in breadth_universe.columns:
            breadth = (breadth_universe['r_30'] > 0).groupby(breadth_universe['trade_date']).mean()
        else:
            breadth = pd.Series(0.5, index=market_return.index)

        feature = pd.DataFrame(index=pd.Index(sorted(set(trade_dates)), name='trade_date'))
        feature['market_return'] = market_return.reindex(feature.index).fillna(0.0)
        feature['breadth'] = breadth.reindex(feature.index).fillna(method='ffill').fillna(0.5)

        ret_series = 1.0 + feature['market_return']
        feature['lookback_return'] = (
            ret_series.rolling(window=max(self.lookback_days, 2), min_periods=max(self.min_history_days, 2))
            .apply(np.prod, raw=True) - 1.0
        )

        feature['vol_20'] = feature['market_return'].rolling(window=20, min_periods=20).std() * np.sqrt(252)
        feature['vol_cutoff'] = (
            feature['vol_20']
            .expanding(min_periods=max(20, min(self.min_history_days, 60)))
            .quantile(self.high_vol_quantile)
        )

        return feature

    def _classify_regimes(self, features: pd.DataFrame, trade_dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, str]:
        regimes: Dict[pd.Timestamp, str] = {}
        prev_state = 'range'
        last_update_date = None
        last_switch_date = None

        for t in trade_dates:
            t = pd.to_datetime(t)
            row = features.loc[t] if t in features.index else None

            if row is None:
                regimes[t] = prev_state
                continue

            if not self._is_update_day(t, last_update_date):
                regimes[t] = prev_state
                continue

            candidate = self._classify_single_row(row)
            if candidate != prev_state and last_switch_date is not None:
                if (t - last_switch_date).days < self.min_regime_days:
                    candidate = prev_state

            if candidate != prev_state:
                last_switch_date = t
            prev_state = candidate
            last_update_date = t
            regimes[t] = prev_state

        return regimes

    def _classify_single_row(self, row: pd.Series) -> str:
        ret = row.get('lookback_return', np.nan)
        vol = row.get('vol_20', np.nan)
        vol_cut = row.get('vol_cutoff', np.nan)
        breadth = row.get('breadth', np.nan)

        if np.isnan(ret) or np.isnan(vol) or np.isnan(vol_cut) or np.isnan(breadth):
            return 'range'

        if ret >= self.bull_return and breadth >= self.breadth_bull and vol <= vol_cut:
            return 'bull'

        bearish = ret <= self.bear_return
        weak_breadth = breadth <= self.breadth_bear
        high_vol = vol >= vol_cut
        if bearish and (weak_breadth or high_vol):
            return 'bear_highvol'
        if high_vol and weak_breadth:
            return 'bear_highvol'

        return 'range'

    def _is_update_day(self, current: pd.Timestamp, previous: pd.Timestamp) -> bool:
        if previous is None:
            return True

        freq = self.update_frequency
        if freq == 'daily':
            return True
        if freq == 'weekly':
            return current.isocalendar().week != previous.isocalendar().week or current.year != previous.year
        if freq == 'monthly':
            return current.to_period('M') != previous.to_period('M')
        if freq in ('quarterly', 'quarter'):
            return current.to_period('Q') != previous.to_period('Q')
        if freq in ('semiannual', 'half_year'):
            return (current.year != previous.year) or ((current.month - 1) // 6 != (previous.month - 1) // 6)
        if freq in ('annual', 'yearly', 'year'):
            return current.year != previous.year
        return current.to_period('M') != previous.to_period('M')

    def _apply_activation_flags(self, config: Dict, flags: Dict):
        left_active = bool(flags.get('left_track_active', True))
        score_exit_active = bool(flags.get('score_exit_active', True))
        strict_left_active = bool(flags.get('strict_left_filter_active', False))
        trend_expand_active = bool(flags.get('aggressive_trend_expand_active', False))
        cross_track_active = bool(flags.get('cross_track_first_active', True))

        if not left_active:
            allocation = config.setdefault('backtest', {}).setdefault('allocation', {})
            left_alloc = float(allocation.get('left', 0.0))
            trend_alloc = float(allocation.get('trend', 0.0))
            allocation['left'] = 0.0
            allocation['trend'] = min(max(trend_alloc + left_alloc, trend_alloc), 1.0)

            pos = config.setdefault('risk_control', {}).setdefault('position', {})
            pos['max_left_total'] = 0.0
            pos['max_left_single'] = 0.0
            pos['max_trend_total'] = max(float(pos.get('max_trend_total', 0.0)), float(allocation['trend']))

            left_sel = config.setdefault('left', {}).setdefault('selection', {})
            left_sel['max_selections'] = 0

        if not score_exit_active:
            stop_loss = config.setdefault('risk_control', {}).setdefault('stop_loss', {})
            stop_loss['score_exit'] = 0

        rebalance = config.setdefault('backtest', {}).setdefault('rebalance', {})
        rebalance['residual_policy'] = 'cross_track_first' if cross_track_active else 'cash_only'

        if strict_left_active:
            delta_cfg = self.flag_overrides.get('strict_left_filter_active', {})
            score_delta = int(delta_cfg.get('score_threshold_delta', 5))
            ret_delta = float(delta_cfg.get('return_threshold_delta', 0.02))
            max_sel_delta = int(delta_cfg.get('max_selections_delta', -2))

            left_sel = config.setdefault('left', {}).setdefault('selection', {})
            left_sel['score_threshold'] = int(left_sel.get('score_threshold', 70)) + score_delta
            left_sel['return_threshold'] = min(float(left_sel.get('return_threshold', -0.10)) + ret_delta, -0.01)
            left_sel['max_selections'] = max(int(left_sel.get('max_selections', 3)) + max_sel_delta, 1)

            supp = config.setdefault('supplement', {}).setdefault('candidate_pool', {})
            supp['score_threshold'] = int(supp.get('score_threshold', 65)) + max(1, score_delta // 2)

        if trend_expand_active:
            delta_cfg = self.flag_overrides.get('aggressive_trend_expand_active', {})
            l2_delta = int(delta_cfg.get('level2_count_delta', 2))
            l4_delta = int(delta_cfg.get('level4_max_total_delta', 4))
            alloc_delta = float(delta_cfg.get('trend_allocation_delta', 0.08))

            trend_sel = config.setdefault('selection', {}).setdefault('trend', {})
            trend_sel.setdefault('level2', {})
            trend_sel.setdefault('level4', {})
            trend_sel['level2']['count'] = max(1, int(trend_sel['level2'].get('count', 8)) + l2_delta)
            trend_sel['level4']['max_total'] = max(1, int(trend_sel['level4'].get('max_total', 16)) + l4_delta)

            allocation = config.setdefault('backtest', {}).setdefault('allocation', {})
            trend_alloc = float(allocation.get('trend', 0.6))
            left_alloc = float(allocation.get('left', 0.4))
            trend_alloc_new = min(max(trend_alloc + alloc_delta, 0.0), 0.95)
            left_alloc_new = max(min(1.0 - trend_alloc_new, left_alloc), 0.0)
            allocation['trend'] = trend_alloc_new
            allocation['left'] = left_alloc_new

            pos = config.setdefault('risk_control', {}).setdefault('position', {})
            pos['max_trend_total'] = max(float(pos.get('max_trend_total', trend_alloc_new)), trend_alloc_new)
            pos['max_left_total'] = min(float(pos.get('max_left_total', left_alloc_new)), left_alloc_new)

    @staticmethod
    def _deep_merge(target: Dict, source: Dict):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                MarketRegimeManager._deep_merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
