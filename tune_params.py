#!/usr/bin/env python3
"""Parameter optimizer with regime-aware activation flags."""

import argparse
import copy
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from utils import load_config, build_hierarchy_mapping
from data import DataLoader, LifecycleBuilder, DataPreprocessor
from backtest import BacktestEngine, BenchmarkBuilder
from evaluation import MetricsCalculator


def _deep_merge(target: Dict, source: Dict):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _ensure_regime_template(config: Dict, segmentation: str, update_frequency: str):
    regime = config.setdefault('regime', {})
    regime['enabled'] = True
    regime['segmentation'] = segmentation
    regime['update_frequency'] = update_frequency

    lookback = {'quarter': 63, 'half_year': 126, 'year': 252}[segmentation]
    regime.setdefault('lookback_days', lookback)
    regime.setdefault('min_history_days', max(40, lookback // 2))
    regime.setdefault('min_regime_days', 20)

    thresholds = regime.setdefault('thresholds', {})
    thresholds.setdefault('bull_return', 0.05)
    thresholds.setdefault('bear_return', -0.05)
    thresholds.setdefault('high_vol_quantile', 0.70)
    thresholds.setdefault('breadth_bull', 0.55)
    thresholds.setdefault('breadth_bear', 0.45)

    regime.setdefault('active_flags_defaults', {
        'left_track_active': True,
        'score_exit_active': True,
        'aggressive_trend_expand_active': False,
        'strict_left_filter_active': False,
        'cross_track_first_active': True,
    })

    regime.setdefault('flag_overrides', {
        'strict_left_filter_active': {
            'score_threshold_delta': 5,
            'return_threshold_delta': 0.02,
            'max_selections_delta': -2,
        },
        'aggressive_trend_expand_active': {
            'level2_count_delta': 2,
            'level4_max_total_delta': 4,
            'trend_allocation_delta': 0.08,
        },
    })

    states = regime.setdefault('states', {})
    for state in ('bull', 'range', 'bear_highvol'):
        st = states.setdefault(state, {})
        st.setdefault('enabled', True)
        st.setdefault('active_flags', {})
        st.setdefault('overrides', {})


def _sample_state_overrides(state: str, rng: random.Random) -> Dict:
    if state == 'bull':
        trend_alloc = rng.uniform(0.72, 0.90)
        l2_count = rng.randint(10, 16)
        l4_max = rng.randint(18, 28)
        atr = rng.uniform(2.8, 3.8)
        left_score = rng.randint(82, 94)
        left_return = -rng.uniform(0.04, 0.10)
    elif state == 'bear_highvol':
        trend_alloc = rng.uniform(0.45, 0.70)
        l2_count = rng.randint(6, 12)
        l4_max = rng.randint(10, 20)
        atr = rng.uniform(2.0, 3.2)
        left_score = rng.randint(86, 96)
        left_return = -rng.uniform(0.03, 0.08)
    else:
        trend_alloc = rng.uniform(0.58, 0.78)
        l2_count = rng.randint(8, 14)
        l4_max = rng.randint(14, 24)
        atr = rng.uniform(2.4, 3.4)
        left_score = rng.randint(84, 94)
        left_return = -rng.uniform(0.04, 0.09)

    left_alloc = max(0.0, 1.0 - trend_alloc)
    max_left_single = rng.uniform(0.03, 0.06)
    min_order_weight = rng.uniform(0.002, 0.01)
    supp_score = max(60, left_score - rng.randint(6, 12))
    supp_max = rng.randint(4, 10)

    overrides = {
        'backtest': {
            'allocation': {
                'trend': round(trend_alloc, 4),
                'left': round(left_alloc, 4),
            },
            'rebalance': {
                'min_order_weight': round(min_order_weight, 4),
            },
        },
        'risk_control': {
            'position': {
                'max_trend_total': round(trend_alloc, 4),
                'max_left_total': round(left_alloc, 4),
                'max_left_single': round(max_left_single, 4),
            },
            'stop_loss': {
                'atr_multiplier': round(atr, 3),
            },
        },
        'selection': {
            'trend': {
                'level2': {'count': l2_count},
                'level4': {'max_total': l4_max},
            }
        },
        'left': {
            'selection': {
                'score_threshold': int(left_score),
                'return_threshold': round(left_return, 4),
                'max_selections': rng.randint(6, 14),
            }
        },
        'supplement': {
            'candidate_pool': {
                'score_threshold': int(supp_score),
                'max_selections': int(supp_max),
            }
        },
    }
    return overrides


def _sample_regime_candidate(base_config: Dict, rng: random.Random, segmentation: str, update_frequency: str) -> Dict:
    cfg = copy.deepcopy(base_config)
    _ensure_regime_template(cfg, segmentation, update_frequency)

    states = cfg['regime']['states']
    for state in ('bull', 'range', 'bear_highvol'):
        state_cfg = states[state]
        state_cfg['enabled'] = rng.random() > 0.10

        flags = {
            'left_track_active': rng.random() > 0.18,
            'score_exit_active': rng.random() > 0.45,
            'aggressive_trend_expand_active': rng.random() > 0.55,
            'strict_left_filter_active': rng.random() > 0.50,
            'cross_track_first_active': rng.random() > 0.20,
        }

        if state == 'bull':
            flags['aggressive_trend_expand_active'] = rng.random() > 0.35
            flags['strict_left_filter_active'] = rng.random() > 0.65
        elif state == 'bear_highvol':
            flags['strict_left_filter_active'] = rng.random() > 0.25
            flags['left_track_active'] = rng.random() > 0.30

        state_cfg['active_flags'] = flags
        state_cfg['overrides'] = _sample_state_overrides(state, rng)

    cfg['backtest']['rebalance']['residual_policy'] = 'cross_track_first'
    return cfg


def _build_eval_windows(trade_dates: List[pd.Timestamp], window_days: int, step_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not trade_dates:
        return []

    dates = sorted(pd.to_datetime(trade_dates))
    if len(dates) <= window_days:
        return [(dates[0], dates[-1])]

    windows = []
    i = 0
    while i + window_days <= len(dates):
        windows.append((dates[i], dates[i + window_days - 1]))
        i += step_days

    if not windows:
        windows = [(dates[0], dates[-1])]
    return windows


def _evaluate_once(config: Dict,
                   processed_data: pd.DataFrame,
                   hierarchy: Dict,
                   lifecycle_table: pd.DataFrame,
                   window_start: pd.Timestamp,
                   window_end: pd.Timestamp,
                   objective: str,
                   min_win_rate: float,
                   max_drawdown: float,
                   max_cash_mean: float) -> Dict:
    cfg = copy.deepcopy(config)
    cfg.setdefault('backtest', {})
    cfg['backtest']['start_date'] = str(pd.to_datetime(window_start).date())
    cfg['backtest']['end_date'] = str(pd.to_datetime(window_end).date())

    engine = BacktestEngine(cfg)
    engine.initialize(processed_data, hierarchy, lifecycle_table)
    results = engine.run_backtest()

    level1 = processed_data[
        (processed_data['level'] == 1) &
        (processed_data['trade_date'] >= pd.to_datetime(window_start)) &
        (processed_data['trade_date'] <= pd.to_datetime(window_end))
    ].copy()

    benchmark_returns = None
    if not level1.empty:
        bb = BenchmarkBuilder()
        bb.build_benchmark(level1)
        benchmark_returns = bb.get_benchmark_returns()

    metrics = MetricsCalculator().calculate_all_metrics(results, benchmark_returns)

    general = metrics.get('general', {})
    relative = metrics.get('relative', {})

    if objective == 'alpha':
        obj = float(relative.get('alpha', -1e9))
    elif objective == 'information_ratio':
        obj = float(relative.get('information_ratio', -1e9))
    elif objective == 'annualized_return':
        obj = float(general.get('annualized_return', -1e9))
    else:
        obj = float(general.get('sharpe_ratio', -1e9))

    win_rate = float(general.get('win_rate', 0.0))
    mdd = float(general.get('max_drawdown', -1.0))

    cash_df = results.get('cash_weights', pd.DataFrame())
    if isinstance(cash_df, pd.DataFrame) and not cash_df.empty and 'cash_weight' in cash_df.columns:
        cash_mean = float(cash_df['cash_weight'].mean())
    else:
        cash_mean = 1.0

    penalty = 0.0
    if win_rate < min_win_rate:
        penalty += (min_win_rate - win_rate) * 2.0
    if mdd < max_drawdown:
        penalty += (max_drawdown - mdd) * 2.5
    if cash_mean > max_cash_mean:
        penalty += (cash_mean - max_cash_mean) * 1.2

    score = obj - penalty

    return {
        'score': score,
        'objective_raw': obj,
        'penalty': penalty,
        'annualized_return': float(general.get('annualized_return', 0.0)),
        'sharpe_ratio': float(general.get('sharpe_ratio', 0.0)),
        'win_rate': win_rate,
        'max_drawdown': mdd,
        'alpha': float(relative.get('alpha', 0.0)),
        'information_ratio': float(relative.get('information_ratio', 0.0)),
        'cash_mean': cash_mean,
    }


def _aggregate_window_metrics(metrics_list: List[Dict]) -> Dict:
    if not metrics_list:
        return {'score': -1e9}

    keys = [
        'score', 'objective_raw', 'penalty', 'annualized_return', 'sharpe_ratio',
        'win_rate', 'max_drawdown', 'alpha', 'information_ratio', 'cash_mean'
    ]
    agg = {}
    for key in keys:
        vals = [m[key] for m in metrics_list]
        agg[key] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))

    agg['score'] = agg['score'] - 0.20 * agg['score_std']
    agg['windows'] = len(metrics_list)
    return agg


def parse_args():
    parser = argparse.ArgumentParser(description='Regime-aware parameter optimizer')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--objective', type=str, default='alpha',
                        choices=['alpha', 'information_ratio', 'sharpe_ratio', 'annualized_return'])
    parser.add_argument('--segmentation', type=str, default='quarter',
                        choices=['quarter', 'half_year', 'year'])
    parser.add_argument('--update-frequency', type=str, default='monthly',
                        choices=['monthly', 'quarterly', 'semiannual', 'annual'])
    parser.add_argument('--window-days', type=int, default=126)
    parser.add_argument('--step-days', type=int, default=63)
    parser.add_argument('--min-win-rate', type=float, default=0.48)
    parser.add_argument('--max-drawdown', type=float, default=-0.22)
    parser.add_argument('--max-cash-mean', type=float, default=0.40)
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--output-config', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    if not os.getenv('PG_CONN_STRING'):
        raise RuntimeError('PG_CONN_STRING is required for parameter optimization')

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    base_config = load_config(args.config)
    _ensure_regime_template(base_config, args.segmentation, args.update_frequency)

    data_cfg = base_config.get('data', {})
    start_date = data_cfg.get('start_date')
    end_date = data_cfg.get('end_date')

    print('[1/4] Loading data...')
    loader = DataLoader()
    raw = loader.load_all_data(start_date, end_date)

    sector_metadata = raw['sector_metadata']
    sector_hierarchy = raw['sector_hierarchy']
    daily_data = raw['daily_data']

    print('[2/4] Preprocessing...')
    preprocessor = DataPreprocessor(daily_data)
    processed_data = preprocessor.preprocess(
        coverage_threshold=data_cfg.get('validation', {}).get('coverage_threshold', 0.90),
        missing_threshold=data_cfg.get('validation', {}).get('missing_threshold', 0.05),
    )
    processed_data = processed_data.merge(
        sector_metadata[['id', 'name', 'level']],
        left_on='sector_industry_id',
        right_on='id',
        how='left'
    )

    lifecycle_table = LifecycleBuilder(daily_data).build_lifecycle_table()
    hierarchy = build_hierarchy_mapping(sector_hierarchy, sector_metadata)

    bt_cfg = base_config.get('backtest', {})
    bt_start = pd.to_datetime(bt_cfg.get('start_date'))
    bt_end = pd.to_datetime(bt_cfg.get('end_date'))
    trade_dates = (
        processed_data[(processed_data['trade_date'] >= bt_start) & (processed_data['trade_date'] <= bt_end)]['trade_date']
        .drop_duplicates().sort_values().tolist()
    )

    windows = _build_eval_windows(trade_dates, args.window_days, args.step_days)
    print(f'[3/4] Optimization windows: {len(windows)}')

    records = []
    best_score = -1e18
    best_cfg = None
    best_summary = None

    for i in range(args.trials):
        candidate = _sample_regime_candidate(base_config, rng, args.segmentation, args.update_frequency)

        per_window = []
        failed = False
        for ws, we in windows:
            try:
                m = _evaluate_once(
                    candidate, processed_data, hierarchy, lifecycle_table,
                    ws, we, args.objective,
                    args.min_win_rate, args.max_drawdown, args.max_cash_mean
                )
                per_window.append(m)
            except Exception as e:
                failed = True
                per_window = []
                print(f'  trial={i+1} window={ws.date()}~{we.date()} failed: {e}')
                break

        summary = _aggregate_window_metrics(per_window)
        score = summary.get('score', -1e9)
        if failed:
            score = -1e9
            summary = {'score': score, 'windows': 0}

        rec = {
            'trial': i + 1,
            'score': score,
            'windows': summary.get('windows', 0),
            'objective_raw': summary.get('objective_raw', np.nan),
            'penalty': summary.get('penalty', np.nan),
            'annualized_return': summary.get('annualized_return', np.nan),
            'sharpe_ratio': summary.get('sharpe_ratio', np.nan),
            'win_rate': summary.get('win_rate', np.nan),
            'max_drawdown': summary.get('max_drawdown', np.nan),
            'alpha': summary.get('alpha', np.nan),
            'information_ratio': summary.get('information_ratio', np.nan),
            'cash_mean': summary.get('cash_mean', np.nan),
        }
        records.append(rec)

        print(
            f"trial {i+1:03d}/{args.trials} score={score:.6f} "
            f"alpha={rec['alpha']:.4f} ir={rec['information_ratio']:.4f} "
            f"win={rec['win_rate']:.4f} mdd={rec['max_drawdown']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_cfg = copy.deepcopy(candidate)
            best_summary = summary

    if best_cfg is None:
        raise RuntimeError('No valid trial was produced')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    trials_path = out_dir / f'tuning_trials_{ts}.csv'
    pd.DataFrame(records).sort_values('score', ascending=False).to_csv(trials_path, index=False, encoding='utf-8-sig')

    if args.output_config:
        best_cfg_path = Path(args.output_config)
    else:
        best_cfg_path = out_dir / f'config_tuned_{ts}.yaml'

    with open(best_cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False, allow_unicode=True)

    summary_path = out_dir / f'tuning_best_summary_{ts}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)

    print('[4/4] Done')
    print(f'best_score={best_score:.6f}')
    print(f'best_config={best_cfg_path}')
    print(f'trials={trials_path}')
    print(f'summary={summary_path}')


if __name__ == '__main__':
    main()
