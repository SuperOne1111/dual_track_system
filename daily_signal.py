#!/usr/bin/env python3
"""
Generate next-day trade orders without running backtest.

This script reuses existing project logic:
- monthly rebalance selection (trend + left pools)
- daily sell monitor
- supplement engine

Outputs a CSV order list for T+1 execution.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, build_hierarchy_mapping
from data import DataLoader, DataPreprocessor
from features import TrendFeatureCalculator, LeftFeatureCalculator
from selection import TrendSelector, LeftSelector
from rebalance import SellMonitor, SupplementEngine, PositionManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily trade orders (no backtest)")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument(
        "--holdings",
        default="current_holdings.csv",
        help="Current holdings CSV path (optional). If missing, treated as empty holdings.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "rebalance", "monitor"],
        default="auto",
        help="auto: month-end => rebalance else monitor",
    )
    parser.add_argument(
        "--signal-date",
        default=None,
        help="Signal date YYYY-MM-DD. Default: latest available trade date in loaded data",
    )
    parser.add_argument("--start-date", default=None, help="Override data start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Override data end date YYYY-MM-DD")
    return parser.parse_args()


def _normalize_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _allocate_with_caps(candidate_ids: List[str], target_total: float,
                        per_name_cap: float, scores: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    if target_total <= 0 or per_name_cap <= 0 or not candidate_ids:
        return {}, max(target_total, 0.0)

    ids = list(dict.fromkeys(candidate_ids))
    equal_w = target_total / len(ids)
    weights = {sid: min(equal_w, per_name_cap) for sid in ids}
    remaining = max(target_total - sum(weights.values()), 0.0)

    for _ in range(10):
        if remaining <= 1e-12:
            break
        headroom = {sid: max(per_name_cap - w, 0.0) for sid, w in weights.items()}
        open_ids = [sid for sid, h in headroom.items() if h > 1e-12]
        if not open_ids:
            break

        raw_scores = {sid: max(_safe_float(scores.get(sid), 0.0), 0.0) for sid in open_ids}
        ssum = sum(raw_scores.values())
        if ssum <= 1e-12:
            alloc = {sid: remaining / len(open_ids) for sid in open_ids}
        else:
            alloc = {sid: remaining * (raw_scores[sid] / ssum) for sid in open_ids}

        used = 0.0
        for sid in open_ids:
            add = min(alloc[sid], headroom[sid])
            if add > 0:
                weights[sid] += add
                used += add
        if used <= 1e-12:
            break
        remaining = max(remaining - used, 0.0)

    return weights, remaining


def load_market_data(config: dict, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    data_cfg = config.get("data", {})
    validation_cfg = data_cfg.get("validation", {})

    start_date = args.start_date or data_cfg.get("start_date") or config.get("backtest", {}).get("start_date")
    if args.end_date:
        end_date = args.end_date
    elif args.signal_date:
        end_date = args.signal_date
    else:
        end_date = datetime.today().strftime("%Y-%m-%d")

    if not start_date:
        raise ValueError("Missing start date. Set data.start_date in config or pass --start-date.")

    loader = DataLoader()
    raw = loader.load_all_data(start_date, end_date)

    metadata = raw["sector_metadata"]
    hierarchy = raw["sector_hierarchy"]
    daily_data = raw["daily_data"]

    preprocessor = DataPreprocessor(daily_data)
    processed = preprocessor.preprocess(
        coverage_threshold=validation_cfg.get("coverage_threshold", 0.90),
        missing_threshold=validation_cfg.get("missing_threshold", 0.05),
    )

    processed = processed.merge(
        metadata[["id", "name", "level"]],
        left_on="sector_industry_id",
        right_on="id",
        how="left",
    )
    hierarchy_mapping = build_hierarchy_mapping(hierarchy, metadata)
    return processed, hierarchy_mapping, metadata


def determine_signal_date(data: pd.DataFrame, arg_date: str = None) -> pd.Timestamp:
    available = sorted(data["trade_date"].dropna().unique())
    if not available:
        raise ValueError("No trade data available after preprocessing.")

    available_ts = [pd.to_datetime(d).normalize() for d in available]
    if arg_date:
        target = _normalize_date(arg_date)
        if target not in available_ts:
            max_date = max(available_ts)
            raise ValueError(
                f"Signal date {target.date()} not found in loaded data. Latest available is {max_date.date()}."
            )
        return target

    return max(available_ts)


def is_month_end_trading_day(data: pd.DataFrame, signal_date: pd.Timestamp) -> bool:
    dates = pd.DataFrame({"trade_date": pd.to_datetime(data["trade_date"]).dt.normalize().unique()})
    dates["ym"] = dates["trade_date"].dt.to_period("M")
    month_ends = set(pd.to_datetime(dates.groupby("ym")["trade_date"].max().tolist()).normalize())
    return signal_date in month_ends


def load_current_holdings(path: str, signal_day_data: pd.DataFrame) -> PositionManager:
    pm = PositionManager(config={})

    if not os.path.exists(path):
        print(f"Holdings file not found: {path}. Using empty holdings.")
        return pm

    df = pd.read_csv(path)
    required_cols = {"sector_industry_id", "weight", "type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Holdings CSV missing required columns: {sorted(missing)}")

    close_by_id = signal_day_data.set_index("sector_industry_id")["close"].to_dict()

    for _, row in df.iterrows():
        sid = str(row["sector_industry_id"])
        weight = _safe_float(row.get("weight"), 0.0)
        sector_type = str(row.get("type", "left"))

        entry_date_val = row.get("entry_date")
        if pd.isna(entry_date_val):
            entry_date = pd.to_datetime(signal_day_data["trade_date"].iloc[0])
        else:
            entry_date = pd.to_datetime(entry_date_val)

        entry_price = row.get("entry_price")
        if pd.isna(entry_price):
            try:
                sid_num = int(float(row["sector_industry_id"]))
            except (TypeError, ValueError):
                sid_num = row["sector_industry_id"]
            entry_price = _safe_float(close_by_id.get(sid_num), 0.0)
        else:
            entry_price = _safe_float(entry_price, 0.0)

        score = _safe_float(row.get("current_score"), 0.0)
        entry_score = _safe_float(row.get("entry_score"), score)

        pm.positions[sid] = {
            "weight": weight,
            "type": sector_type,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "current_score": score,
            "entry_score": entry_score,
        }

    print(f"Loaded current holdings: {len(pm.positions)} positions")
    return pm


def build_target_portfolio(
    config: dict,
    signal_day_data: pd.DataFrame,
    hierarchy: Dict,
    full_history: pd.DataFrame = None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float], Dict[str, float], List[str], List[str]]:
    trend_calc = TrendFeatureCalculator(config)
    left_calc = LeftFeatureCalculator(config)
    trend_selector = TrendSelector(config)
    left_selector = LeftSelector(config)

    signal_date = signal_day_data["trade_date"].iloc[0]
    history = full_history if full_history is not None else signal_day_data
    trend_features = trend_calc.calculate_snapshot(history, signal_date, hierarchy)
    trend_scores = trend_features.copy()
    trend_scores["S_trend"] = trend_calc.calculate_score(trend_features)

    trend_selection = trend_selector.select(
        trend_scores,
        hierarchy,
        signal_date,
    )
    trend_pool = trend_selection["trend_pool"]

    left_features = left_calc.calculate_snapshot(history, signal_date, hierarchy)
    left_features["S_recovery"] = left_calc.calculate_score(left_features)

    returns_30 = signal_day_data.set_index("sector_industry_id")["r_30"]
    left_selection = left_selector.select(
        left_features,
        trend_pool,
        returns_30,
        signal_date,
    )
    left_pool = left_selection["left_pool"]

    allocation = config.get("backtest", {}).get("allocation", {})
    trend_w = allocation.get("trend", 0.85)
    left_w = allocation.get("left", 0.15)
    pos_cfg = config.get("risk_control", {}).get("position", {})
    trend_cap = pos_cfg.get("max_single_sector", 0.10)
    left_cap = pos_cfg.get("max_left_single", 0.02)

    target_weights: Dict[str, float] = {}
    target_types: Dict[str, str] = {}

    trend_score_map = {}
    if "S_trend" in trend_scores.columns:
        trend_score_map = trend_scores.set_index("sector_industry_id")["S_trend"].to_dict()

    left_score_map = {}
    if "S_recovery" in left_features.columns:
        left_score_map = left_features.set_index("sector_industry_id")["S_recovery"].to_dict()

    trend_alloc, _ = _allocate_with_caps(
        [str(s) for s in trend_pool],
        trend_w,
        trend_cap,
        {str(k): _safe_float(v) for k, v in trend_score_map.items()},
    )
    left_alloc, _ = _allocate_with_caps(
        [str(s) for s in left_pool],
        left_w,
        left_cap,
        {str(k): _safe_float(v) for k, v in left_score_map.items()},
    )

    for sid, w in trend_alloc.items():
        if w > 0:
            target_weights[sid] = w
            target_types[sid] = "trend"
    for sid, w in left_alloc.items():
        if w > 0:
            target_weights[sid] = w
            target_types[sid] = "left"

    return target_weights, target_types, trend_score_map, left_score_map, list(trend_pool), list(left_pool)


def rebalance_orders(
    signal_date: pd.Timestamp,
    signal_day_data: pd.DataFrame,
    current_pm: PositionManager,
    target_weights: Dict[str, float],
    target_types: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict] = []
    tol = 1e-8

    close_by_id = signal_day_data.set_index("sector_industry_id")["close"].to_dict()
    current_positions = current_pm.get_portfolio().get("positions", {})

    all_ids = sorted(set(current_positions.keys()) | set(target_weights.keys()))

    for sid in all_ids:
        cur = current_positions.get(sid, {})
        cur_w = _safe_float(cur.get("weight"), 0.0)
        tgt_w = _safe_float(target_weights.get(sid), 0.0)
        delta = tgt_w - cur_w

        if abs(delta) <= tol:
            continue

        try:
            sid_num = int(float(sid))
        except (TypeError, ValueError):
            sid_num = sid

        ref_price = _safe_float(close_by_id.get(sid_num), 0.0)

        if delta > 0:
            action = "buy"
            reason = "rebalance_increase" if cur_w > 0 else "rebalance_new"
            sector_type = target_types.get(sid, cur.get("type", ""))
            order_w = delta
        else:
            action = "sell"
            reason = "rebalance_decrease" if tgt_w > 0 else "rebalance_exit"
            sector_type = cur.get("type", target_types.get(sid, ""))
            order_w = abs(delta)

        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "execution": "T+1",
                "action": action,
                "sector_industry_id": sid,
                "sector_type": sector_type,
                "order_weight": order_w,
                "current_weight": cur_w,
                "target_weight": tgt_w,
                "reference_close": ref_price,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


def monitor_orders(
    config: dict,
    signal_date: pd.Timestamp,
    signal_day_data: pd.DataFrame,
    hierarchy: Dict,
    current_pm: PositionManager,
    full_history: pd.DataFrame = None,
) -> pd.DataFrame:
    sell_monitor = SellMonitor(config)
    supplement_engine = SupplementEngine(config)
    left_calc = LeftFeatureCalculator(config)

    rows: List[Dict] = []

    portfolio = current_pm.get_portfolio()
    sell_signals = sell_monitor.get_sell_signals(portfolio, signal_day_data, signal_date)

    current_positions = portfolio.get("positions", {})
    close_by_id = signal_day_data.set_index("sector_industry_id")["close"].to_dict()

    for sig in sell_signals:
        sid = str(sig["sector_industry_id"])
        pos = current_positions.get(sid, {})
        cur_w = _safe_float(pos.get("weight"), 0.0)

        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "execution": "T+1",
                "action": "sell",
                "sector_industry_id": sid,
                "sector_type": pos.get("type", ""),
                "order_weight": cur_w,
                "current_weight": cur_w,
                "target_weight": 0.0,
                "reference_close": _safe_float(sig.get("current_price"), 0.0),
                "reason": sig.get("signal_type", "sell_signal"),
            }
        )

        current_pm.remove_position(
            sid,
            signal_date,
            _safe_float(sig.get("current_price"), 0.0),
            sig.get("signal_type", "sell_signal"),
        )

    if not sell_signals:
        return pd.DataFrame(rows)

    all_level4 = signal_day_data[signal_day_data["level"] == 4]["sector_industry_id"].tolist()
    trend_pool = [s for s, p in portfolio["positions"].items() if p.get("type") == "trend"]

    history = full_history if full_history is not None else signal_day_data
    left_features = left_calc.calculate_snapshot(history, signal_date, hierarchy)
    left_features["S_recovery"] = left_calc.calculate_score(left_features)
    scores = left_features.set_index("sector_industry_id")["S_recovery"]

    supplement_result = supplement_engine.execute(
        sell_signals=sell_signals,
        portfolio=portfolio,
        all_level4=all_level4,
        trend_pool=trend_pool,
        scores=scores,
        portfolio_value=1.0,
    )

    for sid, w in supplement_result.get("weights", {}).items():
        try:
            sid_num = int(float(sid))
        except (TypeError, ValueError):
            sid_num = sid

        ref_price = _safe_float(close_by_id.get(sid_num), 0.0)
        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "execution": "T+1",
                "action": "buy",
                "sector_industry_id": str(sid),
                "sector_type": "left",
                "order_weight": _safe_float(w),
                "current_weight": 0.0,
                "target_weight": _safe_float(w),
                "reference_close": ref_price,
                "reason": "supplement",
            }
        )

    return pd.DataFrame(rows)


def save_outputs(
    output_dir: str,
    signal_date: pd.Timestamp,
    mode_used: str,
    orders: pd.DataFrame,
    target_weights: Dict[str, float],
    target_types: Dict[str, str],
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    date_tag = signal_date.strftime("%Y%m%d")
    orders_path = os.path.join(output_dir, f"next_day_orders_{date_tag}_{mode_used}.csv")
    orders.to_csv(orders_path, index=False, encoding="utf-8-sig")

    target_rows = [
        {
            "signal_date": signal_date.date().isoformat(),
            "sector_industry_id": sid,
            "sector_type": target_types.get(sid, ""),
            "target_weight": _safe_float(w),
        }
        for sid, w in sorted(target_weights.items(), key=lambda x: x[0])
    ]
    target_df = pd.DataFrame(target_rows)
    target_path = os.path.join(output_dir, f"target_portfolio_{date_tag}.csv")
    target_df.to_csv(target_path, index=False, encoding="utf-8-sig")

    return orders_path, target_path


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return 1

    config = load_config(args.config)

    processed, hierarchy, _metadata = load_market_data(config, args)
    signal_date = determine_signal_date(processed, args.signal_date)

    day_data = processed[processed["trade_date"].dt.normalize() == signal_date].copy()
    if day_data.empty:
        print(f"No data on signal date: {signal_date.date()}")
        return 1

    current_pm = load_current_holdings(args.holdings, day_data)

    target_weights, target_types, _trend_scores, _left_scores, trend_pool, left_pool = build_target_portfolio(
        config, day_data, hierarchy, processed[processed["trade_date"] <= signal_date]
    )

    if args.mode == "auto":
        mode_used = "rebalance" if is_month_end_trading_day(processed, signal_date) else "monitor"
    else:
        mode_used = args.mode

    if mode_used == "rebalance":
        orders = rebalance_orders(signal_date, day_data, current_pm, target_weights, target_types)
    else:
        orders = monitor_orders(
            config, signal_date, day_data, hierarchy, current_pm,
            processed[processed["trade_date"] <= signal_date]
        )

    if orders.empty:
        orders = pd.DataFrame(
            [
                {
                    "signal_date": signal_date.date().isoformat(),
                    "execution": "T+1",
                    "action": "none",
                    "sector_industry_id": "",
                    "sector_type": "",
                    "order_weight": 0.0,
                    "current_weight": 0.0,
                    "target_weight": 0.0,
                    "reference_close": 0.0,
                    "reason": "no_trade",
                }
            ]
        )

    orders_path, target_path = save_outputs(
        args.output,
        signal_date,
        mode_used,
        orders,
        target_weights,
        target_types,
    )

    print("=" * 60)
    print("Daily signal generation completed")
    print(f"Signal date: {signal_date.date()}")
    print(f"Mode used: {mode_used}")
    print(f"Trend pool size: {len(trend_pool)}")
    print(f"Left pool size: {len(left_pool)}")
    print(f"Orders: {len(orders)}")
    print(f"Orders file: {orders_path}")
    print(f"Target portfolio file: {target_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
