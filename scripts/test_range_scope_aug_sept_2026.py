"""
Range Scope Strategy (11:00 AM IST Strategy) — Instance Backtest
Period: August 4, 2026 to September 7, 2026 on Gold (XAUUSD_5m.parquet)

Rules:
  - Range: Sydney or Tokyo open -> 11:00 AM IST
  - Proximity Bias: Nearer range low -> SHORT; Nearer range high -> LONG
  - Scope Gate: Dist to target boundary >= X pts (otherwise skip)
  - Entry: Immediate market order at 11:00 AM IST
  - SL: Fixed stop loss (sl_points)
  - TP Modes:
      * Mode 1 (dynamic): TP at boundary before 12:30; adjusted at 12:30 to min/max with offset y
      * Mode 2 (wait 12:30): No TP before 12:30; at 12:30, recomputes expanded range & sets TP min/max with offset y
  - EOD close at 21:00 UTC (02:30 IST next day)
"""

import argparse
import datetime as dt
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.session.engine import SessionEngine
from src.core.risk_service import RiskService, RiskServiceLimits
from src.execution.ambiguity_resolver import load_1min_dataframe
from src.strategies.range_scope_v1 import RangeScopeStrategy, RangeScopeConfig
from src.execution.range_scope_simulator import RangeScopeSimulator, RangeScopeTradeResult
from src.execution.simulator import FeeModel, ZERO_FEE_MODEL
from src.portfolio.portfolio import PortfolioEngine

DEFAULT_PARQUET_PATH = Path("/Users/prince/algo-trading/data/historical/XAUUSD_5m.parquet")
DEFAULT_REPORT_PATH = Path("reports/range_scope_aug_sept_2026_report.html")


def run_range_scope_backtest(
    session_start: str = "sydney",
    eval_time_ist: str = "11:00",
    scope_min_x: float = 5.0,
    sl_points: float = 10.0,
    ctc_points: float = 10.0,
    tp_offset_y: float = 3.5,
    tp_mode: str = "mode_1_dynamic",
    units: float = 10.0,
    start_date: str = "2026-08-04",
    end_date: str = "2026-09-07",
    initial_equity: float = 50000.0,
    zero_fees: bool = False,
    broker_mode: str = "xm_broker",
    dataset_path: str = str(DEFAULT_PARQUET_PATH),
    output_path: str = str(DEFAULT_REPORT_PATH),
):
    parquet_path = Path(dataset_path)
    report_path = Path(output_path)

    print("=" * 95)
    print(f"HYPOTRADER: STRATEGY 1 — RANGE SCOPE ({eval_time_ist} IST) — {start_date} TO {end_date}")
    print(f"Dataset : {parquet_path}")
    print(
        f"Config  : Session: {session_start.upper()} | Eval: {eval_time_ist} IST | Scope X: {scope_min_x} pts | "
        f"SL: {sl_points} pts | CTC: {ctc_points} pts | TP Offset Y: {tp_offset_y} pts | Mode: {tp_mode} | Units: {units} | zero_fees: {zero_fees}"
    )
    print("=" * 95)

    if not parquet_path.exists():
        print(f"ERROR: Dataset not found at {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["utc_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["utc_time"] = pd.to_datetime(df["timestamp"], utc=True)

    # In XM Broker mode, bypass position-size clamping and daily circuit breaker so user's explicit size is honoured.
    risk_pct_cap = 999.0 if broker_mode == "xm_broker" else 2.0
    daily_cb_cap = 999.0 if broker_mode == "xm_broker" else 3.0
    risk_service = RiskService(RiskServiceLimits(
        max_size_multiplier_per_chain_step=1.0,
        max_chain_depth=1,
        daily_loss_circuit_breaker_pct=daily_cb_cap,
        max_cumulative_open_risk_pct=risk_pct_cap,
    ))
    strategy_config = RangeScopeConfig(
        session_start=session_start,
        eval_time_ist=eval_time_ist,
        scope_min_x=scope_min_x,
        sl_points=sl_points,
        ctc_points=ctc_points,
        tp_offset_y=tp_offset_y,
        tp_mode=tp_mode,
        base_units=units,
    )
    strategy = RangeScopeStrategy(strategy_config, risk_service)
    session_engine = strategy.session_engine
    df_1m = load_1min_dataframe(str(parquet_path).replace("5m.parquet", "1m.parquet"), reference_df=df)
    simulator = RangeScopeSimulator(
        fee_model=ZERO_FEE_MODEL if zero_fees else FeeModel(),
        point_value=1.0,
        df_1m=df_1m,
    )

    start_dt = dt.date.fromisoformat(start_date)
    end_dt = dt.date.fromisoformat(end_date)

    current_equity = initial_equity
    equity_records = {}
    daily_results = []
    daily_chart_records = []
    trade_results: list[RangeScopeTradeResult] = []
    trade_pnls = []

    curr = start_dt
    while curr <= end_dt:
        if curr.weekday() in (5, 6):  # Weekend
            curr += dt.timedelta(days=1)
            continue

        window = session_engine.get_session_window(curr)
        day_mask = (df["utc_time"] >= window.session_start_utc) & (df["utc_time"] <= window.eod_utc)
        day_df = df.loc[day_mask].copy()

        if len(day_df) < 20:
            curr += dt.timedelta(days=1)
            continue

        # Prepare 5m bar records for candlestick chart
        bar_records = [
            {
                "t": int(r["utc_time"].timestamp()),
                "o": float(r["open"]),
                "h": float(r["high"]),
                "l": float(r["low"]),
                "c": float(r["close"]),
                "v": float(r.get("volume", 0)),
                "time": int(r["utc_time"].timestamp()),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r.get("volume", 0)),
            }
            for _, r in day_df.iterrows()
        ]

        signal, ctx = strategy.evaluate_day_context(curr, day_df, current_equity)

        time_1230_utc = dt.datetime(curr.year, curr.month, curr.day, 12, 30, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(dt.timezone.utc)
        candle_1230 = day_df[day_df["utc_time"] == time_1230_utc]
        exact_price_1230 = float(candle_1230.iloc[0]["open"]) if len(candle_1230) > 0 else None

        sessions_dict = {
            "session_start": int(window.session_start_utc.timestamp()),
            "session_name": window.session_name,
            "eval_time": int(window.eval_time_utc.timestamp()),
            "eval_1100": int(window.eval_time_utc.timestamp()),
            "eval_1230": int(time_1230_utc.timestamp()),
            "time_1230": int(time_1230_utc.timestamp()),
            "london_close": int(dt.datetime(curr.year, curr.month, curr.day, 16, 30, tzinfo=dt.timezone.utc).timestamp()),
            "eod": int(window.eod_utc.timestamp()),
        }

        if signal is None:
            reason = ctx.get("reason", "NO_TRADE")
            equity_records[curr] = current_equity
            outcome_summary = f"Skipped at {eval_time_ist} IST ({reason}). Scope: {ctx.get('scope', 0.0):.1f} pts < {strategy_config.scope_min_x:.1f} pts."
            daily_results.append({
                "date": curr,
                "direction": "SKIP",
                "scope": ctx.get("scope", 0.0),
                "status": f"SKIP ({reason})",
                "entry_price": 0.0,
                "exit_price": 0.0,
                "pnl_net": 0.0,
                "equity": current_equity,
                "reason": reason,
            })
            daily_chart_records.append({
                "date": str(curr),
                "day_type": "SKIP",
                "day_of_week": curr.strftime("%A"),
                "strategy_tag": "SCOPE-1100",
                "range_high": ctx.get("range_high", 0.0),
                "range_low": ctx.get("range_low", 0.0),
                "midpoint": ctx.get("midpoint", 0.0),
                "range_pts": ctx.get("range_pts", 0.0),
                "eval_price": ctx.get("eval_price", 0.0),
                "dist_to_low": ctx.get("dist_to_low", 0.0),
                "dist_to_high": ctx.get("dist_to_high", 0.0),
                "scope_points": ctx.get("scope", ctx.get("scope_pts", 0.0)),
                "scope_min_x": scope_min_x,
                "tp_offset_y": tp_offset_y,
                "tp_mode": tp_mode,
                "direction": "SKIP",
                "entry_price": 0.0,
                "is_instant": False,
                "sl_price": 0.0,
                "tp_price": 0.0,
                "price_1230": exact_price_1230,
                "price_at_1230": exact_price_1230,
                "pnl_net": 0.0,
                "outcome_summary": outcome_summary,
                "legs": [],
                "events": [],
                "bars": bar_records,
                "sessions": sessions_dict,
            })
        else:
            trade = simulator.simulate_day(signal, day_df, window.session_start_utc)
            trade_results.append(trade)
            current_equity += trade.pnl_net
            equity_records[curr] = current_equity
            trade_pnls.append(trade.pnl_net)

            if trade.exit_reason == "TP":
                outcome_summary = f"{eval_time_ist} IST {trade.direction} filled @ {trade.entry_price:.1f} → Hit TP @ {trade.exit_price:.1f} (+{trade.pnl_points:.1f} pts, ${trade.pnl_net:+.2f})."
            elif trade.exit_reason == "TP_1230_LOCKIN":
                outcome_summary = f"{eval_time_ist} IST {trade.direction} filled @ {trade.entry_price:.1f} → 12:30 IST profit lock-in exit @ {trade.exit_price:.1f} (+{trade.pnl_points:.1f} pts, ${trade.pnl_net:+.2f})."
            elif trade.exit_reason == "CTC":
                outcome_summary = f"{eval_time_ist} IST {trade.direction} filled @ {trade.entry_price:.1f} → CTC-SL hit @ {trade.exit_price:.1f} (0.0 pts, ${trade.pnl_net:+.2f})."
            elif trade.exit_reason == "SL":
                outcome_summary = f"{eval_time_ist} IST {trade.direction} filled @ {trade.entry_price:.1f} → Hit SL @ {trade.exit_price:.1f} ({trade.pnl_points:.1f} pts, ${trade.pnl_net:+.2f})."
            else:
                outcome_summary = f"{eval_time_ist} IST {trade.direction} filled @ {trade.entry_price:.1f} → Closed at EOD @ {trade.exit_price:.1f} ({trade.pnl_points:.1f} pts, ${trade.pnl_net:+.2f})."

            legs_data = [{
                "leg_id": f"SCOPE_{curr.isoformat()}",
                "depth": 1,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "entry_time": str(trade.entry_time),
                "initial_sl_price": signal.sl_price,
                "sl_price": trade.sl_price,
                "ctc_armed": trade.ctc_armed,
                "ctc_price": trade.ctc_price,
                "ctc_time": str(trade.ctc_time) if trade.ctc_time else None,
                "initial_tp": signal.initial_tp,
                "tp_price": signal.initial_tp,
                "tp_at_1230": trade.tp_at_1230,
                "price_at_1230": trade.price_at_1230,
                "exit_price": trade.exit_price,
                "exit_time": str(trade.exit_time),
                "exit_reason": trade.exit_reason,
                "pnl_net": trade.pnl_net,
                "is_pending": False,
                "strategy_tag": "SCOPE-1100",
            }]

            daily_results.append({
                "date": curr,
                "direction": trade.direction,
                "scope": signal.scope_pts,
                "status": trade.exit_reason,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl_net": trade.pnl_net,
                "equity": current_equity,
                "reason": trade.exit_reason,
            })

            daily_chart_records.append({
                "date": str(curr),
                "day_type": "TRADE",
                "day_of_week": curr.strftime("%A"),
                "strategy_tag": "SCOPE-1100",
                "range_high": signal.range_high,
                "range_low": signal.range_low,
                "midpoint": (signal.range_high + signal.range_low) / 2.0,
                "range_pts": signal.range_high - signal.range_low,
                "eval_price": signal.eval_price,
                "dist_to_low": signal.dist_to_low,
                "dist_to_high": signal.dist_to_high,
                "scope_points": signal.scope_pts,
                "scope_min_x": scope_min_x,
                "tp_offset_y": tp_offset_y,
                "tp_mode": tp_mode,
                "direction": signal.direction,
                "entry_price": signal.entry_price,
                "is_instant": True,  # 11:00 AM entries are instant market fills
                "sl_price": signal.sl_price,
                "initial_tp": signal.initial_tp,
                "tp_price": signal.initial_tp,
                "tp_at_1230": trade.tp_at_1230,
                "price_1230": trade.price_at_1230 if trade.price_at_1230 is not None else exact_price_1230,
                "price_at_1230": trade.price_at_1230 if trade.price_at_1230 is not None else exact_price_1230,
                "pnl_net": trade.pnl_net,
                "outcome_summary": outcome_summary,
                "legs": legs_data,
                "events": trade.events,
                "bars": bar_records,
                "sessions": sessions_dict,
            })

        curr += dt.timedelta(days=1)

    # Print summary table
    print("\n" + "-" * 105)
    print(f"{'Date':<12} | {'Bias':<6} | {'Scope':<7} | {'Status':<15} | {'Entry':<8} | {'Exit':<8} | {'Net PnL ($)':<12} | {'Equity ($)':<10}")
    print("-" * 105)
    for r in daily_results:
        pnl_str = f"{r['pnl_net']:+9.2f}" if r['pnl_net'] != 0.0 else "     0.00"
        entry_str = f"{r['entry_price']:7.1f}" if r['entry_price'] > 0 else "    ---"
        exit_str = f"{r['exit_price']:7.1f}" if r['exit_price'] > 0 else "    ---"
        print(
            f"{r['date'].isoformat():<12} | {r['direction']:<6} | {r['scope']:6.1f}p | "
            f"{r['status']:<15} | {entry_str} | {exit_str} | {pnl_str} | ${r['equity']:,.2f}"
        )
    print("-" * 105)

    equity_series = pd.Series(equity_records)
    portfolio_engine = PortfolioEngine(initial_equity=initial_equity)
    metrics = portfolio_engine.compute_metrics(equity_series, trade_pnls)

    # Overall metrics
    total_trades = len(trade_results)
    wins = [t for t in trade_results if t.pnl_net > 0]
    losses = [t for t in trade_results if t.pnl_net < 0]
    ties = [t for t in trade_results if t.pnl_net == 0]
    net_pnl = sum(t.pnl_net for t in trade_results)

    print(f"\nBACKTEST SUMMARY:")
    print(f"Total Days Evaluated : {len(daily_results)}")
    print(f"Total Trades Executed: {total_trades}")
    print(f"Wins / Losses / Ties : {len(wins)} / {len(losses)} / {len(ties)}")
    print(f"Win Rate             : {metrics.win_rate_pct:.1f}%")
    print(f"Profit Factor        : {metrics.profit_factor:.2f}")
    print(f"Total Net PnL        : ${metrics.total_net_pnl:+,.2f}")
    print(f"Final Account Equity : ${metrics.final_equity:,.2f}")
    print("=" * 95)

    return {
        "metrics": metrics,
        "trade_results": trade_results,
        "daily_chart_records": daily_chart_records,
        "daily_equity_series": equity_series,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Range Scope (11:00 AM IST) Strategy Backtester")
    parser.add_argument("--session-start", choices=["sydney", "tokyo"], default="sydney")
    parser.add_argument("--scope-x", type=float, default=5.0, help="Minimum scope in points")
    parser.add_argument("--sl", type=float, default=10.0, help="Stop loss in points")
    parser.add_argument("--tp-y", type=float, default=3.5, help="Take profit offset Y in points at 12:30")
    parser.add_argument("--tp-mode", choices=["mode_1_dynamic", "mode_2_wait_1230"], default="mode_1_dynamic")
    parser.add_argument("--units", type=float, default=10.0, help="Trade lot size / units")
    parser.add_argument("--start", default="2026-08-04", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-09-07", help="End date (YYYY-MM-DD)")
    parser.add_argument("--equity", type=float, default=50000.0, help="Initial equity")
    parser.add_argument("--dataset", default=str(DEFAULT_PARQUET_PATH), help="Parquet dataset path")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="HTML report path")

    args = parser.parse_args()
    run_range_scope_backtest(
        session_start=args.session_start,
        scope_min_x=args.scope_x,
        sl_points=args.sl,
        tp_offset_y=args.tp_y,
        tp_mode=args.tp_mode,
        units=args.units,
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        dataset_path=args.dataset,
        output_path=args.report,
    )
