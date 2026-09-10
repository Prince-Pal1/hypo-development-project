"""
Instance Execution & Backtest of Sydney Range Fade + SL-Flip Reversal Strategy
Period: August 1, 2026 to September 8, 2026 on Gold (XAUUSD_5m.parquet)
Parameters:
  - x_offset: 3.5 points
  - sl_1: 10.0 points
  - tp_1: 20.0 points
  - y (sl_2): 10.0 points
  - tp_2: 20.0 points
  - base_units: 10.0 troy oz
  - Fee model: IC Markets cTrader Raw Spread (0.12 pt spread, $0.07/oz commission, 0.03 pt slippage)
"""

import datetime as dt
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.session.engine import SessionEngine
from src.core.risk_service import RiskService, RiskServiceLimits
from src.strategies.sydney_range_fade import SydneyRangeFadeStrategy, SydneyRangeFadeConfig
from src.sequencer.fsm import TradeChain, ChainedFollowUpConfig, ChainStatus
from src.execution.range_sweep_v2_simulator import RangeSweepV2Simulator, RangeSweepV2SimConfig
from src.execution.simulator import FeeModel
from src.portfolio.portfolio import PortfolioEngine
from src.reports.visualizer import ReportVisualizer
from src.cms.models import HypothesisCard, HypothesisStatus
from src.cms.repository import DatabaseRepository
from scripts.test_range_sweep_v2_aug_sept_2026 import run_range_sweep_v2_backtest

PARQUET_PATH = Path("/Users/prince/algo-trading/data/historical/XAUUSD_5m.parquet")
DB_PATH = Path("hypotrader.db")
REPORT_PATH = Path("reports/hypotrader_aug_sept_2026_report.html")


def run_aug_sept_2026_backtest(
    dataset_path: str = str(PARQUET_PATH),
    output_path: str = str(REPORT_PATH),
    start_date: str = "2026-08-04",
    end_date: str = "2026-09-07",
    initial_equity: float = 50000.0,
    session_start: str = "sydney",
    x_offset: float = 3.5,
    sl1: float = 10.0,
    tp1: float = 20.0,
    sl2: float = 10.0,
    tp2: float = 20.0,
    units: float = 10.0,
    ctc1: float = 10.0,
    ctc2: float = 10.0,
    decay: float = 15.0,
    enable_flip: bool = True,
    expiry_time: str = "17:00",
):
    return run_range_sweep_v2_backtest(
        dataset_path=dataset_path,
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
        initial_equity=initial_equity,
        session_start=session_start,
        x_offset=x_offset,
        sl1=sl1,
        tp1=tp1,
        ctc1=ctc1,
        ctc2=ctc2,
        decay=decay,
        sl2=sl2,
        tp2=tp2,
        enable_flip=enable_flip,
        expiry_time=expiry_time,
        units=units,
        db_path=str(DB_PATH),
    )
    dataset_file = Path(dataset_path)
    report_path = Path(output_path)

    print("=" * 95)
    print(f"HYPOTRADER: INSTANCE BACKTEST — {start_date} TO {end_date} (GOLD)")
    print(f"Dataset : {dataset_file}")
    print(f"Database: {DB_PATH}")
    print(f"Strategy: x = {x_offset} pts | sl1 = {sl1} pts | tp1 = {tp1} pts | ctc = {ctc} pts | sl2 (y) = {sl2} pts | tp2 = {tp2} pts")
    print(f"Sizing  : {units} Troy Oz | Fee Profile: IC Markets Raw Spread (0.12 spread + $7/lot RT comm)")
    print("=" * 95)

    if not dataset_file.exists():
        print(f"ERROR: Dataset not found at {dataset_file}")
        return

    # Load 5-minute Gold data
    df = pd.read_parquet(dataset_file)
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["utc_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["utc_time"] = pd.to_datetime(df["timestamp"], utc=True)

    session_engine = SessionEngine()
    risk_limits = RiskServiceLimits(
        max_size_multiplier_per_chain_step=1.0,
        max_chain_depth=2,
        daily_loss_circuit_breaker_pct=3.0,
    )
    risk_service = RiskService(risk_limits)

    strategy_config = SydneyRangeFadeConfig(
        x_offset=x_offset,
        sl_points=sl1,
        tp_points=tp1,
        base_units=units,
    )
    strategy = SydneyRangeFadeStrategy(
        config=strategy_config,
        session_engine=session_engine,
        risk_service=risk_service,
    )

    chain_config = ChainedFollowUpConfig(
        sl_flip_offset=sl2,  # y = sl2 points
        tp_flip_offset=tp2,  # tp2 = tp2 points
        requested_multiplier=1.0,
        max_chain_depth=2,
    )
    simulator = ExecutionSimulator(fee_model=FeeModel(), point_value=1.0, ctc_threshold_pts=ctc)
    repo = DatabaseRepository(DB_PATH)


    # Register/Update Hypothesis in CMS
    hypo_id = "HYPO-GOLD-AUG2026"
    hypo = HypothesisCard(
        id=hypo_id,
        title=f"Gold 12:30 IST Sydney Range Fade with {sl2}pt SL-Flip Reversal ({start_date} to {end_date})",
        economic_rationale="Session liquidity sweep fade with contingent reversal breakout on stop run.",
        asset_symbol="XAUUSD",
        author="Prince",
        status=HypothesisStatus.IN_BACKTEST,
        target_regimes=[f"{start_date} to {end_date} Bull Market"],
        param_manifest={"x": x_offset, "sl1": sl1, "tp1": tp1, "y": sl2, "tp2": tp2, "units": units},
    )
    repo.save_hypothesis(hypo)

    eval_dates = pd.date_range(start_date, end_date, freq="B").date

    current_equity = initial_equity
    daily_equity_records = {}
    trade_pnls = []
    executed_chains = []
    daily_chart_records = []


    print(f"\n{'Date':<11} | {'Eval Price':<10} | {'Range [Low - High]':<21} | {'Dir':<5} | {'Entry':<9} | {'Leg 1 Result':<20} | {'Leg 2 (Flip) Result':<23} | {'Net PnL':<9}")
    print("-" * 125)

    for target_date in eval_dates:
        # Reset daily circuit breaker for the new trading day
        risk_service.reset_daily_circuit_breaker()

        window = session_engine.get_session_window(target_date)
        end_of_day_utc = dt.datetime(target_date.year, target_date.month, target_date.day, 21, 0, tzinfo=dt.timezone.utc)

        # Slice bars for session window up to 21:00 UTC
        day_df = df[(df["utc_time"] >= window.sydney_open_utc) & (df["utc_time"] <= end_of_day_utc)]

        bar_records = []
        for _, r in day_df.iterrows():
            bar_records.append({
                "t": int(r["utc_time"].timestamp()),
                "o": round(float(r["open"]), 2),
                "h": round(float(r["high"]), 2),
                "l": round(float(r["low"]), 2),
                "c": round(float(r["close"]), 2),
                "v": round(float(r["volume"]), 1),
            })

        # Check if enough bars exist for evaluation
        session_bars = day_df[(day_df["utc_time"] >= window.sydney_open_utc) & (day_df["utc_time"] <= window.eval_time_utc)]
        if len(session_bars) < 20:
            daily_equity_records[target_date] = current_equity
            daily_chart_records.append({
                "date": str(target_date),
                "day_type": "INSUFFICIENT_DATA",
                "day_of_week": target_date.strftime("%A"),
                "range_high": 0,
                "range_low": 0,
                "midpoint": 0,
                "range_pts": 0,
                "eval_price": 0,
                "dist_to_low": 0,
                "dist_to_high": 0,
                "pnl_net": 0.0,
                "outcome_summary": "Skipped: insufficient market bars in session window.",
                "legs": [],
                "events": [],
                "bars": bar_records,
                "sessions": {
                    "sydney_open": int(window.sydney_open_utc.timestamp()),
                    "eval_time": int(window.eval_time_utc.timestamp()),
                    "london_close": int(dt.datetime(target_date.year, target_date.month, target_date.day, 16, 30, tzinfo=dt.timezone.utc).timestamp()),
                    "eod": int(end_of_day_utc.timestamp()),
                },
            })
            continue

        signal = strategy.evaluate_day(target_date, day_df, account_equity=current_equity)
        if signal is None:
            daily_equity_records[target_date] = current_equity
            r_high = float(session_bars["high"].max())
            r_low = float(session_bars["low"].min())
            eval_p = float(session_bars.iloc[-1]["close"])
            dist_l = eval_p - r_low
            dist_h = r_high - eval_p
            is_tie = abs(dist_l - dist_h) <= getattr(strategy.config, "tie_tolerance", 0.05)
            daily_chart_records.append({
                "date": str(target_date),
                "day_type": "TIE" if is_tie else "SKIPPED",
                "day_of_week": target_date.strftime("%A"),
                "range_high": r_high,
                "range_low": r_low,
                "midpoint": (r_high + r_low) / 2.0,
                "range_pts": r_high - r_low,
                "eval_price": eval_p,
                "dist_to_low": dist_l,
                "dist_to_high": dist_h,
                "tie_detected": is_tie,
                "pnl_net": 0.0,
                "outcome_summary": f"No trade — Proximity tie at 12:30 IST evaluation ({dist_l:.2f} pts to Low vs {dist_h:.2f} pts to High)." if is_tie else "Skipped by risk management.",
                "legs": [],
                "events": [],
                "bars": bar_records,
                "sessions": {
                    "sydney_open": int(window.sydney_open_utc.timestamp()),
                    "eval_time": int(window.eval_time_utc.timestamp()),
                    "london_close": int(dt.datetime(target_date.year, target_date.month, target_date.day, 16, 30, tzinfo=dt.timezone.utc).timestamp()),
                    "eod": int(end_of_day_utc.timestamp()),
                },
            })
            continue

        chain_id = f"CHAIN-{target_date.strftime('%Y%m%d')}"
        chain = TradeChain(
            chain_id=chain_id,
            config=chain_config,
            risk_service=risk_service,
            account_equity=current_equity,
        )
        chain.computed_context = {
            "target_date": target_date,
            "range_high": signal.range_high,
            "range_low": signal.range_low,
            "midpoint": (signal.range_high + signal.range_low) / 2.0,
            "range_pts": signal.range_high - signal.range_low,
            "eval_price": signal.eval_price,
            "dist_to_low": signal.eval_price - signal.range_low,
            "dist_to_high": signal.range_high - signal.eval_price,
        }

        # Start Leg 1 (with pending check against eval_price)
        chain.start_leg1(
            direction=signal.direction,
            entry_time=signal.eval_time_utc,
            entry_price=signal.entry_price,
            base_units=signal.units,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            eval_price=signal.eval_price,
        )

        # Simulate forward bars after 12:30 IST
        forward_bars = day_df[day_df["utc_time"] > signal.eval_time_utc]
        for _, row in forward_bars.iterrows():
            simulator.process_bar(
                chain=chain,
                bar_time=row["utc_time"],
                open_p=row["open"],
                high_p=row["high"],
                low_p=row["low"],
                close_p=row["close"],
            )
            if chain.status == ChainStatus.COMPLETED or chain.current_leg is None:
                break

        # If trade remained open or pending at end of day, close at session close
        active_leg = chain.current_leg
        if active_leg and active_leg.is_open:
            if active_leg.is_pending:
                # Limit order never filled -> cancel without PnL
                active_leg.exit_reason = "EXPIRATION"
                active_leg.is_open = False
                chain.record_event(
                    "EXPIRATION",
                    forward_bars.iloc[-1]["utc_time"] if len(forward_bars) > 0 else window.eval_time_utc,
                    active_leg.entry_price,
                    "Order canceled unfilled at session close",
                    leg_id=active_leg.leg_id,
                )
            else:
                last_bar = forward_bars.iloc[-1]
                active_leg.close(
                    exit_time=last_bar["utc_time"],
                    exit_price=float(last_bar["close"]),
                    reason="EXPIRATION",
                    commission=simulator.fee_model.commission_per_unit * active_leg.units,
                    spread_cost=simulator.fee_model.spread_pts * active_leg.units,
                    slippage_cost=simulator.fee_model.base_slippage_pts * active_leg.units,
                )
                chain.status = ChainStatus.COMPLETED

        # Record and log
        repo.save_trade_chain(chain)
        executed_chains.append(chain)

        current_equity += chain.total_pnl_net
        daily_equity_records[target_date] = current_equity

        l1 = chain.legs[0] if len(chain.legs) > 0 else None
        l2 = chain.legs[1] if len(chain.legs) > 1 else None

        l1_desc = f"{l1.exit_reason} (${l1.pnl_net:+.1f})" if l1 and l1.exit_price else ("UNFILLED" if l1 and l1.is_pending else "OPEN")
        l2_desc = f"{l2.direction} {l2.exit_reason} (${l2.pnl_net:+.1f})" if l2 and l2.exit_price else ("—" if not l2 else "OPEN")

        range_str = f"[{signal.range_low:.1f} - {signal.range_high:.1f}]"
        print(f"{target_date.strftime('%Y-%m-%d')} | ${signal.eval_price:<9.2f} | {range_str:<21} | {signal.direction:<5} | ${signal.entry_price:<8.2f} | {l1_desc:<20} | {l2_desc:<23} | ${chain.total_pnl_net:+<8.2f}")

        for leg in chain.legs:
            if not leg.is_open and leg.exit_price is not None and leg.exit_reason not in ("EXPIRATION",):
                trade_pnls.append(leg.pnl_net)

        legs_data = []
        for leg in chain.legs:
            legs_data.append({
                "leg_id": leg.leg_id,
                "depth": leg.depth,
                "direction": leg.direction,
                "entry_price": leg.entry_price,
                "entry_time": str(leg.entry_time),
                "initial_sl_price": getattr(leg, "initial_sl_price", leg.sl_price),
                "sl_price": leg.sl_price,
                "tp_price": leg.tp_price,
                "ctc_armed": leg.ctc_armed,
                "ctc_price": getattr(leg, "ctc_price", None),
                "ctc_time": str(leg.ctc_time) if getattr(leg, "ctc_time", None) else None,
                "exit_price": leg.exit_price,
                "exit_time": str(leg.exit_time) if leg.exit_time else None,
                "exit_reason": leg.exit_reason,
                "pnl_net": leg.pnl_net,
                "is_pending": leg.is_pending,
            })

        day_status = "TRADE"
        if l1 and l1.is_pending and l1.exit_reason == "EXPIRATION":
            day_status = "EXPIRED"
        elif l1 and l1.exit_reason == "CTC" and len(chain.legs) == 1:
            day_status = "CTC"

        daily_chart_records.append({
            "date": str(target_date),
            "day_type": day_status,
            "day_of_week": target_date.strftime("%A"),
            "range_high": signal.range_high,
            "range_low": signal.range_low,
            "midpoint": (signal.range_high + signal.range_low) / 2.0,
            "range_pts": signal.range_high - signal.range_low,
            "eval_price": signal.eval_price,
            "dist_to_low": signal.eval_price - signal.range_low,
            "dist_to_high": signal.range_high - signal.eval_price,
            "direction": signal.direction,
            "entry_price": signal.entry_price,
            "is_instant": getattr(signal, "is_instant", False),
            "sl_price": signal.sl_price,
            "tp_price": signal.tp_price,
            "pnl_net": chain.total_pnl_net,
            "outcome_summary": chain.generate_outcome_path_summary(),
            "legs": legs_data,
            "events": chain.events,
            "bars": bar_records,
            "sessions": {
                "sydney_open": int(window.sydney_open_utc.timestamp()),
                "eval_time": int(window.eval_time_utc.timestamp()),
                "london_close": int(dt.datetime(target_date.year, target_date.month, target_date.day, 16, 30, tzinfo=dt.timezone.utc).timestamp()),
                "eod": int(end_of_day_utc.timestamp()),
            },
        })

    equity_series = pd.Series(daily_equity_records)

    # Portfolio Analytics
    portfolio_engine = PortfolioEngine(initial_equity=initial_equity)
    metrics = portfolio_engine.compute_metrics(equity_series, trade_pnls)

    print("-" * 125)
    print("\n" + "=" * 80)
    print("BACKTEST PERFORMANCE SUMMARY (AUG 1 – SEP 8, 2026):")
    print(f"  Trading Days Evaluated : {len(executed_chains)}")
    print(f"  Total Trades (Legs)    : {metrics.total_trades}")
    print(f"  Winning Trades         : {metrics.winning_trades} ({metrics.win_rate_pct:.1f}%)")
    print(f"  Losing Trades          : {metrics.losing_trades}")
    print(f"  Initial Balance        : ${metrics.initial_equity:,.2f}")
    print(f"  Final Balance          : ${metrics.final_equity:,.2f}")
    print(f"  Total Net Profit       : ${metrics.total_net_pnl:+,.2f} ({metrics.total_return_pct:+.2f}%)")
    print(f"  Annualized Return      : {metrics.annualized_return_pct:+.2f}%")
    print(f"  Sharpe Ratio           : {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio          : {metrics.sortino_ratio:.2f}")
    print(f"  Profit Factor          : {metrics.profit_factor:.2f}")
    print(f"  Max Drawdown           : -{metrics.max_drawdown_pct:.2f}% (${metrics.max_drawdown_usd:,.2f})")
    print(f"  Calmar Ratio           : {metrics.calmar_ratio:.2f}")
    print("=" * 80)

    # Contingent Reversal SL-Flip & CTC Breakdown
    flips = [c for c in executed_chains if len(c.legs) > 1]
    ctc_exits = [c for c in executed_chains for leg in c.legs if leg.exit_reason == "CTC"]
    flip_wins = [c for c in flips if len(c.legs) > 1 and c.legs[1].pnl_net > 0]
    flip_losses = [c for c in flips if len(c.legs) > 1 and c.legs[1].pnl_net < 0]

    print("\nCONTINGENT SL-FLIP & CTC STATS:")
    print(f"  CTC Breakeven Exits (SL->BE)           : {len(ctc_exits)}")
    print(f"  Total Leg 1 Stop-Outs (Flips Triggered): {len(flips)}")
    print(f"  Leg 2 Reversal Wins (TP2 Hit)          : {len(flip_wins)} ({len(flip_wins)/max(len(flips),1)*100:.1f}%)")
    print(f"  Leg 2 Reversal Losses (SL2 Hit)        : {len(flip_losses)}")
    total_flip_contribution = sum(c.legs[1].pnl_net for c in flips)
    print(f"  Net PnL Generated by Leg 2 Flips       : ${total_flip_contribution:+,.2f}")
    print("=" * 80)

    # Generate HTML Report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_file = ReportVisualizer.generate_html_report(
        hypothesis_title=hypo.title,
        hypothesis_id=hypo.id,
        metrics=metrics,
        daily_equity_series=equity_series,
        executed_chains=executed_chains,
        output_filepath=report_path,
        daily_chart_data=daily_chart_records,
    )
    print(f"\n[REPORT GENERATED] Interactive HTML Visualizer Report saved to:")
    print(f"  file://{report_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run 12:30 Range Fade Backtest with custom inputs & parameters."
    )
    parser.add_argument("--session-start", type=str, default="sydney", choices=["sydney", "tokyo"], help="Session interval start: 'sydney' or 'tokyo' (default: sydney)")
    parser.add_argument("--x", type=float, default=3.5, help="X-offset (points) from high/low for Leg 1 limit order (default: 3.5)")
    parser.add_argument("--sl1", type=float, default=10.0, help="Stop loss in points for Leg 1 (default: 10.0)")
    parser.add_argument("--tp1", type=float, default=20.0, help="Take profit in points for Leg 1 (default: 20.0)")
    parser.add_argument("--ctc1", type=float, default=10.0, help="CTC activation threshold in points for Leg 1 (default: 10.0)")
    parser.add_argument("--ctc2", type=float, default=10.0, help="CTC activation threshold in points for Leg 2 flip (default: 10.0)")
    parser.add_argument("--ctc", type=float, default=None, help="Shorthand for both ctc1 and ctc2")
    parser.add_argument("--decay", type=float, default=15.0, help="Decay threshold in points for canceling unfilled Leg 1 (default: 15.0)")
    parser.add_argument("--sl2", "--y", dest="sl2", type=float, default=10.0, help="Stop loss in points for Leg 2 flip / y-offset (default: 10.0)")
    parser.add_argument("--tp2", type=float, default=20.0, help="Take profit in points for Leg 2 flip (default: 20.0)")
    parser.add_argument("--expiry", type=str, default="17:00", help="IST time cutoff for unfilled limit order (default: 17:00)")
    parser.add_argument("--enable-flip", dest="enable_flip", action="store_true", default=True, help="Enable reverse follow-back trade on SL hit (default: True)")
    parser.add_argument("--no-flip", dest="enable_flip", action="store_false", help="Disable reverse follow-back trade on SL hit")
    parser.add_argument("--units", type=float, default=10.0, help="Position size in units/ounces (default: 10.0)")
    parser.add_argument("--start", type=str, default="2026-08-04", help="Start date YYYY-MM-DD (default: 2026-08-04)")
    parser.add_argument("--end", type=str, default="2026-09-07", help="End date YYYY-MM-DD (default: 2026-09-07)")
    parser.add_argument("--equity", type=float, default=50000.0, help="Initial equity balance in USD (default: 50000.0)")
    parser.add_argument("--dataset", type=str, default=str(PARQUET_PATH), help="Path to 5-minute parquet file")
    parser.add_argument("--output", type=str, default=str(REPORT_PATH), help="Output HTML report path")

    args = parser.parse_args()
    ctc1_val = args.ctc if args.ctc is not None else args.ctc1
    ctc2_val = args.ctc if args.ctc is not None else args.ctc2

    run_aug_sept_2026_backtest(
        dataset_path=args.dataset,
        output_path=args.output,
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        session_start=args.session_start,
        x_offset=args.x,
        sl1=args.sl1,
        tp1=args.tp1,
        ctc1=ctc1_val,
        ctc2=ctc2_val,
        decay=args.decay,
        sl2=args.sl2,
        tp2=args.tp2,
        enable_flip=args.enable_flip,
        expiry_time=args.expiry,
        units=args.units,
    )


