"""
Range Sweep Version 2 — Instance Backtest
Period: August 4, 2026 to September 7, 2026 on Gold (XAUUSD_5m.parquet)

Rules:
  - Range: Sydney open → 12:30 PM IST
  - Tie on proximity → no trade
  - Limit entry with decay (+15 pts) and 5 PM IST cancel
  - CTC at +10 pts on Position 1 and flip leg
  - SL-flip on initial SL only (not CTC)
  - EOD close at 21:00 UTC
"""

import argparse
import datetime as dt
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.session.engine import SessionEngine
from src.core.risk_service import RiskService, RiskServiceLimits
from src.strategies.range_sweep_v2 import RangeSweepV2Strategy, RangeSweepV2Config
from src.sequencer.fsm import TradeChain, ChainedFollowUpConfig, ChainStatus
from src.execution.range_sweep_v2_simulator import RangeSweepV2Simulator, RangeSweepV2SimConfig
from src.execution.simulator import FeeModel, ZERO_FEE_MODEL
from src.portfolio.portfolio import PortfolioEngine
from src.reports.visualizer import ReportVisualizer
from src.cms.models import HypothesisCard, HypothesisStatus
from src.cms.repository import DatabaseRepository

DEFAULT_PARQUET_PATH = Path("/Users/prince/algo-trading/data/historical/XAUUSD_5m.parquet")
DEFAULT_DB_PATH = Path("hypotrader.db")
DEFAULT_REPORT_PATH = Path("reports/range_sweep_v2_aug_sept_2026_report.html")


def run_range_sweep_v2_backtest(
    x_offset: float = 3.5,
    sl1: float = 10.0,
    tp1: float = 20.0,
    ctc1: float = 10.0,
    ctc2: float = 10.0,
    decay: float = 15.0,
    sl2: float = 10.0,
    tp2: float = 20.0,
    enable_flip: bool = True,
    expiry_time: str = "17:00",
    session_start: str = "sydney",
    eval_time_ist: str = "12:30",
    units: float = 10.0,
    start_date: str = "2026-08-04",
    end_date: str = "2026-09-07",
    initial_equity: float = 50000.0,
    zero_fees: bool = False,
    broker_mode: str = "xm_broker",
    dataset_path: str = str(DEFAULT_PARQUET_PATH),
    output_path: str = str(DEFAULT_REPORT_PATH),
    db_path: str = str(DEFAULT_DB_PATH),
):
    parquet_path = Path(dataset_path)
    report_path = Path(output_path)
    db_path_obj = Path(db_path)

    print("=" * 95)
    print(f"HYPOTRADER: RANGE SWEEP VERSION 2 — {start_date} TO {end_date} (GOLD)")
    print(f"Dataset : {parquet_path}")
    print(f"Database: {db_path_obj}")
    print(
        f"Strategy: Range Sweep V2 | Session: {session_start.upper()} | Eval: {eval_time_ist} IST | x={x_offset} | "
        f"sl1={sl1} | tp1={tp1} | ctc1={ctc1} | decay={decay} | expiry={expiry_time} IST | "
        f"flip={enable_flip} | sl2={sl2} | tp2={tp2} | ctc2={ctc2} | units={units} | zero_fees={zero_fees}"
    )
    print(f"Report  : {report_path}")
    print("=" * 95)

    if not parquet_path.exists():
        print(f"ERROR: Dataset not found at {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["utc_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["utc_time"] = pd.to_datetime(df["timestamp"], utc=True)

    parts = eval_time_ist.strip().split(":")
    eval_h = int(parts[0]) if len(parts) > 0 and parts[0] else 12
    eval_m = int(parts[1]) if len(parts) > 1 and parts[1] else 30
    session_engine = SessionEngine(
        session_start=session_start,
        eval_hour_ist=eval_h,
        eval_minute_ist=eval_m,
    )
    session_engine.set_limit_cancel_time_ist(expiry_time)

    # In XM Broker mode, bypass position-size clamping and daily circuit breaker so user's explicit size and flip rules are honoured.
    risk_pct_cap = 999.0 if broker_mode == "xm_broker" else 2.0
    daily_cb_cap = 999.0 if broker_mode == "xm_broker" else 3.0
    risk_limits = RiskServiceLimits(
        max_size_multiplier_per_chain_step=1.0,
        max_chain_depth=2 if enable_flip else 1,
        daily_loss_circuit_breaker_pct=daily_cb_cap,
        max_cumulative_open_risk_pct=risk_pct_cap,
    )
    risk_service = RiskService(risk_limits)

    strategy = RangeSweepV2Strategy(
        config=RangeSweepV2Config(
            x_offset=x_offset,
            sl_points=sl1,
            tp_points=tp1,
            sl2_points=sl2,
            tp2_points=tp2,
            ctc1_points=ctc1,
            ctc2_points=ctc2,
            decay_points=decay,
            enable_flip=enable_flip,
            session_start=session_start,
            limit_cancel_ist=expiry_time,
            eval_time_ist=eval_time_ist,
            base_units=units,
        ),
        session_engine=session_engine,
        risk_service=risk_service,
    )

    chain_config = ChainedFollowUpConfig(
        sl_flip_offset=sl2,
        tp_flip_offset=tp2,
        enable_flip=enable_flip,
        requested_multiplier=1.0,
        max_chain_depth=2 if enable_flip else 1,
    )
    simulator = RangeSweepV2Simulator(
        fee_model=ZERO_FEE_MODEL if zero_fees else FeeModel(),
        point_value=1.0,
        config=RangeSweepV2SimConfig(
            ctc1_points=ctc1,
            ctc2_points=ctc2,
            decay_points=decay,
            enable_flip=enable_flip,
        ),
    )
    repo = DatabaseRepository(db_path_obj)

    hypo_id = f"HYPO-RSV2-S{session_start[0].upper()}-X{x_offset}-SL{sl1}-TP{tp1}"
    hypo = HypothesisCard(
        id=hypo_id,
        title=f"Range Sweep V2 ({session_start.capitalize()}) — x={x_offset} sl1={sl1} tp1={tp1} ctc1={ctc1} flip={enable_flip} ({start_date} to {end_date})",
        economic_rationale=(
            f"Fade toward {session_start.capitalize()}-to-12:30 IST range extremes with limit entry, "
            "CTC breakeven protection, decay cancel, and contingent SL-flip reversal."
        ),
        asset_symbol="XAUUSD",
        author="Prince",
        status=HypothesisStatus.IN_BACKTEST,
        target_regimes=[f"{start_date} to {end_date}"],
        rules_summary=(
            f"Range {session_start.capitalize()}→12:30 IST; tie=skip; limit decay +{decay}; cancel {expiry_time} IST; "
            f"CTC1 +{ctc1}; flip={enable_flip} (SL2={sl2}, TP2={tp2}, CTC2={ctc2}); EOD close."
        ),
        param_manifest={
            "session_start": session_start,
            "x": x_offset,
            "sl1": sl1,
            "tp1": tp1,
            "ctc1": ctc1,
            "ctc2": ctc2,
            "decay": decay,
            "enable_flip": enable_flip,
            "sl2": sl2,
            "tp2": tp2,
            "units": units,
            "limit_cancel_ist": expiry_time,
        },
    )
    repo.save_hypothesis(hypo)

    eval_dates = pd.date_range(start_date, end_date, freq="B").date

    current_equity = initial_equity
    daily_equity_records = {}
    trade_pnls = []
    executed_chains = []
    daily_chart_records = []
    skipped_tie = 0
    skipped_no_data = 0

    print(f"\n{'Date':<11} | {'Eval':<8} | {'Range [L-H]':<19} | {'Dir':<5} | {'Entry':<8} | {'Leg 1':<22} | {'Leg 2 (Flip)':<22} | {'Net PnL':<9}")
    print("-" * 130)

    for target_date in eval_dates:
        risk_service.reset_daily_circuit_breaker()
        window = session_engine.get_session_window(target_date)

        day_df = df[(df["utc_time"] >= window.session_start_utc) & (df["utc_time"] <= window.eod_utc)]
        prior_bars = day_df[
            (day_df["utc_time"] >= window.session_start_utc) & (day_df["utc_time"] < window.eval_time_utc)
        ]
        eval_candle = day_df[day_df["utc_time"] == window.eval_time_utc]

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

        time_1100_utc = window.eval_time_utc - dt.timedelta(hours=1, minutes=30)
        candle_1100 = day_df[day_df["utc_time"] == time_1100_utc]
        exact_price_1100 = float(candle_1100.iloc[0]["open"]) if len(candle_1100) > 0 else None

        if len(prior_bars) < 15 or len(eval_candle) == 0:
            skipped_no_data += 1
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
                "price_1100": exact_price_1100,
                "price_1230": 0,
                "dist_to_low": 0,
                "dist_to_high": 0,
                "pnl_net": 0.0,
                "outcome_summary": "Skipped: insufficient market bars in session window.",
                "legs": [],
                "events": [],
                "bars": bar_records,
                "sessions": {
                    "session_start": int(window.session_start_utc.timestamp()),
                    "sydney_open": int(window.session_start_utc.timestamp()),
                    "session_name": window.session_name,
                    "eval_time": int(window.eval_time_utc.timestamp()),
                    "eval_1230": int(window.eval_time_utc.timestamp()),
                    "eval_1100": int(time_1100_utc.timestamp()),
                    "time_1100": int(time_1100_utc.timestamp()),
                    "london_close": int(dt.datetime(target_date.year, target_date.month, target_date.day, 16, 30, tzinfo=dt.timezone.utc).timestamp()),
                    "eod": int(window.eod_utc.timestamp()),
                },
            })
            continue

        signal = strategy.evaluate_day(target_date, day_df, account_equity=current_equity)
        if signal is None:
            eval_p = float(eval_candle.iloc[0]["open"])
            r_high = max(float(prior_bars["high"].max()), eval_p)
            r_low = min(float(prior_bars["low"].min()), eval_p)
            dist_low = eval_p - r_low
            dist_high = r_high - eval_p
            is_tie = abs(dist_low - dist_high) <= strategy.config.tie_tolerance
            if is_tie:
                skipped_tie += 1
            daily_equity_records[target_date] = current_equity

            daily_chart_records.append({
                "date": str(target_date),
                "day_type": "TIE" if is_tie else "SKIPPED",
                "day_of_week": target_date.strftime("%A"),
                "range_high": r_high,
                "range_low": r_low,
                "midpoint": (r_high + r_low) / 2.0,
                "range_pts": r_high - r_low,
                "eval_price": eval_p,
                "price_1100": exact_price_1100,
                "price_1230": eval_p,
                "dist_to_low": dist_low,
                "dist_to_high": dist_high,
                "tie_detected": is_tie,
                "pnl_net": 0.0,
                "outcome_summary": f"No trade — Proximity tie at 12:30 IST evaluation (Dist to Low: {dist_low:.2f} pts vs Dist to High: {dist_high:.2f} pts)." if is_tie else "Skipped by risk management.",
                "legs": [],
                "events": [],
                "bars": bar_records,
                "sessions": {
                    "session_start": int(window.session_start_utc.timestamp()),
                    "sydney_open": int(window.session_start_utc.timestamp()),
                    "session_name": window.session_name,
                    "eval_time": int(window.eval_time_utc.timestamp()),
                    "eval_1230": int(window.eval_time_utc.timestamp()),
                    "eval_1100": int(time_1100_utc.timestamp()),
                    "time_1100": int(time_1100_utc.timestamp()),
                    "limit_cancel": int(window.limit_cancel_utc.timestamp()),
                    "london_close": int(dt.datetime(target_date.year, target_date.month, target_date.day, 16, 30, tzinfo=dt.timezone.utc).timestamp()),
                    "eod": int(window.eod_utc.timestamp()),
                },
            })
            continue

        chain_id = f"RSV2-{target_date.strftime('%Y%m%d')}"
        chain = TradeChain(
            chain_id=chain_id,
            config=chain_config,
            risk_service=risk_service,
            account_equity=current_equity,
        )
        chain.computed_context = {
            "target_date": target_date,
            "session_name": window.session_name,
            "range_high": signal.range_high,
            "range_low": signal.range_low,
            "midpoint": signal.midpoint,
            "range_pts": signal.range_pts,
            "eval_price": signal.eval_price,
            "dist_to_low": signal.dist_to_low,
            "dist_to_high": signal.dist_to_high,
            "decay_points": decay,
            "ctc1_points": ctc1,
            "ctc2_points": ctc2,
            "enable_flip": enable_flip,
            "expiry_time": expiry_time,
        }

        chain.start_leg1(
            direction=signal.direction,
            entry_time=signal.eval_time_utc,
            entry_price=signal.entry_price,
            base_units=signal.units,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            eval_price=signal.eval_price,
        )

        forward_bars = day_df[(day_df["utc_time"] >= signal.eval_time_utc) & (day_df["utc_time"] <= window.eod_utc)]
        for _, row in forward_bars.iterrows():
            # If pending and this is the 12:30 bar, skip (limit order placed at 12:30, fills on subsequent movement)
            if chain.current_leg and chain.current_leg.is_pending and row["utc_time"] == signal.eval_time_utc:
                continue
            simulator.process_bar(
                chain=chain,
                bar_time=row["utc_time"],
                open_p=row["open"],
                high_p=row["high"],
                low_p=row["low"],
                close_p=row["close"],
                eval_price=signal.eval_price,
                limit_cancel_utc=signal.limit_cancel_utc,
            )
            if chain.status == ChainStatus.COMPLETED or chain.current_leg is None:
                break

        active_leg = chain.current_leg
        if active_leg and active_leg.is_open:
            if active_leg.is_pending:
                active_leg.exit_reason = "EXPIRATION"
                active_leg.is_open = False
            else:
                last_bar = forward_bars.iloc[-1] if len(forward_bars) > 0 else prior_bars.iloc[-1]
                simulator._close_leg(
                    chain=chain,
                    leg=active_leg,
                    bar_time=last_bar["utc_time"],
                    fill_price=float(last_bar["close"]),
                    reason="EXPIRATION",
                )
                chain.status = ChainStatus.COMPLETED

        repo.save_trade_chain(chain)
        executed_chains.append(chain)
        current_equity += chain.total_pnl_net
        daily_equity_records[target_date] = current_equity

        l1 = chain.legs[0] if len(chain.legs) > 0 else None
        l2 = chain.legs[1] if len(chain.legs) > 1 else None

        def leg_desc(leg):
            if leg is None:
                return "—"
            if leg.is_pending and not leg.exit_price:
                return "UNFILLED"
            if leg.exit_price is None:
                return "OPEN"
            return f"{leg.exit_reason} (${leg.pnl_net:+.1f})"

        range_str = f"[{signal.range_low:.1f}-{signal.range_high:.1f}]"
        l2_desc = leg_desc(l2)
        if l2 and l2.exit_price:
            l2_desc = f"{l2.direction} {l2_desc}"

        print(
            f"{target_date} | ${signal.eval_price:<7.2f} | {range_str:<19} | "
            f"{signal.direction:<5} | ${signal.entry_price:<7.2f} | {leg_desc(l1):<22} | "
            f"{l2_desc:<22} | ${chain.total_pnl_net:+<8.2f}"
        )

        for leg in chain.legs:
            if not leg.is_open and leg.exit_price is not None and leg.exit_reason not in ("EXPIRATION", "DECAY"):
                trade_pnls.append(leg.pnl_net)
            elif not leg.is_open and leg.exit_reason == "CTC":
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
        if l1 and l1.exit_reason == "DECAY":
            day_status = "DECAY"
        elif l1 and l1.is_pending and l1.exit_reason == "EXPIRATION":
            day_status = "EXPIRED"

        daily_chart_records.append({
            "date": str(target_date),
            "day_type": day_status,
            "day_of_week": target_date.strftime("%A"),
            "range_high": signal.range_high,
            "range_low": signal.range_low,
            "midpoint": signal.midpoint,
            "range_pts": signal.range_pts,
            "eval_price": signal.eval_price,
            "price_1100": exact_price_1100,
            "price_1230": signal.eval_price,
            "dist_to_low": signal.dist_to_low,
            "dist_to_high": signal.dist_to_high,
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
                "session_start": int(window.session_start_utc.timestamp()),
                "sydney_open": int(window.session_start_utc.timestamp()),
                "session_name": window.session_name,
                "eval_time": int(window.eval_time_utc.timestamp()),
                "eval_1230": int(window.eval_time_utc.timestamp()),
                "eval_1100": int(time_1100_utc.timestamp()),
                "time_1100": int(time_1100_utc.timestamp()),
                "limit_cancel": int(window.limit_cancel_utc.timestamp()),
                "london_close": int(dt.datetime(target_date.year, target_date.month, target_date.day, 16, 30, tzinfo=dt.timezone.utc).timestamp()),
                "eod": int(window.eod_utc.timestamp()),
            },
        })

    equity_series = pd.Series(daily_equity_records)
    portfolio_engine = PortfolioEngine(initial_equity=initial_equity)
    metrics = portfolio_engine.compute_metrics(equity_series, trade_pnls)

    print("-" * 130)
    print("\n" + "=" * 80)
    print("RANGE SWEEP V2 — BACKTEST SUMMARY (AUG 4 – SEP 7, 2026):")
    print(f"  Trading Days in Range     : {len(eval_dates)}")
    print(f"  Days Traded (signals)     : {len(executed_chains)}")
    print(f"  Days Skipped (tie)        : {skipped_tie}")
    print(f"  Days Skipped (no data)    : {skipped_no_data}")
    print(f"  Total Trades (Legs)       : {metrics.total_trades}")
    print(f"  Winning Trades            : {metrics.winning_trades} ({metrics.win_rate_pct:.1f}%)")
    print(f"  Losing Trades             : {metrics.losing_trades}")
    print(f"  Initial Balance           : ${metrics.initial_equity:,.2f}")
    print(f"  Final Balance             : ${metrics.final_equity:,.2f}")
    print(f"  Total Net Profit          : ${metrics.total_net_pnl:+,.2f} ({metrics.total_return_pct:+.2f}%)")
    print(f"  Sharpe Ratio              : {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio             : {metrics.sortino_ratio:.2f}")
    print(f"  Profit Factor             : {metrics.profit_factor:.2f}")
    print(f"  Max Drawdown              : -{metrics.max_drawdown_pct:.2f}% (${metrics.max_drawdown_usd:,.2f})")
    print(f"  Calmar Ratio              : {metrics.calmar_ratio:.2f}")
    print("=" * 80)

    flips = [c for c in executed_chains if len(c.legs) > 1]
    ctc_exits = [c for c in executed_chains for leg in c.legs if leg.exit_reason == "CTC"]
    decay_cancels = [c for c in executed_chains for leg in c.legs if leg.exit_reason == "DECAY"]
    expiry_cancels = [
        c for c in executed_chains
        for leg in c.legs
        if leg.exit_reason == "EXPIRATION" and leg.is_pending is False and leg.pnl_net == 0 and leg.depth == 1
    ]

    print("\nRANGE SWEEP V2 — RULE TRIGGER STATS:")
    print(f"  SL-Flip (Leg 2) Triggered     : {len(flips)}")
    print(f"  CTC Breakeven Exits           : {len(ctc_exits)}")
    print(f"  Decay Cancels (unfilled)      : {len(decay_cancels)}")
    if flips:
        flip_wins = sum(1 for c in flips if len(c.legs) > 1 and c.legs[1].pnl_net > 0)
        flip_pnl = sum(c.legs[1].pnl_net for c in flips if len(c.legs) > 1)
        print(f"  Leg 2 Wins                    : {flip_wins}/{len(flips)}")
        print(f"  Leg 2 Net PnL                 : ${flip_pnl:+,.2f}")
    print("=" * 80)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_file = ReportVisualizer.generate_html_report(
        hypothesis_title=hypo.title,
        hypothesis_id=hypo.id,
        metrics=metrics,
        daily_equity_series=equity_series,
        executed_chains=executed_chains,
        daily_chart_data=daily_chart_records,
        output_filepath=report_path,
    )
    print(f"\n[REPORT GENERATED] Interactive HTML report saved to:")
    print(f"  file://{report_file}")

    return {
        "report_file": str(report_file),
        "metrics": metrics,
        "executed_chains": executed_chains,
        "daily_chart_records": daily_chart_records,
        "daily_equity_series": equity_series,
        "hypothesis": hypo,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run 12:30 Range Sweep V2 Backtest with custom inputs & parameters."
    )
    parser.add_argument("--session-start", type=str, default="sydney", choices=["sydney", "tokyo"], help="Session interval start: 'sydney' or 'tokyo' (default: sydney)")
    parser.add_argument("--x", type=float, default=3.5, help="X-offset (points) from high/low for Leg 1 limit order (default: 3.5)")
    parser.add_argument("--sl1", type=float, default=10.0, help="Stop loss in points for Leg 1 (default: 10.0)")
    parser.add_argument("--tp1", type=float, default=20.0, help="Take profit in points for Leg 1 (default: 20.0)")
    parser.add_argument("--ctc1", type=float, default=10.0, help="CTC activation threshold in points for Leg 1 (default: 10.0)")
    parser.add_argument("--ctc2", type=float, default=10.0, help="CTC activation threshold in points for Leg 2 flip (default: 10.0)")
    parser.add_argument("--ctc", type=float, default=None, help="Shorthand to set both ctc1 and ctc2 to this value")
    parser.add_argument("--decay", type=float, default=15.0, help="Decay threshold in points for canceling unfilled Leg 1 (default: 15.0)")
    parser.add_argument("--sl2", type=float, default=10.0, help="Stop loss in points for Leg 2 flip (default: 10.0)")
    parser.add_argument("--tp2", type=float, default=20.0, help="Take profit in points for Leg 2 flip (default: 20.0)")
    parser.add_argument("--expiry", type=str, default="17:00", help="IST time cutoff for unfilled limit order (default: 17:00)")
    parser.add_argument("--enable-flip", dest="enable_flip", action="store_true", default=True, help="Enable reverse follow-back trade on SL hit (default: True)")
    parser.add_argument("--no-flip", dest="enable_flip", action="store_false", help="Disable reverse follow-back trade on SL hit")
    parser.add_argument("--units", type=float, default=10.0, help="Position size in units/ounces (default: 10.0)")
    parser.add_argument("--start", type=str, default="2026-08-04", help="Start date YYYY-MM-DD (default: 2026-08-04)")
    parser.add_argument("--end", type=str, default="2026-09-07", help="End date YYYY-MM-DD (default: 2026-09-07)")
    parser.add_argument("--equity", type=float, default=50000.0, help="Initial equity balance in USD (default: 50000.0)")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_PARQUET_PATH), help="Path to 5-minute parquet file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_REPORT_PATH), help="Output HTML report path")

    args = parser.parse_args()
    ctc1_val = args.ctc if args.ctc is not None else args.ctc1
    ctc2_val = args.ctc if args.ctc is not None else args.ctc2

    run_range_sweep_v2_backtest(
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

