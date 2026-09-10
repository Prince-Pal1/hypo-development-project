"""
Phase 1 Manual Verification Script.
Replays 5 real historical trading days from XAUUSD_1m.parquet through
the Session Engine, isolated RiskService, and single un-chained Leg strategy.
"""

import datetime as dt
import pandas as pd
from pathlib import Path

from src.session.engine import SessionEngine
from src.core.risk_service import RiskService, RiskServiceLimits
from src.strategies.sydney_range_fade import SydneyRangeFadeStrategy, SydneyRangeFadeConfig

PARQUET_PATH = Path("/Users/prince/algo-trading/data/historical/XAUUSD_1m.parquet")


def run_manual_verification():
    print("=" * 80)
    print("HYPOTRADER: PHASE 1 MANUAL HISTORICAL REPLAY & VERIFICATION")
    print(f"Reading dataset: {PARQUET_PATH}")
    print("=" * 80)

    if not PARQUET_PATH.exists():
        print(f"ERROR: Dataset not found at {PARQUET_PATH}")
        return

    # Load 1-minute gold bars
    df = pd.read_parquet(PARQUET_PATH)
    df["utc_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    print(f"Total dataset bars: {len(df):,}")
    print(f"Date range: {df['utc_time'].min()} to {df['utc_time'].max()}\n")

    session_engine = SessionEngine()
    risk_service = RiskService(RiskServiceLimits())
    strategy = SydneyRangeFadeStrategy(
        config=SydneyRangeFadeConfig(x_offset=3.5, sl_points=10.0, tp_points=18.0, base_units=10.0),
        session_engine=session_engine,
        risk_service=risk_service,
    )

    # Let's test 5 consecutive trading days across late April 2024
    test_dates = [
        dt.date(2024, 4, 22),
        dt.date(2024, 4, 23),
        dt.date(2024, 4, 24),
        dt.date(2024, 4, 25),
        dt.date(2024, 4, 26),
    ]

    for date in test_dates:
        window = session_engine.get_session_window(date)
        # Filter bars for the day's session window + some margin for evaluation
        margin_end = window.eval_time_utc + dt.timedelta(hours=2)
        day_df = df[(df["utc_time"] >= window.sydney_open_utc) & (df["utc_time"] <= margin_end)]

        signal = strategy.evaluate_day(date, day_df, account_equity=50000.0)

        print(f"--- TRADING DAY: {date} ({date.strftime('%A')}) ---")
        print(f"  Sydney DST (AEDT): {window.is_sydney_dst} (Window: {window.duration_hours:.1f} hours)")
        print(f"  Sydney Open (UTC): {window.sydney_open_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  12:30 IST (UTC)  : {window.eval_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")

        if signal is None:
            print("  [SIGNAL] No trade signal generated (insufficient data).")
        else:
            print(f"  Session Range    : High = ${signal.range_high:.2f} | Low = ${signal.range_low:.2f} | Width = ${signal.range_high - signal.range_low:.2f}")
            print(f"  Midpoint         : ${signal.midpoint:.2f}")
            print(f"  Eval Price (IST) : ${signal.eval_price:.2f}")
            dist_high = signal.range_high - signal.eval_price
            dist_low = signal.eval_price - signal.range_low
            print(f"  Proximity        : DistToHigh = ${dist_high:.2f}, DistToLow = ${dist_low:.2f} -> Bias = {signal.direction}")
            print(f"  Computed Entry   : ${signal.entry_price:.2f} (x_offset = 3.5 pts)")
            print(f"  Stop Loss        : ${signal.sl_price:.2f} (10.0 pts)")
            print(f"  Take Profit      : ${signal.tp_price:.2f} (18.0 pts)")
            print(f"  Risk Check       : {signal.risk_notes} | Sanctioned Units: {signal.units:.1f} oz")
        print()


if __name__ == "__main__":
    run_manual_verification()
