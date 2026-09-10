"""
Phase 2 Chained Strategy Historical Replay.
Replays continuous 1-minute historical gold bars from XAUUSD_1m.parquet,
testing Leg 1 (Sydney Range Fade) -> If SL -> Leg 2 (Reversal Flip) in live bar-by-bar simulation.
"""

import datetime as dt
import pandas as pd
from pathlib import Path

from src.session.engine import SessionEngine
from src.core.risk_service import RiskService, RiskServiceLimits
from src.strategies.sydney_range_fade import SydneyRangeFadeStrategy, SydneyRangeFadeConfig
from src.sequencer.fsm import TradeChain, ChainedFollowUpConfig, ChainStatus
from src.execution.simulator import ExecutionSimulator, FeeModel

PARQUET_PATH = Path("/Users/prince/algo-trading/data/historical/XAUUSD_1m.parquet")


def run_chained_historical_replay():
    print("=" * 80)
    print("HYPOTRADER: PHASE 2 CHAINED REPLAY (LEG 1 -> SL-FLIP -> LEG 2)")
    print(f"Reading dataset: {PARQUET_PATH}")
    print("=" * 80)

    if not PARQUET_PATH.exists():
        print(f"ERROR: Dataset not found at {PARQUET_PATH}")
        return

    df = pd.read_parquet(PARQUET_PATH)
    df["utc_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    session_engine = SessionEngine()
    risk_service = RiskService(RiskServiceLimits())
    strategy = SydneyRangeFadeStrategy(
        config=SydneyRangeFadeConfig(x_offset=3.5, sl_points=10.0, tp_points=18.0, base_units=10.0),
        session_engine=session_engine,
        risk_service=risk_service,
    )
    simulator = ExecutionSimulator(fee_model=FeeModel(), point_value=1.0)

    # Let's test 10 consecutive trading days in late April / early May 2024
    test_dates = [
        dt.date(2024, 4, 22),
        dt.date(2024, 4, 23),
        dt.date(2024, 4, 24),
        dt.date(2024, 4, 25),
        dt.date(2024, 4, 26),
        dt.date(2024, 4, 29),
        dt.date(2024, 4, 30),
        dt.date(2024, 5, 1),
        dt.date(2024, 5, 2),
        dt.date(2024, 5, 3),
    ]

    total_net_pnl = 0.0
    chains_count = 0
    flips_count = 0

    for target_date in test_dates:
        window = session_engine.get_session_window(target_date)

        # Slice data from Sydney open to end of trading day (e.g. 21:00 UTC / NY close)
        end_of_day_utc = dt.datetime(target_date.year, target_date.month, target_date.day, 21, 0, tzinfo=dt.timezone.utc)
        day_df = df[(df["utc_time"] >= window.sydney_open_utc) & (df["utc_time"] <= end_of_day_utc)]

        # 1. Evaluate at 12:30 IST
        signal = strategy.evaluate_day(target_date, day_df, account_equity=50000.0)
        if signal is None:
            continue

        chains_count += 1
        chain_id = f"CHAIN-{target_date.strftime('%Y%m%d')}"
        chain = TradeChain(
            chain_id=chain_id,
            config=ChainedFollowUpConfig(sl_flip_offset=8.0, tp_flip_offset=20.0, max_chain_depth=2),
            risk_service=risk_service,
            account_equity=50000.0,
        )

        # Start Leg 1
        leg1 = chain.start_leg1(
            direction=signal.direction,
            entry_time=signal.eval_time_utc,
            entry_price=signal.entry_price,
            base_units=signal.units,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
        )

        # 2. Simulate subsequent 1m bars after 12:30 IST
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

        # If trade remained open at end of day, close at last available price
        if chain.current_leg and chain.current_leg.is_open:
            last_bar = forward_bars.iloc[-1]
            simulator.process_bar(
                chain=chain,
                bar_time=last_bar["utc_time"],
                open_p=last_bar["open"],
                high_p=last_bar["high"],
                low_p=last_bar["low"],
                close_p=last_bar["close"],
            )

        # Report results for the day
        print(f"--- CHAIN: {chain_id} ({target_date.strftime('%A %Y-%m-%d')}) ---")
        for leg in chain.legs:
            exit_str = f"Exit: ${leg.exit_price:.2f} ({leg.exit_reason}) at {leg.exit_time.strftime('%H:%M')}" if leg.exit_price else "OPEN"
            print(f"  Leg {leg.depth} [{leg.direction} {leg.units:.1f}oz]: Entry: ${leg.entry_price:.2f} -> {exit_str} | Net PnL: ${leg.pnl_net:.2f}")

        if len(chain.legs) > 1:
            flips_count += 1
            print(f"  >> CONTINGENT SL-FLIP ACTIVATED! (Leg 1 SL triggered -> Leg 2 Reversal entered)")

        print(f"  Day Net PnL: ${chain.total_pnl_net:.2f}\n")
        total_net_pnl += chain.total_pnl_net

    print("=" * 80)
    print(f"SUMMARY: {chains_count} trading chains executed | {flips_count} SL-flips triggered")
    print(f"TOTAL SAMPLE NET PNL: ${total_net_pnl:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    run_chained_historical_replay()
