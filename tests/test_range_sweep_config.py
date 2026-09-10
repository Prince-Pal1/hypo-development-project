"""
Unit tests for Range Sweep V2:
  1. Bug 1: Unfilled limit order on 2026-08-06 decays / expires at 17:00 IST ($0.00 PnL).
  2. Bug 2: 2026-08-05 12:30 candle open eval price produces instantaneous short entry and TP1 hit.
  3. Session Start: Tokyo (09:00 JST / 05:30 IST) vs Sydney (07:00 Sydney).
  4. Toggleable Flip: enable_flip=False terminates trade at SL1 without spawning Leg 2.
  5. Independent CTC: ctc1 and ctc2 trigger at their distinct respective point thresholds.
"""

import datetime as dt
from pathlib import Path
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from src.session.engine import SessionEngine, TZ_TOKYO, TZ_SYDNEY, TZ_UTC, TZ_IST
from src.core.risk_service import RiskService
from src.strategies.range_sweep_v2 import RangeSweepV2Strategy, RangeSweepV2Config
from src.sequencer.fsm import TradeChain, ChainedFollowUpConfig, ChainStatus
from src.execution.range_sweep_v2_simulator import RangeSweepV2Simulator, RangeSweepV2SimConfig

PARQUET_PATH = Path("/Users/prince/algo-trading/data/historical/XAUUSD_5m.parquet")


@pytest.fixture(scope="module")
def gold_df():
    df = pd.read_parquet(PARQUET_PATH)
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["utc_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["utc_time"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_bug1_2026_08_06_decay_expiry_no_loss(gold_df):
    """Bug 1: 2026-08-06 should have NO trade executed and $0.00 PnL (decayed or expired)."""
    target_date = dt.date(2026, 8, 6)
    se = SessionEngine(session_start="sydney")
    window = se.get_session_window(target_date)
    day_df = gold_df[(gold_df["utc_time"] >= window.session_start_utc) & (gold_df["utc_time"] <= window.eod_utc)]

    strat = RangeSweepV2Strategy(RangeSweepV2Config(), session_engine=se)
    signal = strat.evaluate_day(target_date, day_df)
    assert signal is not None
    assert signal.direction == "LONG"
    assert round(signal.range_low, 1) == 4245.3
    assert round(signal.range_high, 1) == 4303.7
    assert round(signal.entry_price, 1) == 4248.8

    chain = TradeChain("TEST-0806", ChainedFollowUpConfig(), RiskService())
    chain.start_leg1(signal.direction, signal.eval_time_utc, signal.entry_price, 10.0, signal.sl_price, signal.tp_price, signal.eval_price)

    sim = RangeSweepV2Simulator(config=RangeSweepV2SimConfig(decay_points=15.0))
    forward_bars = day_df[(day_df["utc_time"] >= signal.eval_time_utc) & (day_df["utc_time"] <= window.eod_utc)]
    for _, row in forward_bars.iterrows():
        if chain.current_leg and chain.current_leg.is_pending and row["utc_time"] == signal.eval_time_utc:
            continue
        sim.process_bar(chain, row["utc_time"], row["open"], row["high"], row["low"], row["close"], signal.eval_price, signal.limit_cancel_utc)
        if chain.status == ChainStatus.COMPLETED or chain.current_leg is None:
            break

    # Verify no filled trade and $0.00 PnL
    assert chain.total_pnl_net == 0.0
    l1 = chain.legs[0]
    assert l1.exit_reason in ("DECAY", "EXPIRATION")
    assert l1.pnl_net == 0.0


def test_bug2_2026_08_05_instant_fill_and_tp1(gold_df):
    """Bug 2: 2026-08-05 should evaluate open of 12:30 candle, execute instantaneously at 12:30, and hit TP1."""
    target_date = dt.date(2026, 8, 5)
    se = SessionEngine(session_start="sydney")
    window = se.get_session_window(target_date)
    day_df = gold_df[(gold_df["utc_time"] >= window.session_start_utc) & (gold_df["utc_time"] <= window.eod_utc)]

    strat = RangeSweepV2Strategy(RangeSweepV2Config(x_offset=3.5, sl_points=10.0, tp_points=20.0), session_engine=se)
    signal = strat.evaluate_day(target_date, day_df)
    assert signal is not None
    assert signal.direction == "SHORT"
    assert round(signal.range_high, 1) == 4178.2
    assert round(signal.range_low, 1) == 4065.1
    # 12:30 candle open is 4177.665. max(4178.215 - 3.5, 4177.665) = 4177.665
    assert abs(signal.eval_price - 4177.665) < 0.01
    assert abs(signal.entry_price - 4177.665) < 0.01
    assert signal.is_instant is True

    chain = TradeChain("TEST-0805", ChainedFollowUpConfig(sl_flip_offset=10.0, tp_flip_offset=20.0), RiskService())
    chain.start_leg1(signal.direction, signal.eval_time_utc, signal.entry_price, 10.0, signal.sl_price, signal.tp_price, signal.eval_price)
    assert chain.legs[0].is_pending is False  # Instant fill at 12:30

    sim = RangeSweepV2Simulator(config=RangeSweepV2SimConfig(ctc1_points=10.0, decay_points=15.0))
    forward_bars = day_df[(day_df["utc_time"] >= signal.eval_time_utc) & (day_df["utc_time"] <= window.eod_utc)]
    for _, row in forward_bars.iterrows():
        sim.process_bar(chain, row["utc_time"], row["open"], row["high"], row["low"], row["close"], signal.eval_price, signal.limit_cancel_utc)
        if chain.status == ChainStatus.COMPLETED or chain.current_leg is None:
            break

    # Leg 1 should hit TP1 (+20 pts)
    assert len(chain.legs) == 1
    l1 = chain.legs[0]
    assert l1.exit_reason == "TP"
    assert l1.pnl_net > 190.0  # +$197.50 after IC Markets raw spread fees


def test_tokyo_session_start_calculation():
    """Verify Tokyo session interval: 09:00 JST -> 00:00 UTC -> 05:30 IST."""
    se = SessionEngine(session_start="tokyo")
    target_date = dt.date(2026, 8, 5)
    w = se.get_session_window(target_date)

    assert w.session_name == "tokyo"
    assert w.session_start_utc == dt.datetime(2026, 8, 5, 0, 0, tzinfo=TZ_UTC)
    assert w.eval_time_utc == dt.datetime(2026, 8, 5, 7, 0, tzinfo=TZ_UTC)
    assert w.duration_hours == 7.0


def test_enable_flip_toggle(gold_df):
    """Verify that when enable_flip=False, an SL1 exit terminates the chain without spawning Leg 2."""
    target_date = dt.date(2026, 8, 12)  # A day where SL1 was hit
    se = SessionEngine(session_start="sydney")
    window = se.get_session_window(target_date)
    day_df = gold_df[(gold_df["utc_time"] >= window.session_start_utc) & (gold_df["utc_time"] <= window.eod_utc)]

    strat = RangeSweepV2Strategy(RangeSweepV2Config(), session_engine=se)
    signal = strat.evaluate_day(target_date, day_df)
    assert signal is not None

    # With flip disabled
    chain_no_flip = TradeChain("TEST-NOFLIP", ChainedFollowUpConfig(enable_flip=False), RiskService())
    chain_no_flip.start_leg1(signal.direction, signal.eval_time_utc, signal.entry_price, 10.0, signal.sl_price, signal.tp_price, signal.eval_price)

    sim = RangeSweepV2Simulator(config=RangeSweepV2SimConfig(enable_flip=False))
    forward_bars = day_df[(day_df["utc_time"] >= signal.eval_time_utc) & (day_df["utc_time"] <= window.eod_utc)]
    for _, row in forward_bars.iterrows():
        if chain_no_flip.current_leg and chain_no_flip.current_leg.is_pending and row["utc_time"] == signal.eval_time_utc:
            continue
        sim.process_bar(chain_no_flip, row["utc_time"], row["open"], row["high"], row["low"], row["close"], signal.eval_price, signal.limit_cancel_utc)
        if chain_no_flip.status == ChainStatus.COMPLETED or chain_no_flip.current_leg is None:
            break

    # Must terminate with only 1 leg
    assert len(chain_no_flip.legs) == 1
    assert chain_no_flip.legs[0].exit_reason == "SL"
    assert chain_no_flip.status == ChainStatus.COMPLETED


def test_independent_ctc_thresholds():
    """Verify independent ctc1 and ctc2 thresholds in simulator."""
    sim = RangeSweepV2Simulator(config=RangeSweepV2SimConfig(ctc1_points=8.0, ctc2_points=14.0))
    assert sim.config.get_ctc_points(depth=1) == 8.0
    assert sim.config.get_ctc_points(depth=2) == 14.0


def test_configurable_limit_cancel_cutoff():
    """Verify configurable limit cancel cutoff times."""
    se = SessionEngine()
    se.set_limit_cancel_time_ist("15:30")
    w = se.get_session_window(dt.date(2026, 8, 5))
    # 15:30 IST is 10:00 UTC
    assert w.limit_cancel_utc == dt.datetime(2026, 8, 5, 10, 0, tzinfo=TZ_UTC)
