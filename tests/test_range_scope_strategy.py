"""
Unit tests for Range Scope Strategy Version 1 (11:00 AM IST Strategy).
"""

import datetime as dt
import pandas as pd
import pytest

from src.strategies.range_scope_v1 import RangeScopeConfig, RangeScopeStrategy, RangeScopeSignal
from src.execution.range_scope_simulator import RangeScopeSimulator, RangeScopeTradeResult


def _generate_synthetic_day_bars(
    target_date: dt.date,
    start_utc_hour: int = 21,    # Previous day 21:00 UTC (07:00 Sydney)
    end_utc_hour: int = 16,      # Same day 16:00 UTC (21:30 IST)
    base_price: float = 2500.0,
    price_pattern: str = "drop_to_low",
) -> pd.DataFrame:
    """Generates synthetic 5m bars for testing."""
    records = []
    # Start at previous day 21:00 UTC
    start_dt = dt.datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=dt.timezone.utc) - dt.timedelta(hours=3)
    end_dt = dt.datetime(target_date.year, target_date.month, target_date.day, end_utc_hour, 0, tzinfo=dt.timezone.utc)

    curr = start_dt
    while curr <= end_dt:
        hour_utc = curr.hour
        minute_utc = curr.minute

        # Default prices
        p = base_price
        h = p
        l = p

        if curr <= dt.datetime(target_date.year, target_date.month, target_date.day, 5, 30, tzinfo=dt.timezone.utc):
            # Session range formation up to 05:30 UTC (11:00 AM IST)
            if price_pattern == "drop_to_low":
                if hour_utc == 2:
                    p, h, l = 2520.0, 2520.0, 2520.0
                elif hour_utc == 5 and minute_utc == 0:
                    p, h, l = 2480.0, 2480.0, 2480.0
                elif hour_utc == 5 and minute_utc == 30:
                    p, h, l = 2488.0, 2488.0, 2488.0
                else:
                    p, h, l = 2500.0, 2500.0, 2500.0
            elif price_pattern == "rise_to_high":
                if hour_utc == 2:
                    p, h, l = 2480.0, 2480.0, 2480.0
                elif hour_utc == 5 and minute_utc == 0:
                    p, h, l = 2520.0, 2520.0, 2520.0
                elif hour_utc == 5 and minute_utc == 30:
                    p, h, l = 2512.0, 2512.0, 2512.0
                else:
                    p, h, l = 2500.0, 2500.0, 2500.0
            elif price_pattern == "tie":
                if hour_utc == 1:
                    p, h, l = 2520.0, 2520.0, 2520.0
                elif hour_utc == 3:
                    p, h, l = 2480.0, 2480.0, 2480.0
                elif hour_utc == 5 and minute_utc == 30:
                    p, h, l = 2500.0, 2500.0, 2500.0
                else:
                    p, h, l = 2500.0, 2500.0, 2500.0
            elif price_pattern == "tight_scope":
                if hour_utc == 2:
                    p, h, l = 2520.0, 2520.0, 2520.0
                elif hour_utc == 4:
                    p, h, l = 2498.0, 2498.0, 2498.0
                elif hour_utc == 5 and minute_utc == 30:
                    p, h, l = 2500.0, 2500.0, 2500.0
                else:
                    p, h, l = 2510.0, 2510.0, 2510.0
        else:
            # Forward action after 11:00 AM IST (05:30 UTC)
            if price_pattern == "drop_to_low":
                # Drops to 2478 at 06:15 UTC (reaches range_low 2480 before 12:30!)
                if hour_utc == 6 and minute_utc == 15:
                    p, h, l = 2478.0, 2480.0, 2478.0
                elif hour_utc == 7 and minute_utc == 0:
                    p, h, l = 2482.0, 2482.0, 2482.0
                else:
                    p, h, l = 2485.0, 2485.0, 2485.0
            elif price_pattern == "drop_after_1230":
                if hour_utc == 6:
                    p, h, l = 2485.0, 2485.0, 2485.0
                elif hour_utc == 7 and minute_utc == 0:
                    p, h, l = 2482.0, 2482.0, 2482.0
                else:
                    p, h, l = 2480.0, 2480.0, 2480.0
            else:
                p, h, l = 2500.0, 2500.0, 2500.0

        records.append({
            "timestamp": curr,
            "open": p,
            "high": h,
            "low": l,
            "close": p,
            "volume": 100,
        })
        curr += dt.timedelta(minutes=5)

    return pd.DataFrame(records)


def test_range_scope_direction_and_scope_gate():
    """Tests 11:00 AM IST bias determination and scope filter."""
    target_date = dt.date(2026, 8, 5)
    strategy = RangeScopeStrategy(RangeScopeConfig(scope_min_x=5.0))

    # 1. Drop to low pattern: Range [2480 - 2520], Eval at 2488.
    # Dist to low = 8.0 pts, Dist to high = 32.0 pts.
    # Closer to low -> SHORT. Scope = 8.0 >= 5.0 -> VALID.
    df_short = _generate_synthetic_day_bars(target_date, price_pattern="drop_to_low")
    signal, ctx = strategy.evaluate_day_context(target_date, df_short)
    assert signal is not None
    assert signal.direction == "SHORT"
    assert signal.scope_pts == 8.0
    assert signal.entry_price == 2488.0
    assert signal.sl_price == 2498.0
    assert signal.initial_tp == 2480.0

    # 2. Rise to high pattern: Range [2480 - 2520], Eval at 2512.
    # Dist to high = 8.0 pts, Dist to low = 32.0 pts.
    # Closer to high -> LONG. Scope = 8.0 >= 5.0 -> VALID.
    df_long = _generate_synthetic_day_bars(target_date, price_pattern="rise_to_high")
    signal_long, ctx_long = strategy.evaluate_day_context(target_date, df_long)
    assert signal_long is not None
    assert signal_long.direction == "LONG"
    assert signal_long.scope_pts == 8.0
    assert signal_long.entry_price == 2512.0
    assert signal_long.sl_price == 2502.0
    assert signal_long.initial_tp == 2520.0

    # 3. Tight scope pattern: Range [2498 - 2520], Eval at 2500.
    # Scope to low is 2.0 < 5.0 -> Should skip trade.
    df_tight = _generate_synthetic_day_bars(target_date, price_pattern="tight_scope")
    signal_tight, ctx_tight = strategy.evaluate_day_context(target_date, df_tight)
    assert signal_tight is None
    assert ctx_tight["reason"] == "insufficient_scope"
    assert ctx_tight["scope"] == 2.0

    # 4. Tie pattern: Range [2480 - 2520], Midpoint 2500, Eval at 2500.
    # Dist to low = 20.0, Dist to high = 20.0 -> TIE.
    df_tie = _generate_synthetic_day_bars(target_date, price_pattern="tie")
    signal_tie, ctx_tie = strategy.evaluate_day_context(target_date, df_tie)
    assert signal_tie is None
    assert ctx_tie["reason"] == "tie"


def test_range_scope_mode_1_tp_before_1230():
    """Tests Mode 1 taking profit at range boundary before 12:30 IST."""
    target_date = dt.date(2026, 8, 5)
    config = RangeScopeConfig(tp_mode="mode_1_dynamic")
    strategy = RangeScopeStrategy(config)
    simulator = RangeScopeSimulator()

    df = _generate_synthetic_day_bars(target_date, price_pattern="drop_to_low")
    signal, ctx = strategy.evaluate_day_context(target_date, df)
    assert signal is not None

    session_start_utc = ctx["window"].session_start_utc
    trade = simulator.simulate_day(signal, df, session_start_utc)

    # Reached 2478 at 06:15 UTC (before 12:30 / 07:00 UTC) -> TP hit at 2480
    assert trade.exit_reason == "TP"
    assert trade.exit_price == 2480.0
    assert trade.pnl_points == 8.0  # 2488 entry - 2480 exit = +8 pts
    assert trade.pnl_net > 0


def test_range_scope_mode_2_wait_until_1230():
    """Tests Mode 2 waiting until 12:30 with no early TP."""
    target_date = dt.date(2026, 8, 5)
    config = RangeScopeConfig(tp_mode="mode_2_wait_1230", tp_offset_y=3.5)
    strategy = RangeScopeStrategy(config)
    simulator = RangeScopeSimulator()

    df = _generate_synthetic_day_bars(target_date, price_pattern="drop_to_low")
    signal, ctx = strategy.evaluate_day_context(target_date, df)
    assert signal is not None
    assert signal.tp_mode == "mode_2_wait_1230"

    session_start_utc = ctx["window"].session_start_utc
    trade = simulator.simulate_day(signal, df, session_start_utc)

    # In Mode 2, even though 2480 was touched at 06:15 UTC, it was NOT closed at 06:15 because no TP was active!
    # At 12:30 (07:00 UTC), it evaluated expanded range and price_1230!
    assert trade.price_at_1230 is not None
    assert trade.exit_time >= dt.datetime(target_date.year, target_date.month, target_date.day, 7, 0, tzinfo=dt.timezone.utc)


def test_range_scope_no_lookahead_leakage_on_1100_bar():
    """Tests that the 11:00-11:05 bar's future low/high does not corrupt the 11:00 evaluation range."""
    target_date = dt.date(2026, 8, 11)
    config = RangeScopeConfig(session_start="tokyo", scope_min_x=5.0)
    strategy = RangeScopeStrategy(config)

    # Synthetic bars: Tokyo start 05:30 IST (00:00 UTC) to 10:55 IST (05:25 UTC) has low 4390.0, high 4435.0
    # 11:00 IST (05:30 UTC) candle opens at 4392.0 and plunges to 4372.0 (future low)
    records = []
    start_utc = dt.datetime(2026, 8, 11, 0, 0, tzinfo=dt.timezone.utc)
    curr = start_utc
    while curr <= dt.datetime(2026, 8, 11, 16, 0, tzinfo=dt.timezone.utc):
        if curr < dt.datetime(2026, 8, 11, 5, 30, tzinfo=dt.timezone.utc):
            if curr.hour == 2:
                o, h, l, c = 4435.0, 4435.0, 4435.0, 4435.0
            elif curr.hour == 5 and curr.minute == 25:
                o, h, l, c = 4390.0, 4390.0, 4390.0, 4390.0
            else:
                o, h, l, c = 4410.0, 4410.0, 4410.0, 4410.0
        elif curr == dt.datetime(2026, 8, 11, 5, 30, tzinfo=dt.timezone.utc):
            # 11:00 bar has massive intrabar plunge to 4372.0!
            o, h, l, c = 4392.0, 4392.0, 4372.0, 4380.0
        else:
            o, h, l, c = 4380.0, 4380.0, 4380.0, 4380.0
        records.append({"timestamp": curr, "open": o, "high": h, "low": l, "close": c, "volume": 100})
        curr += dt.timedelta(minutes=5)

    df = pd.DataFrame(records)
    signal, ctx = strategy.evaluate_day_context(target_date, df)

    # Range low MUST be 4390.0, NOT 4372.0!
    assert ctx["range_low"] == 4390.0
    assert ctx["range_high"] == 4435.0
    assert ctx["eval_price"] == 4392.0
    # Scope = 4392.0 - 4390.0 = 2.0 pts
    assert ctx["scope"] == 2.0
    # Since scope (2.0) < scope_min_x (5.0), trade must be skipped!
    assert signal is None
    assert ctx["reason"] == "insufficient_scope"


def test_range_scope_custom_eval_time_and_ctc():
    """Tests custom eval_time_ist and CTC breakeven arming / exit."""
    target_date = dt.date(2026, 8, 12)
    # Config with 10:30 AM IST eval time and ctc_points = 8.0
    config = RangeScopeConfig(eval_time_ist="10:30", ctc_points=8.0, scope_min_x=5.0)
    strategy = RangeScopeStrategy(config)
    simulator = RangeScopeSimulator()

    # Create synthetic bars where 10:30 IST is 05:00 UTC
    records = []
    start_utc = dt.datetime(2026, 8, 12, 0, 0, tzinfo=dt.timezone.utc)
    curr = start_utc
    while curr <= dt.datetime(2026, 8, 12, 16, 0, tzinfo=dt.timezone.utc):
        if curr < dt.datetime(2026, 8, 12, 5, 0, tzinfo=dt.timezone.utc):
            if curr.hour == 2:
                o, h, l, c = 2520.0, 2520.0, 2520.0, 2520.0
            elif curr.hour == 4:
                o, h, l, c = 2480.0, 2480.0, 2480.0, 2480.0
            else:
                o, h, l, c = 2500.0, 2500.0, 2500.0, 2500.0
        elif curr == dt.datetime(2026, 8, 12, 5, 0, tzinfo=dt.timezone.utc):
            # 10:30 AM IST (05:00 UTC) Eval bar opens at 2510 (nearer high 2520 -> LONG, scope=10.0)
            o, h, l, c = 2510.0, 2510.0, 2510.0, 2510.0
        elif curr == dt.datetime(2026, 8, 12, 5, 15, tzinfo=dt.timezone.utc):
            # Moves +8.5 pts in profit to 2518.5 -> arms CTC!
            o, h, l, c = 2515.0, 2518.5, 2514.0, 2517.0
        elif curr == dt.datetime(2026, 8, 12, 5, 20, tzinfo=dt.timezone.utc):
            # Reverses back down to 2509.0 -> hits stepped SL at entry (2510.0) -> CTC exit!
            o, h, l, c = 2515.0, 2515.0, 2509.0, 2509.5
        elif curr > dt.datetime(2026, 8, 12, 5, 0, tzinfo=dt.timezone.utc):
            # Forward prices stay near entry unless explicitly moved
            o, h, l, c = 2512.0, 2512.0, 2512.0, 2512.0
        else:
            o, h, l, c = 2500.0, 2500.0, 2500.0, 2500.0

        records.append({"timestamp": curr, "open": o, "high": h, "low": l, "close": c, "volume": 100})
        curr += dt.timedelta(minutes=5)

    df = pd.DataFrame(records)
    signal, ctx = strategy.evaluate_day_context(target_date, df)

    assert signal is not None
    assert signal.eval_time_ist == "10:30"
    assert signal.ctc_points == 8.0
    assert signal.direction == "LONG"
    assert signal.entry_price == 2510.0

    session_start_utc = ctx["window"].session_start_utc
    trade = simulator.simulate_day(signal, df, session_start_utc)

    # Verify CTC was armed and stepped SL at entry was triggered
    assert trade.ctc_armed is True
    assert trade.ctc_price == 2518.0  # 2510 + 8.0
    assert trade.exit_reason == "CTC"
    assert trade.exit_price == 2510.0
    assert trade.pnl_points == 0.0

