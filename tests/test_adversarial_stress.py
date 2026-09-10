"""
Adversarial Stress Tests for Chained Follow-Up Strategies and Isolated Risk Boundaries.
Verifies chain depth ceilings, circuit-breaker trips, martingale clamping, and lineage integrity.
"""

import datetime as dt
import pytest

from src.core.risk_service import RiskService, RiskServiceLimits
from src.sequencer.fsm import TradeChain, ChainedFollowUpConfig, ChainStatus
from src.execution.simulator import ExecutionSimulator, FeeModel
from src.stress.scenario_generator import ScenarioGenerator


def test_adversarial_max_chain_depth_exhaustion():
    """
    Subject the FSM to a brutal 3-stage whipsaw where every single leg is stopped out.
    Assert that the sequence halts strictly at max_chain_depth=3 and never spawns a 4th leg.
    """
    risk = RiskService(RiskServiceLimits(max_chain_depth=3, daily_loss_circuit_breaker_pct=20.0))
    chain = TradeChain(
        chain_id="STRESS-CHAIN-001",
        config=ChainedFollowUpConfig(sl_flip_offset=8.0, tp_flip_offset=20.0, max_chain_depth=3),
        risk_service=risk,
        account_equity=100000.0,
    )
    sim = ExecutionSimulator(fee_model=FeeModel(), point_value=1.0)

    # Start Leg 1 (Long at 2350.0, SL at 2340.0)
    now = dt.datetime(2025, 5, 1, 7, 0, tzinfo=dt.timezone.utc)
    leg1 = chain.start_leg1(
        direction="LONG",
        entry_time=now,
        entry_price=2350.0,
        base_units=10.0,
        sl_price=2340.0,
        tp_price=2370.0,
    )
    assert leg1 is not None
    assert chain.status == ChainStatus.ACTIVE

    # Generate double/triple whipsaw
    bars = ScenarioGenerator.generate_double_whipsaw(now, start_price=2350.0, sl1_distance=10.0, sl2_distance=8.0)

    for bar in bars:
        sim.process_bar(chain, bar.timestamp, bar.open, bar.high, bar.low, bar.close)

    # Assertions:
    # 1. Total legs recorded must be exactly 3 (L1, L2, L3)
    assert len(chain.legs) == 3
    assert chain.legs[0].leg_id == "STRESS-CHAIN-001-L1"
    assert chain.legs[1].leg_id == "STRESS-CHAIN-001-L2"
    assert chain.legs[2].leg_id == "STRESS-CHAIN-001-L3"

    # 2. Lineage integrity: L2 parent is L1, L3 parent is L2
    assert chain.legs[1].parent_trade_id == "STRESS-CHAIN-001-L1"
    assert chain.legs[2].parent_trade_id == "STRESS-CHAIN-001-L2"

    # 3. Direction alternates (Long -> Short -> Long)
    assert chain.legs[0].direction == "LONG"
    assert chain.legs[1].direction == "SHORT"
    assert chain.legs[2].direction == "LONG"

    # 4. Chain status must be COMPLETED, no 4th leg spawned
    assert chain.status == ChainStatus.COMPLETED
    assert chain.current_leg is None


def test_adversarial_circuit_breaker_halts_chain():
    """
    If a trade causes cumulative daily realized loss to reach 3.0%,
    the RiskService must intercept and BLOCK the contingent Leg 2 from spawning.
    """
    # Set daily circuit breaker to 1.5%
    risk = RiskService(RiskServiceLimits(daily_loss_circuit_breaker_pct=1.5))
    account_equity = 10000.0  # $10k account -> 1.5% is $150
    chain = TradeChain(
        chain_id="STRESS-CHAIN-002",
        config=ChainedFollowUpConfig(sl_flip_offset=8.0, tp_flip_offset=20.0),
        risk_service=risk,
        account_equity=account_equity,
    )
    sim = ExecutionSimulator()

    now = dt.datetime(2025, 5, 1, 7, 0, tzinfo=dt.timezone.utc)
    # 20 oz with 10 pt SL = $200 gross risk (> $150 circuit breaker)
    leg1 = chain.start_leg1(
        direction="LONG",
        entry_time=now,
        entry_price=2350.0,
        base_units=20.0,
        sl_price=2340.0,
        tp_price=2370.0,
    )
    assert leg1 is not None

    # Plunge to hit SL
    bar_sl = dt.datetime(2025, 5, 1, 7, 2, tzinfo=dt.timezone.utc)
    sim.process_bar(chain, bar_sl, 2345.0, 2345.0, 2338.0, 2339.0)

    # Leg 1 is closed at a loss
    assert leg1.is_open is False
    assert leg1.exit_reason == "SL"
    assert leg1.pnl_net < -150.0

    # RiskService circuit breaker has tripped!
    assert risk.is_circuit_breaker_active() is True

    # Leg 2 must be HALTED by risk
    assert chain.status == ChainStatus.HALTED_RISK
    assert len(chain.legs) == 1  # Only Leg 1 exists


def test_adversarial_martingale_multiplier_clamping():
    """
    If an external caller or optimizer attempts to request a 3x position multiplier
    on Leg 2, the RiskService and FSM must clamp it to max 1.0x.
    """
    risk = RiskService(RiskServiceLimits(max_size_multiplier_per_chain_step=1.0))
    chain = TradeChain(
        chain_id="STRESS-CHAIN-003",
        config=ChainedFollowUpConfig(
            sl_flip_offset=8.0,
            tp_flip_offset=20.0,
            requested_multiplier=3.0,  # 3x martingale intent
        ),
        risk_service=risk,
        account_equity=100000.0,
    )
    sim = ExecutionSimulator()

    now = dt.datetime(2025, 5, 1, 7, 0, tzinfo=dt.timezone.utc)
    leg1 = chain.start_leg1(
        direction="LONG",
        entry_time=now,
        entry_price=2350.0,
        base_units=10.0,
        sl_price=2340.0,
        tp_price=2370.0,
    )

    # Trigger SL on Leg 1
    t2 = now + dt.timedelta(minutes=1)
    sim.process_bar(chain, t2, 2345.0, 2345.0, 2339.0, 2339.5)

    assert len(chain.legs) == 2
    leg2 = chain.legs[1]
    # Units must NOT be 30.0 oz; it must be clamped to 10.0 oz!
    assert leg2.units == 10.0


def test_execution_simulator_ctc_breakeven_arm_and_exit():
    """
    Verifies that when a trade gains +10 points in profit:
    1. CTC arms and steps SL to breakeven (entry_price).
    2. When price subsequently pulls back to entry_price, the trade exits as 'CTC'.
    3. The chain completes without triggering Leg 2 (reversal flip).
    """
    risk = RiskService(RiskServiceLimits(max_chain_depth=2))
    chain = TradeChain(
        chain_id="CTC-TEST-001",
        config=ChainedFollowUpConfig(sl_flip_offset=10.0, tp_flip_offset=20.0, max_chain_depth=2),
        risk_service=risk,
        account_equity=50000.0,
    )
    sim = ExecutionSimulator(ctc_threshold_pts=10.0)

    now = dt.datetime(2026, 8, 3, 7, 0, tzinfo=dt.timezone.utc)
    leg1 = chain.start_leg1(
        direction="LONG",
        entry_time=now,
        entry_price=4050.0,
        base_units=10.0,
        sl_price=4040.0,
        tp_price=4070.0,
    )
    assert leg1 is not None
    assert leg1.ctc_armed is False
    assert leg1.sl_price == 4040.0

    # Bar 1: Price rallies to 4062.0 (+12 pts in profit, past the +10 pt CTC mark)
    t1 = now + dt.timedelta(minutes=5)
    sim.process_bar(chain, t1, open_p=4052.0, high_p=4062.0, low_p=4051.0, close_p=4060.0)

    assert leg1.ctc_armed is True
    assert leg1.sl_price == 4050.0  # Stepped to breakeven
    assert leg1.is_open is True

    # Bar 2: Price retraces back down to 4048.0 (hitting breakeven SL at 4050.0)
    t2 = now + dt.timedelta(minutes=10)
    sim.process_bar(chain, t2, open_p=4055.0, high_p=4055.0, low_p=4048.0, close_p=4049.0)

    assert leg1.is_open is False
    assert leg1.exit_reason == "CTC"
    assert len(chain.legs) == 1  # No Leg 2 flip triggered on CTC!
    assert chain.status == ChainStatus.COMPLETED

