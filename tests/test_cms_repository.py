"""
Unit tests for the DatabaseRepository and Hypothesis CMS.
Verifies lifecycle transitions, manual trade journaling, and relational trade logs.
"""

import datetime as dt
import pytest
from pathlib import Path

from src.cms.models import HypothesisCard, HypothesisStatus, ManualTradeRecord
from src.cms.repository import DatabaseRepository
from src.sequencer.fsm import TradeChain, ChainedFollowUpConfig
from src.core.risk_service import RiskService


def test_hypothesis_lifecycle_and_crud(tmp_path):
    db_file = tmp_path / "test_hypotrader.db"
    repo = DatabaseRepository(db_file)

    # 1. Create Hypothesis Card
    hypo = HypothesisCard(
        id="HYPO-GOLD-001",
        title="Gold Sydney-to-IST Range Fade",
        economic_rationale="Asian session liquidity sweep mean reversion",
        asset_symbol="XAUUSD",
        author="Prince",
        status=HypothesisStatus.DRAFT,
        target_regimes=["Low Volatility", "Non-FOMC"],
        param_manifest={"x_offset": [1.0, 5.0], "sl_points": [5.0, 20.0]},
    )
    repo.save_hypothesis(hypo)

    loaded = repo.get_hypothesis("HYPO-GOLD-001")
    assert loaded is not None
    assert loaded.title == "Gold Sydney-to-IST Range Fade"
    assert loaded.status == HypothesisStatus.DRAFT

    # 2. Transition through lifecycle
    repo.transition_hypothesis_status("HYPO-GOLD-001", HypothesisStatus.IN_BACKTEST)
    assert repo.get_hypothesis("HYPO-GOLD-001").status == HypothesisStatus.IN_BACKTEST

    repo.transition_hypothesis_status("HYPO-GOLD-001", HypothesisStatus.LIVE)
    assert repo.get_hypothesis("HYPO-GOLD-001").status == HypothesisStatus.LIVE

    repo.transition_hypothesis_status(
        "HYPO-GOLD-001", HypothesisStatus.RETIRED, retirement_reason="Alpha decayed in 2026 regime shift"
    )
    retired = repo.get_hypothesis("HYPO-GOLD-001")
    assert retired.status == HypothesisStatus.RETIRED
    assert "Alpha decayed" in retired.retirement_reason


def test_manual_trade_logging(tmp_path):
    db_file = tmp_path / "test_hypotrader.db"
    repo = DatabaseRepository(db_file)

    hypo = HypothesisCard(
        id="HYPO-GOLD-002",
        title="Manual Gold Discretionary Scalp",
        economic_rationale="Discretionary range fade",
        asset_symbol="XAUUSD",
        author="Prince",
    )
    repo.save_hypothesis(hypo)

    trade = ManualTradeRecord(
        id="MANUAL-20240501-01",
        hypothesis_id="HYPO-GOLD-002",
        date=dt.date(2024, 5, 1),
        direction="LONG",
        entry_time=dt.datetime(2024, 5, 1, 7, 0, tzinfo=dt.timezone.utc),
        entry_price=2300.50,
        units=10.0,
        exit_time=dt.datetime(2024, 5, 1, 10, 30, tzinfo=dt.timezone.utc),
        exit_price=2315.50,
        exit_reason="TP",
        pnl_net=145.0,
        execution_rating=5,
        notes="Clean bounce off Sydney low",
        tags=["manual", "forward_test"],
    )
    repo.save_manual_trade(trade)

    # Verify retrieval via raw query
    with repo._get_connection() as conn:
        row = conn.execute("SELECT * FROM manual_trades WHERE id = ?", ("MANUAL-20240501-01",)).fetchone()
        assert row is not None
        assert row["pnl_net"] == 145.0
        assert row["execution_rating"] == 5


def test_chained_trade_logs_relational_link(tmp_path):
    db_file = tmp_path / "test_hypotrader.db"
    repo = DatabaseRepository(db_file)

    chain = TradeChain(
        chain_id="CHAIN-RELATIONAL-001",
        config=ChainedFollowUpConfig(sl_flip_offset=8.0, tp_flip_offset=20.0),
        risk_service=RiskService(),
    )
    now = dt.datetime(2025, 6, 1, 7, 0, tzinfo=dt.timezone.utc)
    leg1 = chain.start_leg1("LONG", now, 2350.0, 10.0, 2340.0, 2370.0)
    leg2 = chain.handle_leg_exit(leg1, now + dt.timedelta(minutes=5), 2340.0, "SL", 0.7, 1.2, 0.3)

    assert leg2 is not None
    repo.save_trade_chain(chain)

    with repo._get_connection() as conn:
        rows = conn.execute("SELECT * FROM trade_logs WHERE chain_id = ? ORDER BY depth", ("CHAIN-RELATIONAL-001",)).fetchall()
        assert len(rows) == 2
        assert rows[0]["depth"] == 1
        assert rows[0]["parent_trade_id"] is None
        assert rows[1]["depth"] == 2
        assert rows[1]["parent_trade_id"] == rows[0]["id"]
