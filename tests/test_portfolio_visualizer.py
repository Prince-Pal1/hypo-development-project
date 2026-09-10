"""
Unit tests for PortfolioEngine, CorrelationEngine, and ReportVisualizer.
"""

import datetime as dt
import pandas as pd
import pytest
from pathlib import Path

from src.portfolio.portfolio import PortfolioEngine
from src.portfolio.correlation import CorrelationEngine
from src.reports.visualizer import ReportVisualizer
from src.sequencer.fsm import TradeChain, ChainedFollowUpConfig
from src.core.risk_service import RiskService


def test_portfolio_engine_metrics():
    engine = PortfolioEngine(initial_equity=50000.0)

    # 10 days of synthetic compounding equity
    dates = pd.date_range("2025-01-01", periods=10, freq="B").date
    equities = [50000.0, 50200.0, 50100.0, 50400.0, 50800.0, 50650.0, 51000.0, 51200.0, 51100.0, 51500.0]
    equity_series = pd.Series(equities, index=dates)
    trade_pnls = [200.0, -100.0, 300.0, 400.0, -150.0, 350.0, 200.0, -100.0, 400.0]

    metrics = engine.compute_metrics(equity_series, trade_pnls)

    assert metrics.final_equity == 51500.0
    assert metrics.total_net_pnl == 1500.0
    assert metrics.total_return_pct == 3.0
    assert metrics.total_trades == 9
    assert metrics.winning_trades == 6
    assert metrics.losing_trades == 3
    assert metrics.win_rate_pct == round(6 / 9 * 100, 2)
    assert metrics.sharpe_ratio > 0.0
    assert metrics.max_drawdown_pct > 0.0


def test_correlation_engine_diversification_gate():
    corr_engine = CorrelationEngine(max_allowed_correlation=0.30)

    dates = pd.date_range("2025-01-01", periods=20, freq="B")
    # Strategy 1 & Strategy 2: uncorrelated
    s1_returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.015] * 4, index=dates)
    s2_returns = pd.Series([-0.005, 0.01, -0.015, 0.02, -0.01] * 4, index=dates)

    report = corr_engine.compute_correlation({"Strat1": s1_returns, "Strat2": s2_returns})

    assert "Strat1" in report.correlation_matrix
    assert "Strat2" in report.correlation_matrix
    assert report.diversification_passed is True
    assert report.max_pairwise_correlation < 0.30


def test_report_visualizer_generation(tmp_path):
    out_html = tmp_path / "test_report.html"
    engine = PortfolioEngine(initial_equity=50000.0)

    dates = pd.date_range("2025-01-01", periods=5, freq="B").date
    equities = [50000.0, 50150.0, 50050.0, 50300.0, 50500.0]
    series = pd.Series(equities, index=dates)
    metrics = engine.compute_metrics(series, [150.0, -100.0, 250.0, 200.0])

    chain = TradeChain("TEST-CHAIN-01", ChainedFollowUpConfig(), RiskService())
    chain.start_leg1("LONG", dt.datetime(2025, 1, 1, 7, 0, tzinfo=dt.timezone.utc), 2350.0, 10.0, 2340.0, 2365.0)

    report_path = ReportVisualizer.generate_html_report(
        hypothesis_title="Gold Range Fade Validation",
        hypothesis_id="HYPO-GOLD-001",
        metrics=metrics,
        daily_equity_series=series,
        executed_chains=[chain],
        output_filepath=out_html,
    )

    assert Path(report_path).exists()
    content = Path(report_path).read_text(encoding="utf-8")
    assert "HYPO-GOLD-001" in content
    assert "Compounded Equity Curve" in content


def test_report_visualizer_daily_chart_data(tmp_path):
    out_html = tmp_path / "test_report_chart.html"
    engine = PortfolioEngine(initial_equity=50000.0)

    dates = pd.date_range("2025-01-01", periods=2, freq="B").date
    equities = [50000.0, 50200.0]
    series = pd.Series(equities, index=dates)
    metrics = engine.compute_metrics(series, [200.0])

    chain = TradeChain("TEST-CHAIN-02", ChainedFollowUpConfig(), RiskService())
    chain.start_leg1("LONG", dt.datetime(2025, 1, 1, 7, 0, tzinfo=dt.timezone.utc), 2350.0, 10.0, 2340.0, 2365.0)
    chain.record_event("LIMIT_FILLED", time=dt.datetime(2025, 1, 1, 7, 5, tzinfo=dt.timezone.utc), price=2350.0)
    chain.record_event("CTC_STEP", time=dt.datetime(2025, 1, 1, 8, 0, tzinfo=dt.timezone.utc), price=2360.0)
    summary = chain.generate_outcome_path_summary()
    assert "Long limit filled" in summary

    daily_chart_data = [
        {
            "date": "2025-01-01",
            "day_type": "TRADE",
            "day_of_week": "Wednesday",
            "range_high": 2360.0,
            "range_low": 2340.0,
            "midpoint": 2350.0,
            "range_pts": 20.0,
            "eval_price": 2348.0,
            "dist_to_low": 8.0,
            "dist_to_high": 12.0,
            "direction": "LONG",
            "entry_price": 2350.0,
            "sl_price": 2340.0,
            "tp_price": 2365.0,
            "pnl_net": 200.0,
            "outcome_summary": summary,
            "legs": [],
            "events": chain.events,
            "bars": [
                {"t": 1735714800, "o": 2348.0, "h": 2352.0, "l": 2347.0, "c": 2350.0, "v": 100},
                {"t": 1735715100, "o": 2350.0, "h": 2365.0, "l": 2349.0, "c": 2365.0, "v": 150},
            ],
            "sessions": {
                "sydney_open": 1735689600,
                "eval_time": 1735714800,
                "london_close": 1735749000,
                "eod": 1735765200,
            },
        }
    ]

    report_path = ReportVisualizer.generate_html_report(
        hypothesis_title="Gold Visualizer Daily Chart Test",
        hypothesis_id="HYPO-DAILY-TEST",
        metrics=metrics,
        daily_equity_series=series,
        executed_chains=[chain],
        output_filepath=out_html,
        daily_chart_data=daily_chart_data,
    )

    assert Path(report_path).exists()
    content = Path(report_path).read_text(encoding="utf-8")
    assert "dailyChartData" in content
    assert "renderSchematic" in content
    assert "renderCandles" in content
    assert "toggleFsmOverlay" in content
    assert "btnToggleFsmOverlay" in content
    assert "showFsmOverlay" in content
    assert "2025-01-01" in content
