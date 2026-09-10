"""
Full End-to-End Demonstration of HypoTrader.
1. Registers the Gold Hypothesis in SQLite CMS.
2. Mines market condition setups from XAUUSD_1m.parquet.
3. Simulates Chained Follow-Up Trades (Leg 1 Range Fade -> If SL -> Leg 2 Reversal Flip).
4. Persists all relational records in hypotrader.db.
5. Computes Portfolio Metrics & Compounded Equity Trajectory.
6. Generates a standalone interactive HTML report (reports/hypotrader_gold_report.html).
"""

import datetime as dt
from pathlib import Path
import pandas as pd

from src.cms.models import HypothesisCard, HypothesisStatus
from src.cms.repository import DatabaseRepository
from src.session.engine import SessionEngine
from src.core.risk_service import RiskService, RiskServiceLimits
from src.strategies.sydney_range_fade import SydneyRangeFadeStrategy, SydneyRangeFadeConfig
from src.sequencer.fsm import TradeChain, ChainedFollowUpConfig, ChainStatus
from src.execution.simulator import ExecutionSimulator, FeeModel
from src.portfolio.portfolio import PortfolioEngine
from src.reports.visualizer import ReportVisualizer

PARQUET_PATH = Path("/Users/prince/algo-trading/data/historical/XAUUSD_1m.parquet")
DB_PATH = Path("hypotrader.db")
REPORT_DIR = Path("reports")


def main():
    print("=" * 80)
    print("HYPOTRADER: FULL END-TO-END PIPELINE RUN")
    print(f"Dataset: {PARQUET_PATH}")
    print(f"Database: {DB_PATH}")
    print("=" * 80)

    REPORT_DIR.mkdir(exist_ok=True)
    repo = DatabaseRepository(DB_PATH)

    # 1. Register Trading Hypothesis in CMS
    hypo_id = "HYPO-GOLD-001"
    hypo = HypothesisCard(
        id=hypo_id,
        title="Gold Mid-day Sydney-to-IST Range Fade & Reversal Flip",
        economic_rationale=(
            "Asian session liquidity probes extremes on low volume. At European pre-market open "
            "(12:30 PM IST), prices near the Sydney range boundary tend to mean-revert. If stops are "
            "swept with momentum (triggering SL), the stop-loss flip catches the continuation breakout."
        ),
        asset_symbol="XAUUSD",
        author="Prince",
        status=HypothesisStatus.IN_BACKTEST,
        target_regimes=["Normal Volatility", "Non-FOMC"],
        param_manifest={
            "x_offset": 3.5,
            "sl_points": 10.0,
            "tp_points": 18.0,
            "sl_flip_offset": 8.0,
            "tp_flip_offset": 20.0,
            "base_units": 10.0,
        },
    )
    repo.save_hypothesis(hypo)
    print(f"Registered Hypothesis in CMS: [{hypo.id}] {hypo.title}")

    # 2. Load 1m Gold Data
    df = pd.read_parquet(PARQUET_PATH)
    df["utc_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    session_engine = SessionEngine()
    risk_service = RiskService(RiskServiceLimits(
        max_size_multiplier_per_chain_step=1.0,
        max_chain_depth=2,
        daily_loss_circuit_breaker_pct=3.0,
    ))
    strategy = SydneyRangeFadeStrategy(
        config=SydneyRangeFadeConfig(x_offset=3.5, sl_points=10.0, tp_points=18.0, base_units=10.0),
        session_engine=session_engine,
        risk_service=risk_service,
    )
    simulator = ExecutionSimulator(fee_model=FeeModel(), point_value=1.0)

    # 3. Simulate across a 20-trading-day evaluation window (April 2024)
    test_dates = pd.date_range("2024-04-15", "2024-05-15", freq="B").date

    initial_equity = 50000.0
    current_equity = initial_equity
    daily_equity_records = {}
    trade_pnls = []
    executed_chains = []

    print(f"\nSimulating {len(test_dates)} trading days with continuous 1m bar replay...")

    for target_date in test_dates:
        window = session_engine.get_session_window(target_date)
        end_of_day_utc = dt.datetime(target_date.year, target_date.month, target_date.day, 21, 0, tzinfo=dt.timezone.utc)
        day_df = df[(df["utc_time"] >= window.sydney_open_utc) & (df["utc_time"] <= end_of_day_utc)]

        # Evaluate at 12:30 IST
        signal = strategy.evaluate_day(target_date, day_df, account_equity=current_equity)
        if signal is None:
            daily_equity_records[target_date] = current_equity
            continue

        chain_id = f"CHAIN-{target_date.strftime('%Y%m%d')}"
        chain = TradeChain(
            chain_id=chain_id,
            config=ChainedFollowUpConfig(sl_flip_offset=8.0, tp_flip_offset=20.0, max_chain_depth=2),
            risk_service=risk_service,
            account_equity=current_equity,
        )

        # Start Leg 1
        chain.start_leg1(
            direction=signal.direction,
            entry_time=signal.eval_time_utc,
            entry_price=signal.entry_price,
            base_units=signal.units,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
        )

        # Replay subsequent 1m bars
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

        # Save to SQLite database
        repo.save_trade_chain(chain)
        executed_chains.append(chain)

        current_equity += chain.total_pnl_net
        daily_equity_records[target_date] = current_equity
        for leg in chain.legs:
            if not leg.is_open:
                trade_pnls.append(leg.pnl_net)

    equity_series = pd.Series(daily_equity_records)

    # 4. Compute Portfolio Compounding Metrics
    portfolio_engine = PortfolioEngine(initial_equity=initial_equity)
    metrics = portfolio_engine.compute_metrics(equity_series, trade_pnls)

    print("\n" + "=" * 80)
    print("PORTFOLIO PERFORMANCE RESULTS:")
    print(f"  Initial Equity   : ${metrics.initial_equity:,.2f}")
    print(f"  Final Equity     : ${metrics.final_equity:,.2f}")
    print(f"  Total Net Return : {metrics.total_return_pct:+.2f}% (${metrics.total_net_pnl:+,.2f})")
    print(f"  Sharpe Ratio     : {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio    : {metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown     : -{metrics.max_drawdown_pct:.2f}% (${metrics.max_drawdown_usd:,.2f})")
    print(f"  Total Trades     : {metrics.total_trades} (Win Rate: {metrics.win_rate_pct:.1f}%)")
    print(f"  Profit Factor    : {metrics.profit_factor:.2f}")
    print("=" * 80)

    # 5. Generate Standalone Interactive HTML Report
    report_file = REPORT_DIR / "hypotrader_gold_report.html"
    report_path = ReportVisualizer.generate_html_report(
        hypothesis_title=hypo.title,
        hypothesis_id=hypo.id,
        metrics=metrics,
        daily_equity_series=equity_series,
        executed_chains=executed_chains,
        output_filepath=report_file,
    )
    print(f"\n[REPORT GENERATED] Interactive HTML report saved to: {report_path}")


if __name__ == "__main__":
    main()
