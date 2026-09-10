"""
Unit tests for Optuna Study Engine, Full Trial Logging, and DSR / PSR calculations.
"""

import datetime as dt
import pytest
from pathlib import Path

from src.optimization.dsr import StatisticalValidationEngine
from src.optimization.trial_logger import TrialLogger
from src.optimization.engine import OptunaStudyEngine
from src.optimization.gates import CostFloorGate

PARQUET_PATH = Path("/Users/prince/algo-trading/data/historical/XAUUSD_1m.parquet")


def test_dsr_penalizes_multiple_trials():
    """
    Mathematical property of DSR: As trial count N increases,
    expected maximum Sharpe under the null hypothesis increases,
    meaning the DSR score for the same observed Sharpe drops.
    """
    observed_sr = 2.0
    trials_dist = [0.2, 0.5, 1.1, 1.4, 1.8, 2.0]

    dsr_10_trials, exp_max_10, _ = StatisticalValidationEngine.deflated_sharpe_ratio(
        observed_sharpe=observed_sr,
        trial_sharpe_distribution=trials_dist,
        n_total_trials=10,
        n_samples=50,
    )

    dsr_1000_trials, exp_max_1000, _ = StatisticalValidationEngine.deflated_sharpe_ratio(
        observed_sharpe=observed_sr,
        trial_sharpe_distribution=trials_dist,
        n_total_trials=1000,
        n_samples=50,
    )

    # 1000 trials raises the expected max Sharpe under random noise
    assert exp_max_1000 > exp_max_10
    # Consequently, confidence that observed_sr=2.0 is genuine alpha drops!
    assert dsr_10_trials > dsr_1000_trials


def test_trial_logger_saves_all_trials(tmp_path):
    db_file = tmp_path / "test_trials.db"
    logger = TrialLogger(db_file)

    # Log 1 completed trial and 1 pruned trial
    logger.log_trial(
        trial_id=0,
        study_name="test_study",
        x_offset=3.5,
        sl_points=10.0,
        tp_points=18.0,
        eval_time_ist="12:30",
        cost_sl_ratio=0.025,
        is_pruned=False,
        sharpe_ratio=1.85,
        annualized_return=24.5,
        max_drawdown_pct=-4.2,
        trades_count=15,
    )

    logger.log_trial(
        trial_id=1,
        study_name="test_study",
        x_offset=2.0,
        sl_points=1.0,
        tp_points=10.0,
        eval_time_ist="12:30",
        cost_sl_ratio=0.25,
        is_pruned=True,
        prune_reason="Cost floor exceeded",
    )

    total_trials = logger.get_total_trials_count("test_study")
    assert total_trials == 2

    sharpes = logger.get_all_sharpe_ratios("test_study")
    assert len(sharpes) == 1
    assert sharpes[0] == 1.85


def test_optuna_mini_study_with_dsr(tmp_path):
    if not PARQUET_PATH.exists():
        pytest.skip(f"Parquet not found at {PARQUET_PATH}")

    db_file = tmp_path / "test_optuna.db"
    engine = OptunaStudyEngine(
        parquet_path=PARQUET_PATH,
        study_name="mini_test_study",
        db_path=db_file,
    )

    # Run a 5-trial study over 5 sample days in April 2024
    test_dates = [
        dt.date(2024, 4, 22),
        dt.date(2024, 4, 23),
        dt.date(2024, 4, 24),
        dt.date(2024, 4, 25),
        dt.date(2024, 4, 26),
    ]

    results = engine.run_study(test_dates=test_dates, n_trials=5)

    assert "best_params" in results
    assert "best_sharpe" in results
    assert results["total_trials_evaluated"] == 5
    assert "deflated_sharpe_ratio" in results
    assert 0.0 <= results["deflated_sharpe_ratio"] <= 1.0
