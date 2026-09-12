"""
Optuna Bayesian Optimizer Engine with Hard Cost-Floor Gate & DSR Validation.
Ensures structural isolation of risk limits, pre-simulation cost filtering,
and complete trial logging across all evaluated parameter combinations.
"""

import optuna
import datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Callable

from src.optimization.gates import CostFloorGate
from src.optimization.trial_logger import TrialLogger
from src.optimization.dsr import StatisticalValidationEngine

# Suppress verbose Optuna logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaStudyEngine:
    def __init__(
        self,
        study_name: str,
        suggest_params_callback: Callable[[optuna.Trial], Dict[str, Any]],
        objective_callback: Callable[[Dict[str, Any]], tuple[float, float, float, int]],
        db_path: Union[Path, str] = "hypotrader.db",
        cost_floor_gate: Optional[CostFloorGate] = None,
        sub_interval_id: Optional[str] = None,
        sub_interval_start: Optional[str] = None,
        sub_interval_end: Optional[str] = None,
    ):
        self.study_name = study_name
        self.suggest_params_callback = suggest_params_callback
        self.objective_callback = objective_callback
        self.trial_logger = TrialLogger(db_path)
        self.cost_floor_gate = cost_floor_gate
        self.sub_interval_id = sub_interval_id
        self.sub_interval_start = sub_interval_start
        self.sub_interval_end = sub_interval_end

    def objective(self, trial: optuna.Trial) -> float:
        # 1. Suggest parameters
        params = self.suggest_params_callback(trial)

        # 2. Hard Precondition Gate: Cost Floor Check
        cost_ratio = 0.0
        if self.cost_floor_gate and "sl_points" in params:
            passed, cost_ratio, gate_msg = self.cost_floor_gate.evaluate(params["sl_points"])
            if not passed:
                self.trial_logger.log_trial(
                    trial_id=trial.number,
                    study_name=self.study_name,
                    is_pruned=True,
                    prune_reason=gate_msg,
                    cost_sl_ratio=cost_ratio,
                    sub_interval_id=self.sub_interval_id,
                    sub_interval_start=self.sub_interval_start,
                    sub_interval_end=self.sub_interval_end,
                    params=params,
                )
                raise optuna.TrialPruned(gate_msg)

        # 3. Execute backtest
        sharpe, ann_ret, max_dd, trades = self.objective_callback(params)

        # 4. Log full trial distribution
        self.trial_logger.log_trial(
            trial_id=trial.number,
            study_name=self.study_name,
            is_pruned=False,
            cost_sl_ratio=cost_ratio,
            sharpe_ratio=sharpe,
            annualized_return=ann_ret,
            max_drawdown_pct=max_dd,
            trades_count=trades,
            sub_interval_id=self.sub_interval_id,
            sub_interval_start=self.sub_interval_start,
            sub_interval_end=self.sub_interval_end,
            params=params,
        )

        return sharpe

    def run_study(
        self, 
        n_trials: int = 15, 
        n_samples: int = 252, 
        parent_study_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs the Optuna optimization study and performs DSR validation.
        `parent_study_prefix` groups multiple sub-intervals under one logical DSR distribution.
        """
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(self.objective, n_trials=n_trials)

        best_trial = study.best_trial
        
        prefix = parent_study_prefix if parent_study_prefix else self.study_name
        all_sharpes = self.trial_logger.get_all_sharpe_ratios(self.study_name) # we only use sharpes from THIS sub-interval for the observed SR, but wait
        # DSR should be calculated against ALL sharpe ratios evaluated in the entire walk-forward loop?
        # Actually, for a specific sub-interval study, the trials evaluated *here* are what count for this sub-interval's DSR, 
        # BUT the n_total_trials should be aggregated across all sub-intervals.
        total_trials = self.trial_logger.get_total_trials_count(prefix)

        # Compute Deflated Sharpe Ratio (DSR)
        dsr_score, exp_max_sr, trial_var = StatisticalValidationEngine.deflated_sharpe_ratio(
            observed_sharpe=best_trial.value,
            trial_sharpe_distribution=all_sharpes,
            n_total_trials=total_trials,
            n_samples=n_samples,
        )

        return {
            "study_name": self.study_name,
            "best_params": best_trial.params,
            "best_sharpe": round(best_trial.value, 3),
            "total_trials_evaluated": total_trials,
            "completed_trials_count": len(all_sharpes),
            "trial_sharpe_variance": round(trial_var, 4),
            "expected_max_null_sharpe": round(exp_max_sr, 3),
            "deflated_sharpe_ratio": round(dsr_score, 4),
            "dsr_passed": dsr_score > 0.50,
        }
