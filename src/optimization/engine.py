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
from typing import Optional, List, Dict, Any

from src.session.engine import SessionEngine
from src.core.risk_service import RiskService, RiskServiceLimits
from src.strategies.sydney_range_fade import SydneyRangeFadeStrategy, SydneyRangeFadeConfig
from src.sequencer.fsm import TradeChain, ChainedFollowUpConfig, ChainStatus
from src.execution.simulator import ExecutionSimulator, FeeModel
from src.optimization.gates import CostFloorGate
from src.optimization.trial_logger import TrialLogger
from src.optimization.dsr import StatisticalValidationEngine

# Suppress verbose Optuna logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaStudyEngine:
    def __init__(
        self,
        parquet_path: Path | str,
        study_name: str = "gold_sydney_range_study",
        db_path: Path | str = "hypotrader.db",
        cost_floor_gate: Optional[CostFloorGate] = None,
        risk_limits: Optional[RiskServiceLimits] = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.study_name = study_name
        self.trial_logger = TrialLogger(db_path)
        self.cost_floor_gate = cost_floor_gate or CostFloorGate(max_cost_sl_ratio=0.15)
        
        # Hard non-negotiable RiskService ceiling (isolated from search space)
        self.risk_limits = risk_limits or RiskServiceLimits(
            max_size_multiplier_per_chain_step=1.0,
            max_chain_depth=2,
            daily_loss_circuit_breaker_pct=3.0,
        )
        self.session_engine = SessionEngine()
        self.simulator = ExecutionSimulator(fee_model=FeeModel(), point_value=1.0)
        self._load_data()

    def _load_data(self) -> None:
        self.df = pd.read_parquet(self.parquet_path)
        if pd.api.types.is_numeric_dtype(self.df["timestamp"]):
            self.df["utc_time"] = pd.to_datetime(self.df["timestamp"], unit="ms", utc=True)
        else:
            self.df["utc_time"] = pd.to_datetime(self.df["timestamp"], utc=True)

    def _run_backtest_for_params(
        self,
        test_dates: List[dt.date],
        x_offset: float,
        sl_points: float,
        tp_points: float,
        sl_flip_offset: float,
        tp_flip_offset: float,
    ) -> tuple[float, float, float, int]:
        """Runs the chained backtest across given sample dates."""
        strategy = SydneyRangeFadeStrategy(
            config=SydneyRangeFadeConfig(
                x_offset=x_offset,
                sl_points=sl_points,
                tp_points=tp_points,
                base_units=10.0,
            ),
            session_engine=self.session_engine,
            risk_service=RiskService(self.risk_limits),
        )

        chain_config = ChainedFollowUpConfig(
            sl_flip_offset=sl_flip_offset,
            tp_flip_offset=tp_flip_offset,
            requested_multiplier=1.0,  # Clamped strictly by RiskService
            max_chain_depth=2,
        )

        daily_returns: List[float] = []
        trades_count = 0
        account_equity = 50000.0

        for target_date in test_dates:
            window = self.session_engine.get_session_window(target_date)
            end_of_day_utc = dt.datetime(target_date.year, target_date.month, target_date.day, 21, 0, tzinfo=dt.timezone.utc)
            day_df = self.df[(self.df["utc_time"] >= window.sydney_open_utc) & (self.df["utc_time"] <= end_of_day_utc)]

            signal = strategy.evaluate_day(target_date, day_df, account_equity=account_equity)
            if signal is None:
                continue

            chain = TradeChain(
                chain_id=f"TRIAL-{target_date.strftime('%Y%m%d')}",
                config=chain_config,
                risk_service=RiskService(self.risk_limits),
                account_equity=account_equity,
            )

            chain.start_leg1(
                direction=signal.direction,
                entry_time=signal.eval_time_utc,
                entry_price=signal.entry_price,
                base_units=signal.units,
                sl_price=signal.sl_price,
                tp_price=signal.tp_price,
            )

            forward_bars = day_df[day_df["utc_time"] > signal.eval_time_utc]
            for _, row in forward_bars.iterrows():
                self.simulator.process_bar(
                    chain=chain,
                    bar_time=row["utc_time"],
                    open_p=row["open"],
                    high_p=row["high"],
                    low_p=row["low"],
                    close_p=row["close"],
                )
                if chain.status == ChainStatus.COMPLETED or chain.current_leg is None:
                    break

            trades_count += len(chain.legs)
            daily_returns.append(chain.total_pnl_net / account_equity)
            account_equity += chain.total_pnl_net

        if not daily_returns or len(daily_returns) < 3:
            return 0.0, 0.0, 0.0, trades_count

        arr = np.array(daily_returns)
        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr, ddof=1))
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
        annualized_ret = mean_ret * 252 * 100.0

        # Max drawdown
        cum_ret = np.cumprod(1.0 + arr)
        peaks = np.maximum.accumulate(cum_ret)
        drawdowns = (cum_ret - peaks) / peaks
        max_dd = float(np.min(drawdowns)) * 100.0 if len(drawdowns) > 0 else 0.0

        return sharpe, annualized_ret, max_dd, trades_count

    def objective(self, trial: optuna.Trial, test_dates: List[dt.date]) -> float:
        # 1. Parameter suggestions (strictly bounded, risk limits NOT in search space)
        x_offset = trial.suggest_float("x_offset", 1.5, 5.0, step=0.5)
        sl_points = trial.suggest_float("sl_points", 6.0, 20.0, step=1.0)
        tp_points = trial.suggest_float("tp_points", 12.0, 30.0, step=2.0)
        sl_flip_offset = trial.suggest_float("sl_flip_offset", 5.0, 15.0, step=1.0)
        tp_flip_offset = trial.suggest_float("tp_flip_offset", 15.0, 30.0, step=2.0)

        # 2. Hard Precondition Gate: Cost Floor Check
        passed, cost_ratio, gate_msg = self.cost_floor_gate.evaluate(sl_points)
        if not passed:
            self.trial_logger.log_trial(
                trial_id=trial.number,
                study_name=self.study_name,
                x_offset=x_offset,
                sl_points=sl_points,
                tp_points=tp_points,
                eval_time_ist="12:30",
                cost_sl_ratio=cost_ratio,
                is_pruned=True,
                prune_reason=gate_msg,
            )
            raise optuna.TrialPruned(gate_msg)

        # 3. Execute backtest
        sharpe, ann_ret, max_dd, trades = self._run_backtest_for_params(
            test_dates=test_dates,
            x_offset=x_offset,
            sl_points=sl_points,
            tp_points=tp_points,
            sl_flip_offset=sl_flip_offset,
            tp_flip_offset=tp_flip_offset,
        )

        # 4. Log full trial distribution
        self.trial_logger.log_trial(
            trial_id=trial.number,
            study_name=self.study_name,
            x_offset=x_offset,
            sl_points=sl_points,
            tp_points=tp_points,
            eval_time_ist="12:30",
            cost_sl_ratio=cost_ratio,
            is_pruned=False,
            sharpe_ratio=sharpe,
            annualized_return=ann_ret,
            max_drawdown_pct=max_dd,
            trades_count=trades,
        )

        return sharpe

    def run_study(self, test_dates: List[dt.date], n_trials: int = 15) -> Dict[str, Any]:
        """Runs the Optuna optimization study and performs DSR validation across all trials."""
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(lambda trial: self.objective(trial, test_dates), n_trials=n_trials)

        best_trial = study.best_trial
        all_sharpes = self.trial_logger.get_all_sharpe_ratios(self.study_name)
        total_trials = self.trial_logger.get_total_trials_count(self.study_name)

        # Compute Deflated Sharpe Ratio (DSR)
        dsr_score, exp_max_sr, trial_var = StatisticalValidationEngine.deflated_sharpe_ratio(
            observed_sharpe=best_trial.value,
            trial_sharpe_distribution=all_sharpes,
            n_total_trials=total_trials,
            n_samples=len(test_dates),
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
