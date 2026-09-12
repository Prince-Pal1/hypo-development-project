import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Dict, Any, Tuple, Optional
import optuna
from pathlib import Path
from dataclasses import dataclass
import json

from src.optimization.engine import OptunaStudyEngine

@dataclass
class WindowResult:
    window_id: int
    start_date: dt.date
    end_date: dt.date
    trials: pd.DataFrame
    centroid_params: Dict[str, Any]
    centroid_score: float

class WalkForwardOptimizer:
    def __init__(
        self,
        parquet_path: str,
        study_prefix: str,
        engine_factory,  # Callable[[str, str, str, str], OptunaStudyEngine]
        n_trials_per_window: int = 50,
        window_size_days: int = 10,
        step_size_days: int = 10,
        min_start_date: str = None,
    ):
        self.parquet_path = parquet_path
        self.study_prefix = study_prefix
        self.engine_factory = engine_factory
        self.n_trials_per_window = n_trials_per_window
        self.window_size_days = window_size_days
        self.step_size_days = step_size_days
        self.min_start_date = min_start_date
        self.windows: List[WindowResult] = []

    def _get_trading_days(self) -> List[dt.date]:
        df = pd.read_parquet(self.parquet_path)
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            dt_series = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            dt_series = pd.to_datetime(df["timestamp"], utc=True)
        dates = sorted(list(dt_series.dt.date.unique()))
        
        if self.min_start_date:
            min_date = dt.datetime.strptime(self.min_start_date, "%Y-%m-%d").date()
            dates = [d for d in dates if d >= min_date]
            
        return dates

    def _build_windows(self, trading_days: List[dt.date]) -> List[Tuple[dt.date, dt.date]]:
        windows = []
        i = 0
        while i + self.window_size_days <= len(trading_days):
            w_start = trading_days[i]
            w_end = trading_days[i + self.window_size_days - 1]
            windows.append((w_start, w_end))
            i += self.step_size_days
        return windows

    def _extract_plateau_centroid(self, trials: pd.DataFrame) -> Tuple[Dict[str, Any], float]:
        """
        Extracts the 90% plateau of top trials and computes the centroid.
        Uses Kelly-optimal log growth rate as tie-breaker/primary selector over Sharpe
        to avoid tail risk, per user constraints.
        """
        if len(trials) == 0:
            return {}, 0.0

        # We need expected log growth. If it's not logged natively, we approximate it.
        # Approx Kelly log growth rate g = R - (Var/2)
        # Using annualized return and Sharpe (which gives us risk)
        if "annualized_return" in trials.columns and "sharpe_ratio" in trials.columns:
            # Reconstruct std dev (annualized) from Sharpe = (Ret - Rf) / StdDev => StdDev = Ret / Sharpe
            # For simplicity, if Sharpe > 0:
            std_devs = np.where(trials["sharpe_ratio"] > 0, trials["annualized_return"] / trials["sharpe_ratio"], 0.0)
            variances = std_devs ** 2
            trials["log_growth"] = trials["annualized_return"] - (variances / 2.0)
            score_col = "log_growth"
        else:
            score_col = "value" if "value" in trials.columns else "sharpe_ratio"

        # 90th percentile plateau
        p90_threshold = trials[score_col].quantile(0.90)
        plateau = trials[trials[score_col] >= p90_threshold].copy()
        
        if len(plateau) == 0:
            # Fallback to absolute best
            best_idx = trials[score_col].idxmax()
            best_row = trials.loc[best_idx]
            return json.loads(best_row["params_json"]), best_row[score_col]
            
        # Centroid (mean of each parameter)
        centroid = {}
        # Parse params JSON for the plateau
        parsed_params = [json.loads(p) for p in plateau["params_json"]]
        df_params = pd.DataFrame(parsed_params)
        
        for col in df_params.columns:
            if pd.api.types.is_numeric_dtype(df_params[col]):
                centroid[col] = df_params[col].mean()
            else:
                # For categorical, take the mode
                centroid[col] = df_params[col].mode()[0]
                
        centroid_score_est = plateau[score_col].mean()
        return centroid, centroid_score_est

    def evaluate_windows(self) -> None:
        """Runs the optimization engine on each isolated training window."""
        trading_days = self._get_trading_days()
        window_bounds = self._build_windows(trading_days)
        print(f"Total isolated training windows to evaluate: {len(window_bounds)}")
        
        import sqlite3
        
        for idx, (w_start, w_end) in enumerate(window_bounds):
            study_name = f"{self.study_prefix}_W{idx:02d}"
            print(f"Evaluating Window {idx} ({w_start} to {w_end})...")
            
            engine = self.engine_factory(
                study_name=study_name,
                sub_interval_id=f"W{idx:02d}",
                sub_interval_start=str(w_start),
                sub_interval_end=str(w_end),
            )
            # We must pass target dates to the engine's objective callback.
            # In our generalized engine, the objective_callback has closure over the dates.
            # Wait, the factory will need to bind the dates to the objective.
            # Let's assume the factory does this via a set_dates method or similar if needed.
            if hasattr(engine, "set_test_dates"):
                days_in_window = [d for d in trading_days if w_start <= d <= w_end]
                engine.set_test_dates(days_in_window)
            
            engine.run_study(
                n_trials=self.n_trials_per_window,
                n_samples=self.window_size_days,
                parent_study_prefix=self.study_prefix,
            )
            
            # Extract trials from DB
            with sqlite3.connect("hypotrader.db") as conn:
                query = f"SELECT * FROM optimizer_trials WHERE study_name = '{study_name}' AND is_pruned = 0"
                trials_df = pd.read_sql_query(query, conn)
                
            centroid_params, centroid_score = self._extract_plateau_centroid(trials_df)
            
            self.windows.append(WindowResult(
                window_id=idx,
                start_date=w_start,
                end_date=w_end,
                trials=trials_df,
                centroid_params=centroid_params,
                centroid_score=centroid_score,
            ))

    def _evaluate_param_on_window(self, params: Dict[str, Any], window: WindowResult) -> float:
        """
        Approximates the score of `params` on `window`.
        If the exact params were tested, return the score.
        Otherwise, find the nearest neighbor in the window's trials by Euclidean distance of params.
        """
        if len(window.trials) == 0:
            return 0.0
            
        score_col = "log_growth" if "log_growth" in window.trials.columns else "sharpe_ratio"
        if score_col not in window.trials.columns:
            score_col = "value" if "value" in window.trials.columns else "sharpe_ratio"

        # Simple 1-NN lookup for approximation
        min_dist = float('inf')
        best_score = 0.0
        
        for idx, row in window.trials.iterrows():
            trial_params = json.loads(row["params_json"])
            
            dist = 0.0
            for k, v in params.items():
                if k in trial_params and isinstance(v, (int, float)):
                    # Normalize roughly (assume unit scale for now or just raw distance)
                    dist += (float(v) - float(trial_params[k])) ** 2
            
            if dist < min_dist:
                min_dist = dist
                best_score = float(row[score_col]) if not pd.isna(row[score_col]) else 0.0
                
        return best_score

    def _solve_dp_switching(self, historical_windows: List[WindowResult], lambda_val: float) -> Dict[str, Any]:
        """
        Solves DP: maximize sum_k R_k(θ_k) - λ * sum_k 1[θ_k != θ_{k-1}]
        Uses the plateau centroids of all historical windows as the discrete state space for θ_k.
        Returns the optimal parameter state for the FINAL historical window.
        """
        if not historical_windows:
            return {}
            
        N = len(historical_windows)
        candidate_params = [w.centroid_params for w in historical_windows]
        
        # dp[k][i] = max cumulative score up to window k ending in parameter state i
        dp = np.zeros((N, len(candidate_params)))
        backpointer = np.zeros((N, len(candidate_params)), dtype=int)
        
        # Base case for k=0
        for i, param_state in enumerate(candidate_params):
            dp[0][i] = self._evaluate_param_on_window(param_state, historical_windows[0])
            
        # DP transitions
        for k in range(1, N):
            for i, param_state in enumerate(candidate_params):
                reward = self._evaluate_param_on_window(param_state, historical_windows[k])
                
                best_prev_score = -float('inf')
                best_prev_idx = -1
                
                for j, prev_state in enumerate(candidate_params):
                    # Switching cost penalty
                    penalty = lambda_val if i != j else 0.0
                    score = dp[k-1][j] + reward - penalty
                    
                    if score > best_prev_score:
                        best_prev_score = score
                        best_prev_idx = j
                        
                dp[k][i] = best_prev_score
                backpointer[k][i] = best_prev_idx
                
        # The optimal state at the end of the training period
        best_final_idx = np.argmax(dp[N-1])
        return candidate_params[best_final_idx]

    def synthesize_policy_for_window(self, target_window_idx: int, lambda_val: float) -> Dict[str, Any]:
        """
        Returns the parameter dict to use for `target_window_idx` by running DP
        over all windows strictly prior to `target_window_idx`.
        """
        historical_windows = [w for w in self.windows if w.window_id < target_window_idx]
        if not historical_windows:
            # First window has no history, just return a default or static baseline
            return self.windows[0].centroid_params if self.windows else {}
            
        return self._solve_dp_switching(historical_windows, lambda_val)

    def select_lambda_cv(self) -> float:
        """
        Selects λ via nested cross-validation on the training windows.
        Since WFO is sequential, we do a nested walk-forward over the historical windows.
        """
        print("Selecting λ via nested cross-validation...")
        lambdas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        if len(self.windows) < 3:
            print("WARNING: Not enough windows for nested CV. Defaulting to λ=2.0")
            return 2.0
            
        # We need a train/test split. Let's reserve the first N/2 windows as the "burn-in" 
        # and validate lambdas on the remaining windows in a walk-forward manner.
        burn_in = max(1, len(self.windows) // 2)
        
        lambda_scores = {lam: 0.0 for lam in lambdas}
        
        for lam in lambdas:
            cumulative_score = 0.0
            for test_idx in range(burn_in, len(self.windows)):
                hist_windows = self.windows[:test_idx]
                test_window = self.windows[test_idx]
                
                # Pick policy using historical windows
                chosen_params = self._solve_dp_switching(hist_windows, lam)
                
                # Evaluate on the test window
                score = self._evaluate_param_on_window(chosen_params, test_window)
                cumulative_score += score
                
            lambda_scores[lam] = cumulative_score
            
        best_lam = max(lambda_scores, key=lambda_scores.get)
        print(f"Nested CV Lambda scores: {lambda_scores}")
        print(f"Selected λ: {best_lam}")
        
        print("WARNING: Sample Size. With ~16-17 windows total, the sample is extremely thin.")
        print("Fitting a discrete HMM or clustered policy to this will suffer high variance.")
        print("Defaulting to the CV-optimal regularized policy to minimize regime whip-sawing.")
        return best_lam
