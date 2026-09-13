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
        
        # Filter out Saturday (5) and Sunday (6) to prevent non-standard session inflation
        dates = [d for d in dates if d.weekday() < 5]
        
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
            std_devs = np.where(trials["sharpe_ratio"] != 0, np.abs(trials["annualized_return"]) / np.abs(trials["sharpe_ratio"]), 0.0)
            variances = std_devs ** 2
            trials["log_growth"] = trials["annualized_return"] - (variances / 2.0)
            score_col = "log_growth"
        else:
            score_col = "value" if "value" in trials.columns else "sharpe_ratio"

        if len(trials) > 1 and trials[score_col].var() < 1e-6:
            raise ValueError(f"Score variance across all trials is near zero for metric '{score_col}'. The scoring function is likely broken or returning identical dummy values.")

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
            if pd.api.types.is_numeric_dtype(df_params[col]) and not pd.api.types.is_bool_dtype(df_params[col]):
                centroid[col] = float(df_params[col].mean())
            else:
                # For categorical and boolean, take the mode
                mode_val = df_params[col].mode()[0]
                centroid[col] = bool(mode_val) if pd.api.types.is_bool_dtype(df_params[col]) else mode_val
                
        centroid_score_est = plateau[score_col].mean()
        return centroid, centroid_score_est

    def evaluate_windows(self) -> None:
        """Runs the optimization engine on each isolated training window."""
        trading_days = self._get_trading_days()
        window_bounds = self._build_windows(trading_days)
        print(f"Total isolated training windows to evaluate: {len(window_bounds)}")
        if len(window_bounds) > 0:
            print("First 5 windows:")
            for idx, (w_start, w_end) in enumerate(window_bounds[:5]):
                print(f"  Window {idx}: {w_start} to {w_end}")
            
            if len(window_bounds) > 5:
                print("Last 5 windows:")
                start_idx = max(5, len(window_bounds) - 5)
                for idx, (w_start, w_end) in enumerate(window_bounds[start_idx:], start=start_idx):
                    print(f"  Window {idx}: {w_start} to {w_end}")

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
            
            # Phase B: QMC Coverage Check
            self._verify_qmc_coverage(trials_df)
            
            self.windows.append(WindowResult(
                window_id=idx,
                start_date=w_start,
                end_date=w_end,
                trials=trials_df,
                centroid_params=centroid_params,
                centroid_score=centroid_score,
            ))

    def _verify_qmc_coverage(self, trials: pd.DataFrame) -> None:
        """
        Splits each continuous parameter's range into 5 equal bins and reports the count.
        This verifies QMC sampling uniformity across strata.
        """
        if len(trials) == 0:
            return
            
        print("    [QMC Coverage Check]")
        parsed_params = [json.loads(p) for p in trials["params_json"]]
        df_params = pd.DataFrame(parsed_params)
        
        # Identify discrete combinations (strata)
        categorical_cols = [c for c in df_params.columns if not pd.api.types.is_numeric_dtype(df_params[c])]
        numeric_cols = [c for c in df_params.columns if pd.api.types.is_numeric_dtype(df_params[c])]
        
        if categorical_cols:
            strata = df_params.groupby(categorical_cols)
        else:
            strata = [("All", df_params)]
            
        for stratum_name, stratum_df in strata:
            # print(f"      Stratum {stratum_name} (N={len(stratum_df)}):")
            for col in numeric_cols:
                if len(stratum_df[col].unique()) > 1:
                    bins = pd.cut(stratum_df[col], bins=5)
                    counts = bins.value_counts().sort_index().values
                    
                    if len(counts) > 0 and min(counts) > 0:
                        ratio = max(counts) / min(counts)
                        if ratio > 3.0:
                            print(f"      [WARNING] QMC Coverage Skewed in {col} (ratio={ratio:.1f}). Counts: {counts}")
                    elif len(counts) > 0 and max(counts) > 0:
                        print(f"      [WARNING] QMC Coverage has empty bins in {col}. Counts: {counts}")
                        
                    # print(f"        {col}: {counts}")
                # else:
                #    print(f"        {col}: All values = {stratum_df[col].iloc[0]}")


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

        # Pre-compute Z-score stats for continuous parameters in this window
        parsed_params = [json.loads(p) for p in window.trials["params_json"]]
        df_params = pd.DataFrame(parsed_params)
        
        continuous_cols = [c for c in df_params.columns if pd.api.types.is_numeric_dtype(df_params[c]) and df_params[c].nunique() > 1]
        means = df_params[continuous_cols].mean()
        stds = df_params[continuous_cols].std().replace(0, 1.0)

        # Simple 1-NN lookup for approximation
        min_dist = float('inf')
        best_score = 0.0
        
        for idx, row in window.trials.iterrows():
            trial_params = json.loads(row["params_json"])
            
            dist = 0.0
            for k, v in params.items():
                if k in trial_params and isinstance(v, (int, float)):
                    if k in continuous_cols:
                        v_norm = (float(v) - means[k]) / stds[k]
                        t_norm = (float(trial_params[k]) - means[k]) / stds[k]
                        dist += (v_norm - t_norm) ** 2
                    else:
                        dist += (float(v) - float(trial_params[k])) ** 2
            
            if dist < min_dist:
                min_dist = dist
                best_score = float(row[score_col]) if not pd.isna(row[score_col]) else 0.0
                
        return best_score

    def _solve_ruptures_switching(self, historical_windows: List[WindowResult], lambda_val: float, signal_override: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], int]:
        """
        Uses ruptures.Pelt to find regime shifts based on plateau centroids or an alternative signal stream.
        Returns a tuple of (chosen_params, regime_start_index).
        """
        if not historical_windows:
            return {}, 0
            
        import ruptures as rpt
        
        # 1. Build signal matrix
        keys = sorted(list(historical_windows[0].centroid_params.keys()))
        continuous_keys = [k for k in keys if isinstance(historical_windows[0].centroid_params[k], (int, float))]
        categorical_keys = [k for k in keys if not isinstance(historical_windows[0].centroid_params[k], (int, float))]
        
        if signal_override is not None:
            signal = signal_override
        else:
            signal_data = []
            for w in historical_windows:
                row = []
                for k in continuous_keys:
                    row.append(float(w.centroid_params.get(k, 0.0)))
                signal_data.append(row)
            signal = np.array(signal_data)
        
        if signal.shape[1] > 0 and len(signal) > 1:
            # Normalize signal for L2 cost
            std = signal.std(axis=0)
            std[std == 0] = 1.0
            signal_norm = (signal - signal.mean(axis=0)) / std
            
            # 2. Find changepoints
            try:
                algo = rpt.Pelt(model="l2", min_size=1, jump=1).fit(signal_norm)
                breakpoints = algo.predict(pen=lambda_val)
                # Breakpoints are indices. The last segment starts at the previous breakpoint.
                regime_start = breakpoints[-2] if len(breakpoints) > 1 else 0
            except Exception as e:
                print(f"Ruptures Pelt failed: {e}. Fallback to static.")
                regime_start = 0
        else:
            regime_start = 0
            
        current_regime_windows = historical_windows[regime_start:]
        
        # 3. Compute centroid of the current regime
        final_params = {}
        for k in continuous_keys:
            vals = [w.centroid_params.get(k) for w in current_regime_windows]
            final_params[k] = float(np.mean(vals))
            
        for k in categorical_keys:
            vals = [w.centroid_params.get(k) for w in current_regime_windows]
            from collections import Counter
            final_params[k] = Counter(vals).most_common(1)[0][0]
            
        return final_params, regime_start

    def get_alternative_signal(self, windows: List[WindowResult]) -> np.ndarray:
        import sqlite3
        import os
        db_path = "hypotrader.db"
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path)
        signals = []
        for w in windows:
            df = pd.read_sql_query(
                "SELECT parkinson_volatility, adx FROM market_conditions WHERE date >= ? AND date <= ?",
                conn, params=(w.start_date.isoformat(), w.end_date.isoformat())
            )
            if len(df) > 0:
                vol = float(df["parkinson_volatility"].mean(skipna=True))
                adx = float(df["adx"].mean(skipna=True))
                if np.isnan(vol): vol = 0.0
                if np.isnan(adx): adx = 0.0
                signals.append([vol, adx])
            else:
                signals.append([0.0, 0.0])
        conn.close()
        return np.array(signals)

    def synthesize_policy_for_window(self, target_window_idx: int, lambda_val: float, use_leading_signals: bool = False) -> Tuple[Dict[str, Any], int]:
        """
        Returns the (parameter dict, regime_start_index) to use for `target_window_idx` by running ruptures
        over all windows strictly prior to `target_window_idx`.
        """
        historical_windows = [w for w in self.windows if w.window_id < target_window_idx]
        if not historical_windows:
            # First window has no history, just return a default or static baseline
            return (self.windows[0].centroid_params if self.windows else {}), 0
            
        signal_override = self.get_alternative_signal(historical_windows) if use_leading_signals else None
        return self._solve_ruptures_switching(historical_windows, lambda_val, signal_override=signal_override)

    def select_lambda_cv(self, burn_in_windows: int = 12) -> float:
        """
        Selects λ via nested cross-validation on the training windows.
        Since WFO is sequential, we do a nested walk-forward over the historical windows.
        """
        print("Selecting λ via nested cross-validation...")
        lambdas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        if len(self.windows) < 3:
            print("WARNING: Not enough windows for nested CV. Defaulting to λ=2.0")
            return 2.0
            
        # Lambda Sensitivity Grid (Phase C)
        print("λ Sensitivity Grid (Full History):")
        keys = sorted(list(self.windows[0].centroid_params.keys()))
        continuous_keys = [k for k in keys if isinstance(self.windows[0].centroid_params[k], (int, float))]
        
        # --- NEW RUPTURES DIAGNOSTIC ---
        import ruptures as rpt
        print("\n[Ruptures Diagnostic] Individual Dimension Shifts (λ=2.0):")
        for k in continuous_keys:
            dim_signal = np.array([float(w.centroid_params.get(k, 0.0)) for w in self.windows])
            if len(dim_signal) > 1 and np.std(dim_signal) > 0:
                norm_dim = (dim_signal - np.mean(dim_signal)) / np.std(dim_signal)
                algo = rpt.Pelt(model="l2", min_size=1, jump=1).fit(norm_dim.reshape(-1, 1))
                try:
                    bkpts = algo.predict(pen=2.0)
                    switches = len(bkpts) - 1
                    print(f"  Dimension '{k}': {switches} switches")
                except Exception as e:
                    print(f"  Dimension '{k}': Pelt failed ({e})")
            else:
                print(f"  Dimension '{k}': 0 switches (Constant)")
        print()
        
        signal_data = []
        for w in self.windows:
            signal_data.append([float(w.centroid_params.get(k, 0.0)) for k in continuous_keys])
        
        signal = np.array(signal_data)
        if signal.shape[1] > 0 and len(signal) > 1:
            std = signal.std(axis=0)
            std[std == 0] = 1.0
            signal_norm = (signal - signal.mean(axis=0)) / std
            algo = rpt.Pelt(model="l2", min_size=1, jump=1).fit(signal_norm)
            for lam in lambdas:
                try:
                    breakpoints = algo.predict(pen=lam)
                    n_switches = len(breakpoints) - 1
                    print(f"  λ = {lam:5.1f} -> {n_switches} switches (All dimensions)")
                except Exception:
                    pass

        # Use the passed-in burn_in_windows cutoff to prevent nested-CV leakage into OOS
        burn_in = max(1, burn_in_windows)
        
        lambda_scores = {lam: 0.0 for lam in lambdas}
        
        for lam in lambdas:
            cumulative_score = 0.0
            for test_idx in range(burn_in, len(self.windows)):
                hist_windows = self.windows[:test_idx]
                test_window = self.windows[test_idx]
                
                # Pick policy using historical windows
                chosen_params, _ = self._solve_ruptures_switching(hist_windows, lam)
                
                # Evaluate on the test window
                score = self._evaluate_param_on_window(chosen_params, test_window)
                cumulative_score += score
                
            lambda_scores[lam] = cumulative_score
            
        best_lam = max(lambda_scores, key=lambda_scores.get)
        print(f"Nested CV Lambda scores: {lambda_scores}")
        print(f"Selected λ: {best_lam}")
        
        print(f"WARNING: Sample Size. With {len(self.windows)} windows total, the sample is extremely thin.")
        print("Fitting a discrete HMM or clustered policy to this will suffer high variance.")
        print("Defaulting to the CV-optimal regularized policy to minimize regime whip-sawing.")
        return best_lam
