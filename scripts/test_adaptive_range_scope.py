import argparse
import datetime as dt
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.walk_forward_optimizer import WalkForwardOptimizer, WindowResult
from src.optimization.engine import OptunaStudyEngine
from src.strategies.range_scope_v1 import RangeScopeStrategy, RangeScopeConfig
from src.core.risk_service import RiskService, RiskServiceLimits
from src.execution.range_scope_simulator import RangeScopeSimulator, RangeScopeTradeResult
from src.execution.simulator import FeeModel
from src.session.engine import SessionEngine
from src.execution.ambiguity_resolver import load_1min_dataframe
from src.research.monte_carlo import (
    block_bootstrap_returns,
    compute_confidence_interval,
    permutation_test_regime_effect,
    paired_bootstrap_test_regime_effect
)

DEFAULT_PARQUET_PATH = "/Users/prince/algo-trading/data/historical/XAUUSD_5m.parquet"

def evaluate_params_on_window(
    params: dict,
    start_date: dt.date,
    end_date: dt.date,
    df_5m: pd.DataFrame,
    df_1m: pd.DataFrame,
) -> tuple[float, pd.Series]:
    """Runs a backtest for the specified params over the given date range and returns (Sharpe, daily_returns)."""
    # Extract params with defaults
    sl_points = params.get("sl_points", 10.0)
    tp_offset_y = params.get("tp_offset_y", 3.5)
    ctc_enabled = params.get("ctc_enabled", True)
    ctc_points = params.get("ctc_points", 10.0) if ctc_enabled else 1000.0
    tp_mode = params.get("tp_mode", "mode_1_dynamic")
    scope_min_x = params.get("scope_min_x", 5.0)

    risk_service = RiskService(RiskServiceLimits(
        max_size_multiplier_per_chain_step=1.0,
        max_chain_depth=1,
        daily_loss_circuit_breaker_pct=3.0,
    ))

    config = RangeScopeConfig(
        session_start="sydney",
        eval_time_ist="11:00",
        scope_min_x=scope_min_x,
        sl_points=sl_points,
        ctc_points=ctc_points,
        tp_offset_y=tp_offset_y,
        tp_mode=tp_mode,
        base_units=10.0,
    )

    strategy = RangeScopeStrategy(config, risk_service)
    simulator = RangeScopeSimulator(fee_model=FeeModel(), point_value=1.0, df_1m=df_1m)

    # Filter data for the date range
    window_df = df_5m[(df_5m["utc_time"].dt.date >= start_date) & (df_5m["utc_time"].dt.date <= end_date)]
    unique_dates = sorted(window_df["utc_time"].dt.date.unique())

    account_equity = 50000.0
    daily_returns = []

    for target_date in unique_dates:
        # Extract the day's data up to 21:00 UTC
        window = strategy.session_engine.get_session_window(target_date)
        day_df = window_df[(window_df["utc_time"] >= window.sydney_open_utc) & (window_df["utc_time"] <= window.eod_utc)]
        
        signal, _ = strategy.evaluate_day_context(target_date, day_df, account_equity=account_equity)
        if signal is None or not signal.risk_sanitized:
            daily_returns.append(0.0) # Pad no-trade days with 0.0 return
            continue
            
        trade = simulator.simulate_day(signal, day_df, window.sydney_open_utc)
        
        daily_ret = trade.pnl_net / account_equity
        daily_returns.append(daily_ret)
        account_equity += trade.pnl_net

    if len(daily_returns) < 3:
        return -np.inf, 0.0, 0.0, 0, pd.Series(dtype=float)

    arr = np.array(daily_returns)
    mean_ret = float(np.mean(arr))
    std_ret = float(np.std(arr, ddof=1))
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
    
    ann_ret = mean_ret * 252
    
    # Calculate Max Drawdown
    cum_returns = np.cumprod(1 + arr)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / running_max
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    trades_count = int(np.sum(arr != 0.0))
    if trades_count == 0:
        return -np.inf, 0.0, max_dd, 0, pd.Series(daily_returns, index=unique_dates, dtype=float)
    
    return sharpe, ann_ret, max_dd, trades_count, pd.Series(daily_returns, index=unique_dates, dtype=float)

def run_walk_forward_validation():
    print("=== Walk-Forward Validation: Static vs Adaptive vs Oracle ===")
    
    parquet_path = DEFAULT_PARQUET_PATH
    df_5m = pd.read_parquet(parquet_path)
    if pd.api.types.is_numeric_dtype(df_5m["timestamp"]):
        df_5m["utc_time"] = pd.to_datetime(df_5m["timestamp"], unit="ms", utc=True)
    else:
        df_5m["utc_time"] = pd.to_datetime(df_5m["timestamp"], utc=True)
        
    df_1m = load_1min_dataframe(parquet_path.replace("5m.parquet", "1m.parquet"), reference_df=df_5m)
    min_ts = df_1m["timestamp"].min()
    if pd.api.types.is_numeric_dtype(df_1m["timestamp"]):
        min_start_date_1m = pd.to_datetime(min_ts, unit="ms", utc=True).strftime("%Y-%m-%d")
    else:
        min_start_date_1m = pd.to_datetime(min_ts, utc=True).strftime("%Y-%m-%d")
    
    def engine_factory(study_name, sub_interval_id, sub_interval_start, sub_interval_end):
        def suggest_params(trial):
            return {
                "sl_points": trial.suggest_float("sl_points", 5.0, 20.0, step=1.0),
                "tp_offset_y": trial.suggest_float("tp_offset_y", 1.0, 10.0, step=0.5),
                "ctc_enabled": trial.suggest_categorical("ctc_enabled", [True, False]),
                "ctc_points": trial.suggest_float("ctc_points", 5.0, 20.0, step=1.0),
                "scope_min_x": trial.suggest_float("scope_min_x", 0.0, 10.0, step=1.0),
                "tp_mode": trial.suggest_categorical("tp_mode", ["mode_1_dynamic", "mode_2_wait_1230"])
            }
            
        def objective(params):
            start_d = dt.datetime.strptime(sub_interval_start, "%Y-%m-%d").date()
            end_d = dt.datetime.strptime(sub_interval_end, "%Y-%m-%d").date()
            sharpe, ann_ret, max_dd, trades_count, _ = evaluate_params_on_window(params, start_d, end_d, df_5m, df_1m)
            return sharpe, ann_ret, max_dd, trades_count
            
        return OptunaStudyEngine(
            study_name=study_name,
            suggest_params_callback=suggest_params,
            objective_callback=objective,
            sub_interval_id=sub_interval_id,
            sub_interval_start=sub_interval_start,
            sub_interval_end=sub_interval_end,
        )

    optimizer = WalkForwardOptimizer(
        parquet_path=parquet_path,
        study_prefix="WFO_RangeScope",
        engine_factory=engine_factory,
        n_trials_per_window=251,
        window_size_days=60,
        step_size_days=10,
        min_start_date=min_start_date_1m,
    )
    
    optimizer.evaluate_windows()
    
    if len(optimizer.windows) < 3:
        print("Not enough windows to perform Walk-Forward.")
        return
        
    best_lambda = optimizer.select_lambda_cv()
    
    burn_in_windows = max(1, len(optimizer.windows) // 2)
    test_windows = optimizer.windows[burn_in_windows:]
    
    # 1. Static Baseline
    # Use the centroid of the very first window (or burn-in) for the entire test set
    static_params = optimizer.windows[0].centroid_params
    print(f"Static Baseline Params (from W0): {static_params}")
    
    static_returns = pd.Series(dtype=float)
    adaptive_returns = pd.Series(dtype=float)
    oracle_returns = pd.Series(dtype=float)
    
    regime_switch_count = 0
    param_change_count = 0
    last_adaptive_params = None
    last_regime_start_idx = None
    switch_dates = []
    
    for test_idx, t_window in enumerate(test_windows, start=burn_in_windows):
        last_hist_window = optimizer.windows[t_window.window_id - 1]
        oos_start_date = last_hist_window.end_date + dt.timedelta(days=1)
        oos_end_date = t_window.end_date

        print(f"\n--- Testing Window {t_window.window_id} True OOS ({oos_start_date} to {oos_end_date}) ---")
        
        # Static
        _, _, _, _, rets = evaluate_params_on_window(static_params, oos_start_date, oos_end_date, df_5m, df_1m)
        static_returns = pd.concat([static_returns, rets])
        
        # Adaptive
        adaptive_params, regime_start_idx = optimizer.synthesize_policy_for_window(t_window.window_id, best_lambda)
        print(f"Adaptive Params selected: {adaptive_params}")
        
        if last_adaptive_params is not None and adaptive_params != last_adaptive_params:
            param_change_count += 1
            
        if last_regime_start_idx is not None and regime_start_idx != last_regime_start_idx:
            regime_switch_count += 1
            switch_dates.append(oos_start_date)
            print(f"   => True Regime Start Index shifted to: {regime_start_idx}")
            
        last_adaptive_params = adaptive_params
        last_regime_start_idx = regime_start_idx
        
        _, _, _, _, rets = evaluate_params_on_window(adaptive_params, oos_start_date, oos_end_date, df_5m, df_1m)
        adaptive_returns = pd.concat([adaptive_returns, rets])
        
        # Oracle
        oracle_params = t_window.centroid_params
        print(f"Oracle Params (perfect hindsight): {oracle_params}")
        _, _, _, _, rets = evaluate_params_on_window(oracle_params, oos_start_date, oos_end_date, df_5m, df_1m)
        oracle_returns = pd.concat([oracle_returns, rets])
        
    print("\n=== Walk-Forward Out-Of-Sample Results ===")
    print(f"Adaptive Policy Regime Switch Count: {regime_switch_count}")
    print(f"Adaptive Policy Parameter Change Count: {param_change_count}")
    
    def report_performance(name: str, rets: pd.Series):
        if len(rets) < 3:
            print(f"{name}: Insufficient data.")
            return
            
        arr = rets.values
        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr, ddof=1))
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
        
        # Monte Carlo Assurance
        def sharpe_metric(path):
            mean_r = np.mean(path)
            std_r = np.std(path, ddof=1)
            return (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0
            
        paths = block_bootstrap_returns(rets, num_paths=1000, block_size=5)
        _, lower_ci, upper_ci = compute_confidence_interval(sharpe_metric, paths)
        
        print(f"{name} Out-of-Sample Sharpe: {sharpe:.3f}")
        print(f"   => 95% Confidence Interval: [{lower_ci:.3f}, {upper_ci:.3f}]")

    report_performance("Static Baseline", static_returns)
    report_performance("Adaptive Policy", adaptive_returns)
    report_performance("Oracle Bound   ", oracle_returns)
    
    def sharpe(arr):
        std = np.std(arr, ddof=1)
        return (np.mean(arr) / std) * np.sqrt(252) if std > 0 else 0.0

    print("\n=== Hypothesis Testing ===")
    
    # Paired Permutation Test
    diff_adaptive_static, p_val_ad_st = permutation_test_regime_effect(
        adaptive_returns.values, static_returns.values, metric_func=sharpe, num_permutations=5000, seed=42
    )
    print(f"Permutation Test (Adaptive vs Static) Two-Sided Gap: {diff_adaptive_static:.3f}, p-value: {p_val_ad_st:.4f}")
    
    diff_oracle_static, p_val_or_st = permutation_test_regime_effect(
        oracle_returns.values, static_returns.values, metric_func=sharpe, num_permutations=5000, seed=42
    )
    print(f"Permutation Test (Oracle vs Static) Two-Sided Gap: {diff_oracle_static:.3f}, p-value: {p_val_or_st:.4f}")

    # Paired Difference Bootstrap
    mean_diff, lower_diff, upper_diff = paired_bootstrap_test_regime_effect(
        adaptive_returns.values, static_returns.values, metric_func=sharpe, num_paths=1000, block_size=5, seed=42
    )
    print(f"\nPaired Bootstrap Diff (Adaptive - Static) 95% CI: [{lower_diff:.3f}, {upper_diff:.3f}]")
    
    # Save the real returns to disk for the visualizer
    out_dir = Path("/Users/prince/strategy_development/hypotheses/adaptive_range_scope")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_returns = pd.DataFrame({
        'static': static_returns,
        'adaptive': adaptive_returns,
        'oracle': oracle_returns
    })
    df_returns.to_csv(out_dir / "oos_returns.csv")
    
    with open(out_dir / "switch_dates.txt", "w") as f:
        for d in switch_dates:
            f.write(f"{d}\n")

if __name__ == "__main__":
    run_walk_forward_validation()
