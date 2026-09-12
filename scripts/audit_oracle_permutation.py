import sys
from pathlib import Path
import pandas as pd
import numpy as np
import datetime as dt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.execution.ambiguity_resolver import load_1min_dataframe
from scripts.test_adaptive_range_scope import evaluate_params_on_window, DEFAULT_PARQUET_PATH

def run_permutation_audit():
    print("--- Phase 0: Oracle Null Baseline Permutation Test ---")
    
    # We will pick 10 diverse parameter sets to act as our "parameter universe"
    param_universe = [
        {"sl_points": 8.0, "tp_offset_y": 1.0, "ctc_points": 10.0},
        {"sl_points": 9.0, "tp_offset_y": 1.0, "ctc_points": 12.0}, # The known optimal
        {"sl_points": 10.0, "tp_offset_y": 2.0, "ctc_points": 10.0},
        {"sl_points": 12.0, "tp_offset_y": 5.0, "ctc_points": 15.0},
        {"sl_points": 15.0, "tp_offset_y": 8.0, "ctc_points": 20.0},
        {"sl_points": 20.0, "tp_offset_y": 10.0, "ctc_points": 25.0},
        {"sl_points": 8.0, "tp_offset_y": 5.0, "ctc_points": 10.0},
        {"sl_points": 12.0, "tp_offset_y": 1.0, "ctc_points": 15.0},
        {"sl_points": 18.0, "tp_offset_y": 2.0, "ctc_points": 18.0},
        {"sl_points": 10.0, "tp_offset_y": 10.0, "ctc_points": 12.0}
    ]
    
    print("Loading data...")
    df_5m = pd.read_parquet(DEFAULT_PARQUET_PATH)
    if pd.api.types.is_numeric_dtype(df_5m["timestamp"]):
        df_5m["utc_time"] = pd.to_datetime(df_5m["timestamp"], unit="ms", utc=True)
    else:
        df_5m["utc_time"] = pd.to_datetime(df_5m["timestamp"], utc=True)
        
    df_1m = load_1min_dataframe(df_5m["utc_time"].min(), df_5m["utc_time"].max())
    
    # We only use Jan to Sep for testing (out of sample matching)
    start_dt = dt.date(2026, 1, 1)
    end_dt = dt.date(2026, 9, 30)
    
    print("Pre-computing daily returns for the parameter universe...")
    returns_matrix = {}
    for i, p in enumerate(param_universe):
        _, rets = evaluate_params_on_window(p, start_dt, end_dt, df_5m, df_1m)
        returns_matrix[i] = rets
        
    # Align all series into a single DataFrame
    df_returns = pd.DataFrame(returns_matrix).fillna(0.0)
    print(f"Matrix shape: {df_returns.shape} (Days x Params)")
    
    # 1. True Static
    # Find the parameter set with the highest Sharpe overall
    means = df_returns.mean()
    stds = df_returns.std()
    sharpes = (means / stds) * np.sqrt(252)
    best_static_idx = sharpes.idxmax()
    true_static_sharpe = sharpes[best_static_idx]
    
    # 2. True Oracle
    # For every 10-day window, pick the best parameter set
    window_size = 10
    n_days = len(df_returns)
    oracle_daily_returns = np.zeros(n_days)
    
    for start_idx in range(0, n_days, window_size):
        end_idx = min(start_idx + window_size, n_days)
        window_data = df_returns.iloc[start_idx:end_idx]
        
        # Best param for this window
        w_means = window_data.mean()
        w_stds = window_data.std().replace(0, 1e-9)
        w_sharpes = (w_means / w_stds)
        best_w_idx = w_sharpes.idxmax()
        
        oracle_daily_returns[start_idx:end_idx] = window_data[best_w_idx].values
        
    true_oracle_sharpe = (np.mean(oracle_daily_returns) / np.std(oracle_daily_returns)) * np.sqrt(252)
    true_gap = true_oracle_sharpe - true_static_sharpe
    
    print(f"True Static Sharpe: {true_static_sharpe:.3f}")
    print(f"True Oracle Sharpe: {true_oracle_sharpe:.3f}")
    print(f"True Oracle Gap:    {true_gap:.3f}\n")
    
    # 3. Permutation Test
    print("Running 100 permutations...")
    permuted_gaps = []
    permuted_oracles = []
    
    np.random.seed(42)
    returns_arr = df_returns.values # Shape: (N, K)
    
    for i in range(100):
        # Shuffle rows (days)
        shuffled_idx = np.random.permutation(n_days)
        shuffled_arr = returns_arr[shuffled_idx]
        
        # Static on shuffled data is exactly the same because row order doesn't change mean/std!
        # permuted_static_sharpe = true_static_sharpe
        
        # Oracle on shuffled data
        shuff_oracle_rets = np.zeros(n_days)
        for start_idx in range(0, n_days, window_size):
            end_idx = min(start_idx + window_size, n_days)
            window_data = shuffled_arr[start_idx:end_idx, :] # Shape (10, K)
            
            w_means = np.mean(window_data, axis=0)
            w_stds = np.std(window_data, axis=0, ddof=1)
            w_stds[w_stds == 0] = 1e-9
            w_sharpes = w_means / w_stds
            
            best_w_idx = np.argmax(w_sharpes)
            shuff_oracle_rets[start_idx:end_idx] = window_data[:, best_w_idx]
            
        shuff_oracle_sharpe = (np.mean(shuff_oracle_rets) / np.std(shuff_oracle_rets, ddof=1)) * np.sqrt(252)
        permuted_oracles.append(shuff_oracle_sharpe)
        permuted_gaps.append(shuff_oracle_sharpe - true_static_sharpe)
        
    p95_gap = np.percentile(permuted_gaps, 95)
    mean_shuff_oracle = np.mean(permuted_oracles)
    
    print("=== PERMUTATION TEST RESULTS ===")
    print(f"Mean Oracle Sharpe on Pure Noise: {mean_shuff_oracle:.3f}")
    print(f"95th Percentile Noise Gap:        {p95_gap:.3f}")
    print(f"True Gap vs 95th Percentile:      {true_gap:.3f} vs {p95_gap:.3f}")
    
    if true_gap > p95_gap:
        print("\nCONCLUSION: The Oracle Gap SURVIVES permutation. There is genuine regime predictability.")
    else:
        print("\nCONCLUSION: The Oracle Gap IS A STRUCTURAL ARTIFACT of small-sample fitting.")
        print("There is no regime alpha to chase. Do not proceed to ML phase.")

if __name__ == "__main__":
    run_permutation_audit()
