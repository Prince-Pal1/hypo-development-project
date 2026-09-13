"""
Monte Carlo validation tooling for Regime-Adaptive parameter optimization.
Provides block bootstrapping, permutation testing, and policy stress simulation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Optional

def block_bootstrap_returns(
    daily_returns: pd.Series, 
    num_paths: int = 1000, 
    block_size: int = 10,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate bootstrap resampled equity paths using block bootstrapping
    to preserve short-range autocorrelation.
    """
    if seed is not None:
        np.random.seed(seed)
        
    n = len(daily_returns)
    if n == 0:
        return np.zeros((num_paths, 0))
        
    returns_arr = daily_returns.values
    
    # Pre-compute blocks
    num_blocks = (n + block_size - 1) // block_size
    blocks = [returns_arr[i:i+block_size] for i in range(0, n, block_size)]
    
    paths = np.zeros((num_paths, n))
    
    for i in range(num_paths):
        # Sample blocks with replacement (oversample to guarantee length)
        sampled_blocks_idx = np.random.choice(len(blocks), size=num_blocks * 2, replace=True)
        sampled_returns = np.concatenate([blocks[idx] for idx in sampled_blocks_idx])
        # Trim to exact length
        paths[i] = sampled_returns[:n]
        
    return paths

def compute_confidence_interval(
    metric_func: Callable[[np.ndarray], float],
    bootstrap_paths: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Compute confidence intervals for a given metric across bootstrap paths.
    Returns (mean_metric, lower_bound, upper_bound).
    """
    if bootstrap_paths.size == 0 or bootstrap_paths.shape[1] == 0:
        return (0.0, 0.0, 0.0)
        
    metrics = np.array([metric_func(path) for path in bootstrap_paths])
    
    # Handle NaN or Inf from metric functions
    metrics = metrics[np.isfinite(metrics)]
    if len(metrics) == 0:
        return (0.0, 0.0, 0.0)
        
    return (
        float(np.mean(metrics)),
        float(np.percentile(metrics, alpha/2 * 100)),
        float(np.percentile(metrics, (1 - alpha/2) * 100))
    )

def permutation_test_regime_effect(
    returns_regime_a: np.ndarray,
    returns_regime_b: np.ndarray,
    metric_func: Callable[[np.ndarray], float],
    num_permutations: int = 1000,
    block_size: int = 10,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Check whether the apparent performance gap between two regimes is statistically significant
    by shuffling which sub-interval gets which regime label, using blocks to preserve autocorrelation.
    Returns (observed_diff, p_value).
    """
    if seed is not None:
        np.random.seed(seed)
        
    metric_a = metric_func(returns_regime_a)
    metric_b = metric_func(returns_regime_b)
    observed_diff = abs(metric_a - metric_b)
    
    combined = np.concatenate([returns_regime_a, returns_regime_b])
    n = len(combined)
    n_a = len(returns_regime_a)
    
    # Pre-compute blocks
    blocks = [combined[i:i+block_size] for i in range(0, n, block_size)]
    
    count_exceed = 0
    for _ in range(num_permutations):
        np.random.shuffle(blocks)
        permuted = np.concatenate(blocks)
        
        sim_a = permuted[:n_a]
        sim_b = permuted[n_a:]
        sim_diff = abs(metric_func(sim_a) - metric_func(sim_b))
        
        if np.isfinite(sim_diff) and sim_diff >= observed_diff:
            count_exceed += 1
            
    p_value = count_exceed / num_permutations
    return observed_diff, p_value

def simulate_ruin_risk(
    bootstrap_paths: np.ndarray,
    initial_capital: float = 50000.0,
    ruin_threshold: float = 40000.0
) -> float:
    """
    Simulate ruin risk (drawdown beyond threshold) across all paths.
    Returns the probability of hitting the ruin threshold.
    """
    if bootstrap_paths.size == 0 or bootstrap_paths.shape[1] == 0:
        return 0.0
        
    ruined_count = 0
    for path in bootstrap_paths:
        # Calculate equity curve path
        equity_curve = initial_capital * np.cumprod(1 + path)
        if np.any(equity_curve <= ruin_threshold):
            ruined_count += 1
            
    return ruined_count / len(bootstrap_paths)

def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized Sharpe ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    std_dev = np.std(returns)
    if std_dev == 0:
        return 0.0
        
    return float(np.mean(returns) / std_dev * np.sqrt(periods_per_year))

def expected_log_growth(returns: np.ndarray) -> float:
    """
    Kelly-style expected log growth rate E[log(1+r)]
    """
    if len(returns) == 0:
        return 0.0
        
    # Clip to prevent log(<=0) errors for extreme negative returns
    safe_returns = np.clip(returns, -0.999, None)
    return float(np.mean(np.log1p(safe_returns)))
