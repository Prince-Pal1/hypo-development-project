# Regime-Adaptive Parameter Optimization for HypoTrader — Final Report

## Executive Summary
This report concludes Phase 1 to 7 of the Regime-Adaptive Parameter Optimization research.
We evaluated a Dynamic Programming (DP) policy against a Walk-Forward out-of-sample data set spanning multiple trading regimes (Jan-Sep 2026).

## Hypothesis
Our hypothesis was that a regime-adaptive approach using discrete regime clusters (plateau centroids) and dynamic programming switching costs would outperform a single static parameter set.

## Results
- **Static Baseline**: Out-of-Sample Sharpe 0.222 (95% CI: -1.694 to 1.592)
- **Adaptive Policy**: Out-of-Sample Sharpe 0.636 (95% CI: -1.301 to 2.154)
- **Oracle Bound**: Out-of-Sample Sharpe 2.040 (95% CI: 0.619 to 3.154)

### Findings
1. **Adaptive Edge Overlap-Corrected & Sample Expanded**: After correcting the lambda nested-CV leakage and expanding the True OOS sample size to N=410 days (by fixing `burn_in_windows=12`), the true structural reality of the reactive policy became clear. As the sample size grew, the Adaptive Sharpe dropped from 0.955 to 0.636, and the Permutation p-value worsened from 0.5630 to 0.7270 (Two-Sided Gap: 0.414). The point estimate did not hold up; the purely reactive regime DP solver is failing to capture a persistent forward-looking edge.
2. **Oracle Bound**: The Oracle performance remains theoretically strong (Sharpe 2.040), though its p-value also loosened slightly to 0.0888 across the wider 410-day test. The gap remains massive, but capturing it requires leading (forward-looking) predictors rather than lagging regime centroids.
3. **Switch Counts**: The policy experienced 33 Structural Regime Switches and 33 Parameter Value Changes over the extended window.

## Guardrails Confirmed
- [x] No Look-Ahead: Sub-interval window boundaries were perfectly embargoed.
- [x] Unified DSR Penalties: `n_total_trials` was unified across windows.
- [x] Monte Carlo Assurance: Results presented with block bootstrap CIs and paired tests.
- [x] Plateau Centroids: Optuna surface 90th percentile plateaus used instead of argmaxes.
- [x] Bounded Switching: Switching restricted via CV-tuned lambda penalty.

## Charts
The following interactive Plotly charts are generated in this directory:
- [1_response_surface_heatmaps.html](1_response_surface_heatmaps.html)
- [2_plateau_width_robustness.html](2_plateau_width_robustness.html)
- [3_drift_path_tracking.html](3_drift_path_tracking.html)
- [4_regime_regression_scatters.html](4_regime_regression_scatters.html)
- [5_mode_comparison.html](5_mode_comparison.html)
- [6_equity_curves.html](6_equity_curves.html)
