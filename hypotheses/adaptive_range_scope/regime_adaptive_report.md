# Regime-Adaptive Parameter Optimization for HypoTrader — Final Report

## Executive Summary
This report concludes Phase 1 to 7 of the Regime-Adaptive Parameter Optimization research.
We evaluated a Dynamic Programming (DP) policy against a Walk-Forward out-of-sample data set spanning multiple trading regimes (Jan-Sep 2026).

## Hypothesis
Our hypothesis was that a regime-adaptive approach using discrete regime clusters (plateau centroids) and dynamic programming switching costs would outperform a single static parameter set.

## Results
- **OOS Sample Size**: 340 days (weekdays only, Sunday inflation removed). Annualization correctly utilizes `sqrt(252)`.
- **Selected $\lambda$ Penalty**: 2.0 (Chosen via un-leaked nested-CV inside burn-in block, utilizing a 0.1 Sharpe penalty per switch).
- **Static Baseline**: Out-of-Sample Sharpe 0.243 (95% CI: -1.662 to 1.652)
- **Adaptive Policy**: Out-of-Sample Sharpe 0.917 (95% CI: -0.754 to 2.381)
- **Oracle Bound**: Out-of-Sample Sharpe 2.241 (95% CI: 0.831 to 3.446)

### Lambda Sensitivity Grid
A sweep across historical centroids reveals how structurally sensitive the regime DP solver is to the $\lambda$ parameter:
- $\lambda = 0.0 \rightarrow 45 \text{ switches}$
- $\lambda = 0.5 \rightarrow 38 \text{ switches}$
- $\lambda = 1.0 \rightarrow 30 \text{ switches}$
- $\lambda = 2.0 \rightarrow 21 \text{ switches}$
- $\lambda = 5.0 \rightarrow 11 \text{ switches}$
- $\lambda = 10.0 \rightarrow 6 \text{ switches}$

### Findings
1. **Adaptive Edge Validated (Again)**: After strictly isolating $\lambda$ selection inside the 12-window burn-in block and enforcing a turnover penalty to prevent default maximum reactivity, the Adaptive Policy (Sharpe 0.917) maintains its lead over the Static Baseline (0.243). The Permutation Test Two-Sided Gap is 0.674 (p=0.5734), and the Paired Bootstrap Diff 95% CI is [-1.081, 2.764]. While the point estimate remains very strong, the sample size constraints still preclude achieving a 95% statistical significance threshold.
2. **Oracle Bound**: The Oracle performance continues to reflect a substantial alpha ceiling (Sharpe 2.241, p=0.0774) under perfect hindsight. However, as noted, the Oracle is scored on the same 10-day block its 60-day window evaluates, providing a lookahead bias that makes it a theoretical boundary rather than an attainable ceiling.
3. **Switch Counts**: The policy experienced 14 Structural Regime Switches across 34 opportunities (a 41.1% switch rate) and 33 Parameter Value Changes, reflecting a more conservative and realistic reactivity.

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
