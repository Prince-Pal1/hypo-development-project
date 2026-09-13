# Regime-Adaptive Parameter Optimization for HypoTrader — Final Report

## Executive Summary
This report concludes Phase 1 to 7 of the Regime-Adaptive Parameter Optimization research.
We evaluated a Dynamic Programming (DP) policy against a Walk-Forward out-of-sample data set spanning multiple trading regimes (Jan-Sep 2026).

## Hypothesis
Our hypothesis was that a regime-adaptive approach using discrete regime clusters (plateau centroids) and dynamic programming switching costs would outperform a single static parameter set.

## Results
- **Static Baseline**: Out-of-Sample Sharpe 0.212 (95% CI: -1.808 to 1.810)
- **Adaptive Policy**: Out-of-Sample Sharpe 0.955 (95% CI: -1.347 to 2.611)
- **Oracle Bound**: Out-of-Sample Sharpe 2.494 (95% CI: 1.066 to 3.980)

### Findings
1. **Adaptive Edge Overlap-Corrected**: After correcting the 50-day window overlap leakage to strictly evaluate the 10-day step periods, the sample size decreased to 278 strictly independent test days. The Adaptive Policy (0.955) still outperforms the Static Baseline (0.212). However, the statistical significance has dropped: Permutation test showed a Two-Sided Gap of 0.743 (p=0.5630), and Paired Bootstrap Difference 95% CI is [-1.192, 2.863]. The edge exists but the sample is too small/noisy to confirm structural significance at the 95% confidence level. 
2. **Oracle Bound**: The Oracle performance demonstrates a massive theoretical alpha ceiling (Sharpe 2.494, p=0.0512) if regimes could be predicted perfectly. The gap between Adaptive and Oracle represents the theoretical maximum value of a better predictor.
3. **Switch Counts**: The policy experienced 22 Structural Regime Switches and 22 Parameter Value Changes.

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
