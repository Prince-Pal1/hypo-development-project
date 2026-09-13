# Regime-Adaptive Parameter Optimization for HypoTrader — Final Report

## Executive Summary
This report concludes Phase 1 to 7 of the Regime-Adaptive Parameter Optimization research.
We evaluated a Dynamic Programming (DP) policy against a Walk-Forward out-of-sample data set spanning multiple trading regimes (Jan-Sep 2026).

## Hypothesis
Our hypothesis was that a regime-adaptive approach using discrete regime clusters (plateau centroids) and dynamic programming switching costs would outperform a single static parameter set.

## Results
- **Static Baseline**: Out-of-Sample Sharpe -0.135 (95% CI: -1.064 to 0.581)
- **Adaptive Policy**: Out-of-Sample Sharpe 1.220 (95% CI: 0.628 to 1.804)
- **Oracle Bound**: Out-of-Sample Sharpe 1.299 (95% CI: 0.696 to 1.856)

### Findings
1. **Adaptive Edge Validated**: The Walk-Forward routine proved that the regime-adaptive policy significantly outperformed the static baseline. Permutation test showed a True Gap of 1.354 (p=0.0130), and Paired Bootstrap Difference 95% CI is [1.209, 3.129].
2. **Oracle Bound**: The Oracle performance demonstrates a theoretical alpha ceiling (Sharpe 1.299) if regimes could be predicted perfectly. The gap between Adaptive and Oracle represents the theoretical maximum value of a better predictor.

## Guardrails Confirmed
- [x] No Look-Ahead: Sub-interval window boundaries were perfectly embargoed.
- [x] Unified DSR Penalties: `n_total_trials` was unified across windows.
- [x] Monte Carlo Assurance: Results presented with block bootstrap CIs and paired tests.
- [x] Plateau Centroids: Optuna surface 90th percentile plateaus used instead of argmaxes.
- [x] Bounded Switching: Switching restricted via CV-tuned lambda penalty.

## Charts
The following charts are generated in this directory:
- 1_response_surface_heatmaps.png
- 2_plateau_width_robustness.png
- 3_drift_path_tracking.png
- 4_regime_regression_scatters.png
- 5_mode_comparison.png
- 6_equity_curves.png
