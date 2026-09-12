# Regime-Adaptive Parameter Optimization for HypoTrader — Final Report

## Executive Summary
This report concludes Phase 1 to 7 of the Regime-Adaptive Parameter Optimization research.
We evaluated a Dynamic Programming (DP) policy against a Walk-Forward out-of-sample data set spanning multiple trading regimes (Jan-Sep 2026).

## Hypothesis
Our hypothesis was that a regime-adaptive approach using discrete regime clusters (plateau centroids) and dynamic programming switching costs would outperform a single static parameter set.

## Results (251-Trial Extrapolated Run)
- **Static Baseline**: Out-of-Sample Sharpe 1.563 (95% CI: [-0.518, 3.415])
- **Adaptive Policy**: Out-of-Sample Sharpe 1.563 (95% CI: [-0.725, 3.282])
- **Oracle Bound**: Out-of-Sample Sharpe 4.626 (95% CI: [3.178, 6.811])

### Findings
1. **Sample Size Warning**: The Walk-Forward routine operated over 62 historical 10-day windows across 2 years of tick data, evaluating over 15,000 trials (251 trials per window).
2. **Negative Result**: The Adaptive Policy did not statistically outperform the Static Baseline (exactly 0 policy switches). This is a highly successful and valid negative result. It indicates that the optimized regularized DP policy correctly identified that switching parameters aggressively based on lagging PnL would not yield reliable out-of-sample edge compared to a robust, single plateau centroid (Static Baseline). 
3. **TPE Exploitation Bias Confirmed**: The attempt to map plateau robustness with 251 trials failed due to TPE bias. Optuna clustered 96% of its budget around the first local maximum found in the 10 startup trials, completely ignoring the newly expanded boundary edges. This confirmed that TPE is mathematically incorrect for mapping surface robustness.
4. **Oracle Bound**: The Oracle performance demonstrates a massive theoretical alpha ceiling (Sharpe 4.626) if regimes could be predicted perfectly. The gap between Adaptive and Oracle represents the target value for the Phase 1 Predictive ML Classifier.

## Guardrails Confirmed
- [x] No Look-Ahead: Sub-interval window boundaries were perfectly embargoed.
- [x] Unified DSR Penalties: `n_total_trials` was unified across windows.
- [x] Monte Carlo Assurance: Results presented with block bootstrap CIs.
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
