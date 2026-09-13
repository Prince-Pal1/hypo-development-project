# Regime-Adaptive Optimization (WFO) - Final Research Findings

You are an advanced quantitative AI agent. I am providing you with the final results and findings from our Regime-Adaptive Parameter Optimization research. Use this context to inform our next steps, hypothesis generation, or structural strategy improvements.

## 1. Context & Architecture
We ran a Walk-Forward Optimization (WFO) over 2 years of 1-minute/5-minute intraday data (Aug 2024 to Sept 2026), partitioned into 62 non-overlapping business-day trading windows. For each window, an Optuna engine mapped the response surface using a Quasi-Monte Carlo (Sobol) sampler.

We applied a Dynamic Programming (DP) algorithm (`ruptures.Pelt`) with a cross-validated switching penalty ($\lambda$) to decide when to update the parameters (Adaptive Policy) versus when to hold them constant, combating overfitting.

## 2. The Critical Bugs & Resolutions
Prior runs reported "optimal static baseline parameters" which were discovered to be mathematical artifacts. 
- **The Zero-Variance Bug:** A dummy return of 0.0 for `annualized_return` collapsed the expected log-growth calculation to exactly zero, flattening the entire parameter surface and causing the optimizer to select arbitrary artifact parameters.
- **The $N=1$ Sharpe Explosion:** The system was silently accepting parameter sets that fired only 1 trade in 10 days, generating astronomically unstable Sharpe Ratios (e.g., 39.316 and -12.46) due to near-zero standard deviation.
- **Data Leakage in Windows:** Weekend gaps were artificially compressing window lengths.
- **Distance Normalization:** Parameter distances weren't standardized, causing arbitrary scales (like `tp_offset_y` and `ctc_points`) to skew proximity.
- **The Fix:** We implemented zero-padding for non-trade days, instituted minimum-trade floor limits (with $-np.inf$ penalties), strictly filtered for business days, and applied Z-score normalization to the distances to restore statistical rigor to the micro-window measurements.

## 3. Final Validated Out-of-Sample (OOS) Results
With mathematical rigor restored, the DP model triggers adaptive policy switches across the windows. The performance is aggregated and validated via a block-preserving permutation test (1000 shuffles) and paired bootstrap tests.

- **Static Baseline (OOS Sharpe):** -0.135
  *(95% Confidence Interval: [-1.064, 0.581])*
- **Adaptive Policy (OOS Sharpe):** 1.220
  *(95% Confidence Interval: [0.628, 1.804])*
- **Oracle Bound (Perfect Hindsight Sharpe):** 1.299
  *(95% Confidence Interval: [0.696, 1.856])*

### Statistical Significance:
- Permutation Test (Adaptive > Static) True Gap: 1.354, p-value: 0.0130
- Paired Bootstrap Diff (Adaptive - Static) 95% CI: [1.209, 3.129]

## 4. Key Takeaways for Future Agents
1. **Adaptive Edge Validated:** The Adaptive DP policy successfully outperforms the Static Baseline in strict out-of-sample testing, proving that regime-switching holds a structural edge over static "one-size-fits-all" parameters for this strategy. The negative Static Sharpe confirms that the strategy requires regime adaptation to survive volatility drag.
2. **Oracle Gap / Alpha Ceiling:** The theoretical ceiling of the Oracle Bound indicates that our trailing-PnL/centroid-drift trigger captures most of the available regime alpha, getting very close to the Oracle bound (1.22 vs 1.299). 
3. **Next Steps:** Future research should focus on replacing the trailing DP trigger with *leading* macroeconomic indicators (e.g., Volatility regimes, ADX expansions, event-day boolean flags) to potentially exceed the historical Oracle Bound by anticipating shifts rather than reacting to them.

## 5. Visual Evidence & Observations
The WFO generated 6 key interactive Plotly HTML charts (migrated from static PNGs to allow deep zoom and precise parameter inspection on hover). Below are the analytical observations for each:

**Chart 1 (Response Surface Heatmaps):** The response surface heatmap reveals the fitness landscape mapping Stop Loss (SL) points vs Take Profit (TP) offset Y. It visualizes the high-Sharpe "plateau" regions where parameters are stable, rather than isolated brittle peaks.

**Chart 2 (Plateau Width Robustness):** This scatter plot visualizes robustness. It confirms that the selected parameters belong to a dense cluster of positive outcomes (a wide plateau) rather than a single overfit outlier. 

**Chart 3 (Drift Path Tracking):** Tracks the optimal continuous parameters (SL and CTC points) over time. Categorical parameters like `ctc_enabled` have been decoupled from the continuous point tracker to prevent centroid pollution.

**Chart 4 (Regime Regression Scatters):** A regime-regression scatter plotting average Window Volatility (Parkinson) against the Optimal Take-Profit Offset ($y^*$). This chart helps identify direct, leading relationships between market regimes and structural parameter shifts.

**Chart 5 (Mode Comparison):** Evaluates the performance distributions of the different Take Profit (TP) exit modes across the regimes.

**Chart 6 (Equity Curves (OOS)):** Displays the final cumulative Out-of-Sample equity curve, constructed using the *exact real historical returns* (no simulations). It overlays the Static Baseline, the Adaptive Policy, and the theoretical Oracle Bound with Monte Carlo Confidence Bands. The Static Baseline drops below 1.0 consistently, mapping accurately to its negative Sharpe (-0.135) and demonstrating severe volatility drag on static parameters.
