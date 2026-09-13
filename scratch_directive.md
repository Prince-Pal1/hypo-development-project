<USER_REQUEST>
# Master Execution Directive v2 (Detailed Edition)
## Regime-Adaptive Parameter Optimization — HypoTrader Strategy 1 (RangeScope)

**Read this whole document before writing any code.** It is written to be self-contained: it explains not just what to do, but why, with code sketches for the hardest parts. Where something in your actual codebase doesn't match what's described here (a renamed variable, a different table column), trust your own code and adapt — the *logic* described here is what must be preserved, not the exact variable names guessed at below.

---

## 0. If you are unsure about anything — STOP AND ASK

This document cannot anticipate every detail of your current implementation. If you hit a decision point that isn't clearly resolved below (e.g. "which exact table stores X", "what should happen if a window has zero trades"), **do not silently guess and proceed** — many of the problems in the last two runs (the window-count mismatch, the stale noise threshold) happened because ambiguous points got silently resolved one way without being flagged. Stop, report the ambiguity plainly, and wait for a decision, rather than picking an answer and continuing.

---

## 1. Glossary (so nothing here is ambiguous)

| Term | Meaning here |
|---|---|
| **WFO** | Walk-forward optimization: optimize parameters on a chunk of past data, test on the next unseen chunk, roll forward |
| **Window** | One chunk of the WFO — currently intended to be a fixed number of trading days (this is the thing that's currently inconsistent between runs — see Phase A) |
| **θ (theta)** | The parameter vector being optimized: `(ctc_points, tp_offset_y, tp_mode, sl_points, ctc_toggle)` — check your current `RangeScopeConfig` dataclass for exact field names, this list may have grown |
| **Plateau** | The region of parameter space whose backtest performance is within 90% of that window's best result — a *wide* plateau means the good performance isn't a fragile fluke sitting on one exact parameter combination |
| **Plateau centroid** | The "center of mass" of that plateau region — used instead of the single best point, because the single best point is more likely to be noise |
| **TPE** | Tree-structured Parzen Estimator — Optuna's default Bayesian search algorithm. It is *good at finding a good point fast* and *bad at giving you an honest map of the whole surface*, because it deliberately samples more near points that already look good |
| **QMC / Sobol** | Quasi-Monte Carlo / Sobol sequence — a sampling method that fills the space evenly by construction, with no bias toward "already looks good" regions. This is what plateau-width measurement actually needs |
| **Changepoint detection** | The statistics term for "did the value of some time series shift at some point, and where" — this is what the switching-cost DP is actually doing, and it's a well-studied problem with existing libraries (see `ruptures` in Phase C) |
| **PELT** | "Pruned Exact Linear Time" — a specific, published, peer-reviewed changepoint detection algorithm (Killick, Fearnhead & Eckley, 2012). It's implemented in the Python `ruptures` library. Use this instead of a hand-written DP |
| **DSR** | Deflated Sharpe Ratio — corrects a Sharpe ratio for the fact that you tried many parameter combinations before finding this one (the more you tried, the more you should discount the "best" result you found, because you were more likely to find a lucky one by chance) |
| **`n_total_trials`** | The input DSR needs: the *total* number of parameter combinations tried across the *entire* study, not just in one window. This must be summed across every window and every stratum |
| **Permutation test** | A test that answers "could this result just be random chance?" by shuffling the data's labels many times and checking how often shuffled (fake) data produces a result as extreme as your real result |
| **Leading vs. trailing signal** | A trailing signal (like last window's realized PnL) only tells you what already happened. A leading signal (like a macro score, or DXY trend) is available *before* the thing it's trying to predict happens. Only a leading signal can genuinely trigger a parameter switch *before* a regime change hurts performance |

---

## 2. Your operating principles (non-negotiable, apply to every phase below)

- **State your hypothesis before you test it**, in writing, before running the experiment.
- **A negative result is a valid, useful outcome.** If switches stay at zero even after every fix below, that is a real finding — report it plainly. Do not keep changing the methodology until you get a positive result; that is the definition of overfitting the *analysis*, not just the strategy.
- **Every headline number needs its uncertainty stated** — a Sharpe ratio without a confidence interval, from a Monte Carlo bootstrap or otherwise, is not a complete result.
- **Prefer robust regions over single best points**, throughout — plateau centroids over argmax, at every stage, not just in Phase B.
- **Log every hypothesis tested — including failed ones — into the `hypotheses` table** in `hypotrader.db`. Schema: `id, title, economic_rationale, asset_symbol, author, status, target_regimes, rules_summary, param_manifest, created_at, updated_at, retirement_reason`. A retired/failed hypothesis must have `retirement_reason` filled in — don't just leave the row incomplete or delete it.

---

## 3. Repo orientation — what exists and where

| File | What it does |
|---|---|
| `src/strategies/range_scope_v1.py` | `RangeScopeStrategy` + `RangeScopeConfig` — the strategy itself, 11:00 IST session-range logic |
| `src/execution/range_scope_simulator.py` | Forward bar-by-bar simulation of a `RangeScopeConfig`. **Contains a real structural break at 12:30 IST** — `mode_1_dynamic` has a live take-profit from 11:00, `mode_2_wait_1230` has none until 12:30, then both recompute against the expanded range. This is *why* the objective surface is genuinely non-smooth, not just noisy — keep using Bayesian/QMC-style search here, not gradient-based methods |
| `src/optimization/engine.py` | `OptunaStudyEngine` — runs the Optuna study, applies the DSR gate and a cost-floor gate |
| `src/optimization/dsr.py` | `StatisticalValidationEngine.deflated_sharpe_ratio()` — needs `n_total_trials` as an input; this must reflect the whole study, see Phase E |
| `src/optimization/trial_logger.py` | `TrialLogger`, writes to the `optimizer_trials` SQLite table — every trial's params + Sharpe/return/drawdown |
| `src/data/condition_generator.py` | `ConditionGenerator`, writes to `market_conditions` table, has a free-text `query_conditions()` SQL interface — this is where new volatility/regime columns should be added (Phase A) |
| `src/reports/visualizer.py` | The existing HTML/SVG report builder — extend this for all new charts, don't build a separate reporting system with standalone image files |
| `hypotrader.db` | SQLite DB with tables: `market_conditions`, `optimizer_trials`, `backtest_runs`, `hypotheses` |

Your own `walk_forward_optimizer.py` and `monte_carlo.py` (built across the last few sessions) aren't reflected above because I don't have direct visibility into them — locate them in your working tree and treat everything below as instructions to apply *to that existing code*, not as a rewrite from scratch.

---

## 4. What we currently know (context for why each phase exists)

- **Run 1** (Jan–Sep 2026, 9 months): 60 windows reported → implies ~3.25 trading days/window.
- **Run 2** (2 years, 251 trials/window, widened 5D range): 62 windows reported → implies ~8.1 trading days/window.
- These two numbers don't agree with each other or with the originally intended 10-trading-day window — **this must be resolved first (Phase A)**, because every other number in both reports is downstream of window construction.
- Both runs, despite very different trial budgets (30 vs. 251+) and search space sizes, produced **exactly 0 parameter switches**, Adaptive = Static to 3 decimal places. This consistency across very different setups suggests the cause may not be purely a search-fidelity problem — it may be that the *trailing PnL trigger itself* is too noisy at ~8–10 trades/window to ever justify switching. **Test this directly in Phase D**, don't assume Phase B/C fixes alone will change it.
- Run 2's Oracle Sharpe (4.626) is not comparable to Run 1's (3.382) — a bigger, higher-dimensional search space mechanically inflates a hindsight-cheating Oracle, independent of any real predictability. **The old noise threshold (2.176) is stale and must be recomputed (Phase E)** on the current space before the "Oracle Gap" claim means anything.

---

## 5. PHASE A — Fix the foundations (do this first, block everything else on it)

### A.1 — Resolve the window-length inconsistency

**Task:** Find the actual code that constructs windows (likely in `walk_forward_optimizer.py`). Add a debug print/log that outputs, for the current configuration: the total number of windows generated, and the first and last 5 windows' exact start/end dates. Run it once on Run 1's date range and once on Run 2's date range.

**What you're checking for:** does the window count you get match `(total_trading_days) / (intended_window_length)`? If not, the window construction logic has a bug or an undocumented behavior (e.g. it might be sizing windows by a minimum trade count rather than a fixed calendar length). Find out which, and fix it so window length is one fixed, explicit, documented number of trading days (10, as originally specified, unless there's a documented reason to change it).

**Self-check before moving on:** Can you state, in one sentence, "windows are exactly N trading days long, non-overlapping [or: overlapping by M days — decide and document which], and there are exactly K of them for date range X to Y"? If you can't state this cleanly, you haven't finished this step.

### A.2 — Calendar/event-awareness

**Why this matters:** gold's volatility is heavily driven by scheduled events (NFP, FOMC). A 10-day window that happens to contain an NFP print will show elevated volatility that has nothing to do with a slow-moving macro regime — if that gets fed into the regime classifier as "high vol regime," it's mislabeling a one-day event as a structural shift.

**Task:**
1. You already have infrastructure for this — the `gold-bias-engine` project scrapes ForexFactory for economic calendar data. Reuse that scraper (or its calendar output) rather than building a new one.
2. In `condition_generator.py`, add a boolean column to `market_conditions` (e.g. `is_event_day`) flagging any day containing a high-impact FOMC or NFP release.
3. When computing regime/volatility features (Hurst, realized vol, ADX, etc. — whichever your current code computes), either exclude event days from the rolling calculation, or compute the feature both with and without event days included and report both — don't silently blend them into one number.
4. Separately: compute all regime/volatility features using **only the 11:00–12:30 IST window** (the actual session the strategy trades), not full 24-hour volatility. A quiet day globally with a volatile 11:00–12:30 stretch (or the reverse) is what matters to this specific strategy — daily aggregate volatility can hide exactly the thing you're trying to measure.

**Self-check:** Can you point to the exact column(s) in `market_conditions` holding the event flag and the session-scoped (not full-day) volatility features? If the features are still computed on full-day bars, this step isn't done.

---

## 6. PHASE B — Fix surface mapping (QMC, not TPE, not a plain grid)

**Why not TPE:** TPE spends its trial budget concentrating near whatever region looks good early on. More trials (even 251, even 15,000+) doesn't fix this — it just means TPE clusters *more confidently* around the same region, at the expense of the far corners of the space. This directly corrupts plateau-width measurement, because a real wide plateau can look narrow simply because TPE never visited the edges.

**Why not a plain grid either:** with 5 dimensions now (continuous params + a boolean CTC toggle + categorical `tp_mode`), a literal grid across all dimensions jointly is combinatorially too expensive.

**The fix — stratify on the discrete dimensions, then QMC-sample the continuous ones:**

```python
import optuna
from optuna.samplers import QMCSampler

CONTINUOUS_RANGES = {
    "ctc_points": (CTC_MIN, CTC_MAX),      # use your current widened ranges
    "tp_offset_y": (Y_MIN, Y_MAX),
    "sl_points": (SL_MIN, SL_MAX),
}

def run_qmc_surface_map(tp_mode, ctc_toggle, window_start, window_end, n_trials=256):
    # n_trials as a power of 2 (128/256/512) gives better Sobol low-discrepancy coverage
    sampler = QMCSampler(qmc_type="sobol", scramble=True, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        ctc_points = trial.suggest_float("ctc_points", *CONTINUOUS_RANGES["ctc_points"])
        tp_offset_y = trial.suggest_float("tp_offset_y", *CONTINUOUS_RANGES["tp_offset_y"])
        sl_points = trial.suggest_float("sl_points", *CONTINUOUS_RANGES["sl_points"])
        config = RangeScopeConfig(
            scope_min_x=FIXED_X,          # fixed, not part of this search
            sl_points=sl_points,
            ctc_points=ctc_points,
            tp_offset_y=tp_offset_y,
            tp_mode=tp_mode,               # fixed per stratum, not searched here
            # ctc_toggle: use your actual field name for this boolean
        )
        result = run_backtest(config, window_start, window_end)  # use your existing backtest runner
        return result.sharpe_ratio  # or your DSR-adjusted / growth metric — be consistent across all phases

    study.optimize(objective, n_trials=n_trials)
    return study  # study.trials gives you the full sampled surface for plateau computation

# Stratify manually over the 4 discrete combinations:
all_surfaces = {}
for tp_mode in ["mode_1_dynamic", "mode_2_wait_1230"]:
    for ctc_toggle in [True, False]:
        all_surfaces[(tp_mode, ctc_toggle)] = run_qmc_surface_map(tp_mode, ctc_toggle, window_start, window_end)
```

**Compute the plateau from `study.trials`, not from a separate TPE run** — every trial in this QMC study contributes to the plateau/robustness measurement. TPE, if you use it at all, is only for an optional secondary refinement pass afterward within a region this QMC pass already flagged — it must never be the source of the plateau-width number itself.

**Self-check:** Plot the sampled `(ctc_points, tp_offset_y)` points for one window, one stratum. They should look like an even, space-filling scatter across the whole range box — not clustered in one corner. If they're clustered, something is wrong with the sampler setup (check that you're not accidentally still using the default TPE sampler).

---

## 7. PHASE C — Replace the hand-rolled DP with `ruptures`, and independently reproduce the 0-switches finding

**Why:** the switching-cost objective (`minimize -Σ R_k(θ_k) + λ·Σ 1[θ_k ≠ θ_{k-1}]`) is a textbook L0-penalized changepoint detection problem. This exact problem already has an exact, efficient, peer-reviewed solution — reinventing it by hand risks a subtle boundary/off-by-one bug that could silently produce "0 switches" even when that's wrong. You need an independent second implementation to trust the finding.

```python
# pip install ruptures
import ruptures as rpt
import numpy as np

# signal: the sequence of per-window plateau-centroid parameter values (from Phase B),
# one row per window, one column per continuous parameter you want to track for shifts.
# shape: (n_windows, n_params)
signal = np.array(per_window_plateau_centroids)

# Pelt directly supports a penalty term, which maps onto lambda most naturally:
algo = rpt.Pelt(model="l2", min_size=1, jump=1).fit(signal)

for lam in [LAMBDA_LOW, LAMBDA_MID, LAMBDA_HIGH]:   # the sensitivity grid required below
    changepoints = algo.predict(pen=lam)
    print(f"lambda={lam}: {len(changepoints) - 1} switches detected at windows {changepoints[:-1]}")
```

`model="l2"` detects shifts in the mean level of the signal — appropriate here since you're asking "did the optimal parameter's level shift." If results look odd, also try `model="rbf"` (detects more general distributional shifts, not just mean shifts) as a robustness check.

**Task:**
1. Run this against the exact same window/plateau-centroid data your existing hand-rolled DP used.
2. Compare: does `ruptures` also find 0 switches at whatever `λ`/`pen` value corresponds to your existing nested-CV-chosen penalty? If yes, that's real independent confirmation. If no, your hand-rolled DP has a bug — trust `ruptures` over it and find the bug.
3. Report the **lambda sensitivity grid** required in section 2 (3–5 `pen` values, showing how switch count changes) using `ruptures`, not the old hand-rolled version.

**Self-check:** You should be able to show, side by side, the hand-rolled DP's switch count and `ruptures`'s switch count at the same penalty value. If you can't produce that side-by-side comparison, this phase isn't done.

---

## 8. PHASE D — Test whether the trigger signal itself is the real problem

**Why this phase exists:** two very different runs both landed on exactly 0 switches. That consistency is itself a clue — it may mean the trailing-PnL trigger is simply too noisy at ~8–10 trades/window to ever justify a switch, regardless of how well the surface is mapped or how correct the DP implementation is. This needs to be tested directly, not assumed away by Phases B/C alone.

**Task:** build a second version of the switching decision that uses a **leading** signal instead of trailing window Sharpe/PnL:

| Feature | Source | Why leading, not trailing |
|---|---|---|
| Gold-bias-engine daily macro score (−5 to +5) | Your existing `gold-bias-engine` project | Built from ForexFactory/Reuters/CME FedWatch/FRED — reflects forward-looking macro context, not last week's trade outcomes |
| Higher-timeframe (1H/4H) realized volatility | Computed causally from bars *prior to* the current window only | A volatility shift often shows up on higher timeframes before it's visible in the 5m/session-level data this strategy trades |
| DXY trend | Dollar index — gold's dominant macro driver | Structural, not reactive to this strategy's own recent PnL |

Feed one or more of these into the same `ruptures`-based changepoint framework from Phase C (same `pen`/λ sensitivity grid), replacing the trailing-PnL signal with the leading feature(s), and rerun the switch-count comparison.

**Interpreting the result:**
- Switches still at 0 with leading signals too → meaningful evidence *against* the adaptive-parameter hypothesis in general, for this strategy. Report this plainly as a valid negative finding.
- Switches appear with leading signals but not trailing PnL → confirms the trailing-PnL trigger was the actual bottleneck, and any future adaptive-parameter work should be built around leading features from the start, not layered onto a reactive design.

---

## 9. PHASE E — Fix statistical validation

### E.1 — DSR trial count aggregation

**Task:** before computing `deflated_sharpe_ratio`, query the *combined* trial count across the entire study (every window, every stratum from Phase B):

```sql
SELECT COUNT(*) FROM optimizer_trials WHERE study_name LIKE 'rangescope_%';
```

Pass this combined number as `n_total_trials` — not a per-window or per-stratum count. With Phase B's stratified QMC design now generating far more trials per window than before, verify this aggregation explicitly rather than assuming it's already correct — print the number and sanity-check it against `(n_windows × n_strata × n_trials_per_stratum)`.

### E.2 — Re-run the permutation test properly

**Two fixes required from the last version:** (1) at least 1,000 shuffles, not 100; (2) the shuffle must be **block-preserving** — shuffle which window gets which parameter/regime assignment, keeping each window's own internal day sequence intact — not a naive day-by-day shuffle, which destroys the real short-range autocorrelation in gold returns and can distort the null distribution in either direction.

```python
import numpy as np

def block_preserving_permutation_test(per_window_static_sharpe, per_window_oracle_sharpe, n_shuffles=1000, seed=42):
    # per_window_static_sharpe, per_window_oracle_sharpe: arrays of length n_windows,
    # one value per window under each policy — adapt to however your code
    # currently represents per-window results.
    rng = np.random.default_rng(seed)
    n_windows = len(per_window_static_sharpe)

    true_gap = np.mean(per_window_oracle_sharpe) - np.mean(per_window_static_sharpe)

    null_gaps = []
    for _ in range(n_shuffles):
        # Shuffle WHICH WINDOW's oracle/static labels get paired together,
        # not the days within any window.
        perm = rng.permutation(n_windows)
        shuffled_gap = np.mean(per_window_oracle_sharpe[perm]) - np.mean(per_window_static_sharpe)
        null_gaps.append(shuffled_gap)

    p95_threshold = np.percentile(null_gaps, 95)
    return true_gap, p95_threshold, null_gaps
```

This is a sketch of the *logic* (block-preserving, adequate shuffle count) — adapt the exact indexing to however your actual code represents per-window Static/Oracle results; the important part is that whatever gets shuffled is window-level assignment, never individual days.

**Do not reuse the old 2.176 noise threshold** — it was computed on a smaller, 2D search space. Recompute it fresh on the current (Phase A–D corrected) pipeline before drawing any conclusion about whether the Oracle Gap is real.

---

## 10. Reporting template (use this exact structure for the next report, so it's comparable to future ones)

```
1. Window definition: exact length, overlap policy, count, date range covered
2. Regime feature scope: session-window-only (11:00-12:30 IST) vs full-day — state which; event-day handling — excluded / flagged / both reported
3. Surface mapping method: QMC (Sobol), n_trials per stratum, coverage sanity-check result
4. Changepoint method: ruptures (Pelt/Dynp), model type, lambda sensitivity grid results (table: lambda -> switch count), trailing-PnL result AND leading-signal result side by side
5. DSR: n_total_trials used, source query
6. Permutation test: shuffle count, block-preserving confirmation, true gap, 95th percentile null threshold, verdict
7. Headline Sharpe/growth numbers for Static / Adaptive(trailing) / Adaptive(leading) / Oracle, each with Monte Carlo 95% CI
8. Explicit statement: are these results comparable to prior runs, or does a methodology change (window length, search space) make comparison invalid?
9. Hypotheses table: list every hypothesis row logged this run, including retired ones with retirement_reason
```

---

## 11. Full ordered checklist (tick off in order — do not skip ahead)

- [ ] A.1 — Window length verified consistent, documented, and identical to what will be used going forward
- [ ] A.2 — Event-day flag added; regime features computed on 11:00–12:30 IST session window, event-day handling stated explicitly
- [ ] B — QMC/Sobol stratified surface mapping replaces TPE for plateau measurement; scatter-plot coverage check passed
- [ ] C — `ruptures`-based changepoint detection reproduces (or corrects) the hand-rolled DP's switch count; lambda sensitivity grid reported
- [ ] D — Leading-signal version of the switching trigger built and compared against trailing-PnL version
- [ ] E.1 — DSR `n_total_trials` verified as the true combined count across all windows/strata
- [ ] E.2 — Permutation test re-run with ≥1,000 block-preserving shuffles on the current search space; old threshold discarded
- [ ] Every hypothesis tested this cycle logged into `hypotheses`, including negative results with `retirement_reason`
- [ ] Final report follows the template in Section 10
- [ ] Explicit statement included on whether this run's numbers are comparable to prior runs
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-13T04:57:02+05:30.

The user's current state is as follows:
Active Document: /Users/prince/strategy_development/src/optimization/walk_forward_optimizer.py (LANGUAGE_PYTHON)
Cursor is on line: 5
Other open documents:
- /Users/prince/strategy_development/src/optimization/walk_forward_optimizer.py (LANGUAGE_PYTHON)
- /Users/prince/strategy_development/src/web/static/reports.html (LANGUAGE_HTML)
- /Users/prince/strategy_development/hypotheses/adaptive_range_scope/regime_adaptive_report.md (LANGUAGE_MARKDOWN)
- /Users/prince/strategy_development/src/execution/ambiguity_resolver.py (LANGUAGE_PYTHON)
</ADDITIONAL_METADATA>
