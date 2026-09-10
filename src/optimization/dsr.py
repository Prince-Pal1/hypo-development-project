"""
Deflated Sharpe Ratio (DSR) & Probabilistic Sharpe Ratio (PSR) Engine.
Implements Marcos Lopez de Prado's formulations (2014, "The Deflated Sharpe Ratio")
correcting for selection bias and multi-trial parameter snooping.
"""

import math
import numpy as np
from scipy.stats import norm
from typing import List, Tuple

EULER_MASCHERONI = 0.57721566490153286


class StatisticalValidationEngine:
    @staticmethod
    def expected_max_sharpe(
        n_trials: int,
        sharpe_std: float,
    ) -> float:
        """
        Computes the expected maximum Sharpe ratio under the null hypothesis (zero true alpha)
        given N independent trials and empirical variance of trial Sharpe ratios.
        """
        if n_trials <= 1 or sharpe_std <= 0:
            return 0.0

        # Approximation from Lopez de Prado (2014)
        term1 = (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
        term2 = EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        return sharpe_std * (term1 + term2)

    @staticmethod
    def probabilistic_sharpe_ratio(
        observed_sharpe: float,
        benchmark_sharpe: float,
        n_samples: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Computes Probabilistic Sharpe Ratio (PSR) accounting for skewness and fat tails (kurtosis).
        """
        if n_samples <= 1:
            return 0.5

        # Variance of the Sharpe ratio estimator
        denom_term = 1.0 - skewness * observed_sharpe + ((kurtosis - 1.0) / 4.0) * (observed_sharpe ** 2)
        if denom_term <= 0:
            denom_term = 1e-6

        sr_std = math.sqrt(denom_term / (n_samples - 1.0))
        z = (observed_sharpe - benchmark_sharpe) / sr_std
        return float(norm.cdf(z))

    @classmethod
    def deflated_sharpe_ratio(
        cls,
        observed_sharpe: float,
        trial_sharpe_distribution: List[float],
        n_total_trials: int,
        n_samples: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> Tuple[float, float, float]:
        """
        Computes DSR given:
          - observed_sharpe: Sharpe ratio of the candidate strategy
          - trial_sharpe_distribution: List of Sharpe ratios from ALL evaluated Optuna trials
          - n_total_trials: Total trials N (including pruned)
          - n_samples: Number of observations/trades in backtest

        Returns: (dsr_score: float, expected_max_sr: float, trial_variance: float)
        """
        if len(trial_sharpe_distribution) < 2 or n_total_trials <= 1:
            return 0.5, 0.0, 0.0

        trial_arr = np.array(trial_sharpe_distribution, dtype=float)
        sharpe_std = float(np.std(trial_arr, ddof=1))
        sharpe_var = sharpe_std ** 2

        # Expected maximum Sharpe under data snooping null hypothesis
        exp_max_sr = cls.expected_max_sharpe(n_total_trials, sharpe_std)

        # DSR is PSR against the expected maximum Sharpe threshold
        dsr_score = cls.probabilistic_sharpe_ratio(
            observed_sharpe=observed_sharpe,
            benchmark_sharpe=exp_max_sr,
            n_samples=n_samples,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        return dsr_score, exp_max_sr, sharpe_var
