"""
Cross-Strategy Correlation Engine.
Calculates pairwise correlation across strategy returns and validates portfolio diversification.
Enforces the Stage 6 portfolio gate: Cross-strategy correlation < 0.30.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


@dataclass
class CorrelationReport:
    correlation_matrix: Dict[str, Dict[str, float]]
    max_pairwise_correlation: float
    diversification_passed: bool
    pairs_violating_gate: List[Tuple[str, str, float]]


class CorrelationEngine:
    def __init__(self, max_allowed_correlation: float = 0.30):
        self.max_allowed_correlation = max_allowed_correlation

    def compute_correlation(
        self,
        strategy_returns: Dict[str, pd.Series],  # Dict of strategy_name -> daily_returns Series
    ) -> CorrelationReport:
        """
        Computes pairwise correlation matrix and checks against the 0.30 diversification threshold.
        """
        if len(strategy_returns) < 2:
            single_name = list(strategy_returns.keys())[0] if strategy_returns else "None"
            return CorrelationReport(
                correlation_matrix={single_name: {single_name: 1.0}},
                max_pairwise_correlation=1.0,
                diversification_passed=True,
                pairs_violating_gate=[],
            )

        df = pd.DataFrame(strategy_returns).dropna()
        corr_df = df.corr(method="pearson")

        corr_dict = corr_df.to_dict()
        violating_pairs = []
        max_corr = -1.0

        strategies = list(corr_df.columns)
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                s1 = strategies[i]
                s2 = strategies[j]
                val = float(corr_df.loc[s1, s2])
                if val > max_corr:
                    max_corr = val
                if val >= self.max_allowed_correlation:
                    violating_pairs.append((s1, s2, round(val, 3)))

        return CorrelationReport(
            correlation_matrix={col: {idx: round(float(corr_df.loc[idx, col]), 3) for idx in corr_df.index} for col in corr_df.columns},
            max_pairwise_correlation=round(max_corr, 3),
            diversification_passed=len(violating_pairs) == 0,
            pairs_violating_gate=violating_pairs,
        )
