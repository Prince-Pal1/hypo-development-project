import sys
from pathlib import Path

sys.path.insert(0, "/Users/prince/strategy_development")

from src.cms.repository import DatabaseRepository
from src.cms.models import HypothesisCard, HypothesisStatus

def main():
    repo = DatabaseRepository("/Users/prince/strategy_development/hypotrader.db")
    
    # We are logging the adaptive range scope hypothesis
    hypo = HypothesisCard(
        id="HYPO-ADAPTIVE-RS-001",
        title="Regime-Adaptive Parameter Optimization — HypoTrader Strategy 1",
        economic_rationale="Adaptive parameter optimization will identify changes in regimes.",
        asset_symbol="XAUUSD",
        author="System",
        status=HypothesisStatus.RETIRED,
        target_regimes=["trending", "ranging", "volatile"],
        rules_summary="Uses pelt changepoint detection with QMCSampler over 10-day non-overlapping windows.",
        param_manifest={"window_size_days": 10, "step_size_days": 10},
        retirement_reason="Awaiting final results from WFO. If switches stay at zero, that is a real finding.",
    )
    repo.save_hypothesis(hypo)
    print("Hypothesis saved.")

if __name__ == "__main__":
    main()
