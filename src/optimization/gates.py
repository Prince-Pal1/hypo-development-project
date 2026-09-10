"""
Precondition validation gates for the optimization pipeline.
Enforces hard thresholds (e.g. Cost Floor) prior to executing expensive backtest simulations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostFloorGate:
    """
    Computes total round-trip cost as a percentage of Stop Loss distance.
    Rejects any parameter set where cost exceeds max_cost_sl_ratio (default 15%).
    
    Fee components for XAUUSD (based on IC Markets cTrader Raw Spread):
      - Spread: 0.12 pts (0.30 pips / average liquid ECN)
      - Slippage: 0.06 pts (0.03 entry + 0.03 exit)
      - Commission: 0.07 pts ($7.00 round turn per 100oz standard lot @ gold prices)
      Total default round-turn cost: ~0.25 points.
    """
    spread_pts: float = 0.12
    slippage_pts: float = 0.06
    commission_pts: float = 0.07
    max_cost_sl_ratio: float = 0.15  # 15% threshold

    @property
    def total_cost_pts(self) -> float:
        return self.spread_pts + self.slippage_pts + self.commission_pts

    def evaluate(self, sl_points: float) -> tuple[bool, float, str]:
        """
        Evaluates whether a proposed SL distance survives the cost floor.
        Returns: (passed: bool, cost_sl_ratio: float, message: str)
        """
        if sl_points <= 0:
            return False, float("inf"), "REJECTED: sl_points must be > 0"

        ratio = self.total_cost_pts / sl_points
        if ratio > self.max_cost_sl_ratio:
            return (
                False,
                ratio,
                f"REJECTED: Cost/SL ratio {ratio*100.0:.2f}% exceeds ceiling {self.max_cost_sl_ratio*100.0:.1f}% (total cost: {self.total_cost_pts:.2f} pts, sl: {sl_points:.2f} pts)"
            )
        return True, ratio, f"PASSED: Cost/SL ratio {ratio*100.0:.2f}% <= {self.max_cost_sl_ratio*100.0:.1f}%"
