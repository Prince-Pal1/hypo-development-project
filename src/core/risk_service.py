"""
Risk Service Boundaries.
Enforces hard non-negotiable position-sizing, chain-depth, and circuit-breaker limits.
This module is deliberately isolated from the strategy and optimizer parameter manifests.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class RiskServiceLimits:
    """Immutable risk constraints. Optimizer/ML cannot alter or search these."""
    max_size_multiplier_per_chain_step: float = 1.0  # Hard ceiling: No martingale
    max_chain_depth: int = 3                         # Max consecutive legs
    max_cumulative_open_risk_pct: float = 2.0        # Cap across all open legs in chain
    daily_loss_circuit_breaker_pct: float = 3.0      # Stops all execution for day
    max_ml_scale_factor: float = 1.25                # Hard ceiling on ML confidence scaling


class RiskService:
    def __init__(self, limits: Optional[RiskServiceLimits] = None):
        self.limits = limits or RiskServiceLimits()
        self.daily_realized_loss_pct = 0.0

    def reset_daily_circuit_breaker(self) -> None:
        self.daily_realized_loss_pct = 0.0

    def record_loss(self, loss_pct: float) -> None:
        self.daily_realized_loss_pct += abs(loss_pct)

    def is_circuit_breaker_active(self) -> bool:
        return self.daily_realized_loss_pct >= self.limits.daily_loss_circuit_breaker_pct

    def sanitize_order_intent(
        self,
        base_units: float,
        chain_depth: int = 1,
        requested_multiplier: float = 1.0,
        ml_confidence_scale: float = 1.0,
        account_equity: float = 10000.0,
        sl_points: float = 10.0,
        point_value: float = 1.0,
    ) -> Tuple[bool, float, str]:
        """
        Clamps and approves or rejects an order intent from a Strategy or ML model.
        Returns: (approved: bool, sanctioned_units: float, reason: str)
        """
        # 1. Circuit breaker check
        if self.is_circuit_breaker_active():
            return False, 0.0, f"REJECTED: Daily loss {self.daily_realized_loss_pct:.2f}% >= circuit breaker {self.limits.daily_loss_circuit_breaker_pct:.2f}%"

        # 2. Chain depth check
        if chain_depth > self.limits.max_chain_depth:
            return False, 0.0, f"REJECTED: Chain depth {chain_depth} > max allowed {self.limits.max_chain_depth}"

        # 3. Hard-clamp multiplier to ceiling (prevents Martingale / optimizer creep)
        safe_mult = min(requested_multiplier, self.limits.max_size_multiplier_per_chain_step)

        # 4. Hard-clamp ML confidence scale factor
        safe_ml_scale = min(ml_confidence_scale, self.limits.max_ml_scale_factor) if ml_confidence_scale > 0 else 1.0

        # 5. Proposed size
        units = base_units * safe_mult * safe_ml_scale

        # 6. Cumulative open risk check
        risk_usd = units * sl_points * point_value
        risk_pct = (risk_usd / account_equity) * 100.0 if account_equity > 0 else 100.0
        if risk_pct > self.limits.max_cumulative_open_risk_pct:
            # Clamp units down so total trade risk equals the ceiling
            max_allowed_risk_usd = (self.limits.max_cumulative_open_risk_pct / 100.0) * account_equity
            units = max_allowed_risk_usd / (sl_points * point_value)

        return True, units, "APPROVED"
