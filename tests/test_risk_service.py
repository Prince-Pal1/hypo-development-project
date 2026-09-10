"""
Unit tests for RiskService: testing all ceilings and sanitization logic.
"""

from src.core.risk_service import RiskService, RiskServiceLimits


def test_risk_service_multiplier_clamping():
    """Even if an optimizer or strategy requests multiplier=3.0, it is clamped to 1.0."""
    risk = RiskService()
    approved, units, reason = risk.sanitize_order_intent(
        base_units=10.0,
        chain_depth=1,
        requested_multiplier=3.0,  # Strategy requests 3x size
        ml_confidence_scale=1.0,
        account_equity=100000.0,
        sl_points=10.0,
    )
    assert approved is True
    assert units == 10.0  # Clamped to 10.0 * 1.0 = 10.0


def test_risk_service_ml_confidence_clamping():
    """Even if ML outputs extreme confidence scale=2.5, it is clamped to max 1.25x."""
    risk = RiskService()
    approved, units, reason = risk.sanitize_order_intent(
        base_units=10.0,
        chain_depth=1,
        requested_multiplier=1.0,
        ml_confidence_scale=2.5,  # ML requests 2.5x
        account_equity=100000.0,
        sl_points=10.0,
    )
    assert approved is True
    assert units == 12.5  # Clamped to 10.0 * 1.25 = 12.5


def test_risk_service_max_chain_depth():
    """Chain depth beyond 3 is rejected."""
    risk = RiskService()
    approved, units, reason = risk.sanitize_order_intent(
        base_units=10.0,
        chain_depth=4,  # Depth 4 exceeds max 3
        account_equity=100000.0,
        sl_points=10.0,
    )
    assert approved is False
    assert units == 0.0
    assert "> max allowed" in reason


def test_risk_service_circuit_breaker():
    """Daily realized loss >= 3.0% halts trading."""
    risk = RiskService()
    risk.record_loss(3.1)
    approved, units, reason = risk.sanitize_order_intent(
        base_units=10.0,
        account_equity=100000.0,
        sl_points=10.0,
    )
    assert approved is False
    assert "circuit breaker" in reason
