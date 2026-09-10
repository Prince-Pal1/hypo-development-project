"""
Unit tests for CostFloorGate.
"""

from src.optimization.gates import CostFloorGate


def test_cost_floor_rejection_for_tiny_stop():
    """A stop loss of 1.0 point on Gold fails the 15% cost floor."""
    gate = CostFloorGate(spread_pts=0.12, slippage_pts=0.06, commission_pts=0.07, max_cost_sl_ratio=0.15)
    # Total cost is 0.25 pts. On 1.0 pt SL, ratio is 25% > 15%
    passed, ratio, msg = gate.evaluate(sl_points=1.0)
    assert passed is False
    assert ratio == 0.25
    assert "REJECTED" in msg


def test_cost_floor_acceptance_for_viable_stop():
    """A stop loss of 10.0 points on Gold passes comfortably (2.5% ratio vs 15% ceiling)."""
    gate = CostFloorGate(spread_pts=0.12, slippage_pts=0.06, commission_pts=0.07, max_cost_sl_ratio=0.15)
    # Total cost is 0.25 pts. On 10.0 pt SL, ratio is 2.5% <= 15%
    passed, ratio, msg = gate.evaluate(sl_points=10.0)
    assert passed is True
    assert ratio == 0.025
    assert "PASSED" in msg


def test_cost_floor_boundary():
    """Boundary test: SL = 0.25 / 0.15 = 1.6667 points."""
    gate = CostFloorGate(spread_pts=0.12, slippage_pts=0.06, commission_pts=0.07, max_cost_sl_ratio=0.15)
    passed_sub, _, _ = gate.evaluate(sl_points=1.5)
    passed_sup, _, _ = gate.evaluate(sl_points=2.0)
    assert passed_sub is False
    assert passed_sup is True
