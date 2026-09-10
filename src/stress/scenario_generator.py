"""
Synthetic Stress-Scenario Generator for Adversarial Testing.
Injects hostile market conditions: flash-gap wicks, spread blowouts,
rapid double-whipsaws, and circuit-breaker triggers to stress-test the FSM and RiskService.
"""

from dataclasses import dataclass
import datetime as dt
from typing import List


@dataclass
class SyntheticBar:
    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float
    spread_pts: float = 0.12
    slippage_pts: float = 0.03


class ScenarioGenerator:
    """Generates synthetic adversarial price paths to stress-test execution logic."""

    @staticmethod
    def generate_double_whipsaw(
        start_time: dt.datetime,
        start_price: float = 2350.0,
        sl1_distance: float = 10.0,
        sl2_distance: float = 8.0,
    ) -> List[SyntheticBar]:
        """
        Generates a sequence where:
          1. Price drops to trigger Leg 1 Long Stop Loss.
          2. Price instantly reverses upward to trigger Leg 2 Short Stop Loss.
          3. Price drops again to trigger Leg 3 Long Stop Loss.
        Used to test max_chain_depth exhaustion.
        """
        bars = []
        t = start_time

        # Bar 1: Leg 1 entry baseline
        bars.append(SyntheticBar(t, start_price, start_price + 1.0, start_price - 1.0, start_price))

        # Bar 2: Sharp plunge that breaks Leg 1 SL (e.g. 2350 - 10 = 2340)
        t += dt.timedelta(minutes=1)
        sl1_level = start_price - sl1_distance
        bars.append(SyntheticBar(t, start_price - 2.0, start_price - 1.0, sl1_level - 1.0, sl1_level - 0.5))

        # Bar 3: Instant violent vertical spike that hits Leg 2 SL (reversal was Short at sl1_level)
        # Leg 2 entry is at sl1_level (~2340). Leg 2 SL is sl1_level + sl2_distance = 2348.
        t += dt.timedelta(minutes=1)
        sl2_level = sl1_level + sl2_distance + 2.0
        bars.append(SyntheticBar(t, sl1_level, sl2_level + 2.0, sl1_level - 1.0, sl2_level + 1.0))

        # Bar 4: Instant plunge again that breaks Leg 3 Long SL
        t += dt.timedelta(minutes=1)
        sl3_level = sl2_level - 12.0
        bars.append(SyntheticBar(t, sl2_level, sl2_level + 1.0, sl3_level - 2.0, sl3_level - 1.0))

        return bars

    @staticmethod
    def generate_spread_blowout(
        start_time: dt.datetime,
        start_price: float = 2350.0,
        normal_spread: float = 0.12,
        blowout_spread: float = 6.0,
    ) -> List[SyntheticBar]:
        """Injects a 50x spread expansion (news event) at the exact moment of SL trigger."""
        bars = []
        t = start_time
        bars.append(SyntheticBar(t, start_price, start_price + 0.5, start_price - 0.5, start_price, spread_pts=normal_spread))

        # Bar 2: Hits SL with massive spread blowout
        t += dt.timedelta(minutes=1)
        bars.append(SyntheticBar(t, start_price - 2.0, start_price - 1.0, start_price - 12.0, start_price - 11.0, spread_pts=blowout_spread, slippage_pts=0.50))
        return bars

    @staticmethod
    def generate_flash_crash_gap(
        start_time: dt.datetime,
        start_price: float = 2350.0,
        gap_points: float = 25.0,
    ) -> List[SyntheticBar]:
        """Price gaps 25 points through the stop loss in a single bar."""
        bars = []
        t = start_time
        bars.append(SyntheticBar(t, start_price, start_price + 1.0, start_price - 1.0, start_price))

        # Bar 2: Flash crash bar
        t += dt.timedelta(minutes=1)
        bars.append(SyntheticBar(t, start_price, start_price, start_price - gap_points, start_price - gap_points, slippage_pts=1.5))
        return bars
