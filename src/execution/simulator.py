"""
Execution Simulator for Chained Orders.
Performs sub-minute / 1m bar replay with intra-bar high/low resolution,
applies calibrated IC Markets fees (spread, commission, slippage), and drives the FSM.
"""

from dataclasses import dataclass
import datetime as dt
from typing import Optional, List, Tuple
import pandas as pd

from src.sequencer.fsm import TradeChain, TradeLeg


@dataclass(frozen=True)
class FeeModel:
    spread_pts: float = 0.12          # 0.30 pips IC Markets XAUUSD raw spread
    commission_per_unit: float = 0.07 # $7.00 round-turn / 100oz -> $0.07/oz round-turn ($0.035/side)
    base_slippage_pts: float = 0.03   # 0.03 pts slippage per side


ZERO_FEE_MODEL = FeeModel(spread_pts=0.0, commission_per_unit=0.0, base_slippage_pts=0.0)


class ExecutionSimulator:
    def __init__(
        self,
        fee_model: Optional[FeeModel] = None,
        point_value: float = 1.0,
        ctc_threshold_pts: Optional[float] = 10.0,
    ):
        self.fee_model = fee_model or FeeModel()
        self.point_value = point_value
        self.ctc_threshold_pts = ctc_threshold_pts

    def _maybe_arm_ctc(
        self,
        chain: TradeChain,
        leg: TradeLeg,
        bar_time: dt.datetime,
        high_p: float,
        low_p: float,
    ) -> None:
        if self.ctc_threshold_pts is None or leg.ctc_armed:
            return

        if leg.direction == "LONG":
            if high_p >= leg.entry_price + self.ctc_threshold_pts:
                leg.ctc_time = bar_time
                leg.ctc_price = leg.entry_price + self.ctc_threshold_pts
                old_sl = leg.sl_price
                leg.sl_price = leg.entry_price
                leg.ctc_armed = True
                chain.record_event(
                    f"CTC_STEP_L{leg.depth}",
                    bar_time,
                    leg.ctc_price,
                    f"+{self.ctc_threshold_pts:.1f} pts reached -> SL stepped to breakeven",
                    old_sl=old_sl,
                    new_sl=leg.sl_price,
                    trigger_price=leg.ctc_price,
                )
        elif leg.direction == "SHORT":
            if low_p <= leg.entry_price - self.ctc_threshold_pts:
                leg.ctc_time = bar_time
                leg.ctc_price = leg.entry_price - self.ctc_threshold_pts
                old_sl = leg.sl_price
                leg.sl_price = leg.entry_price
                leg.ctc_armed = True
                chain.record_event(
                    f"CTC_STEP_L{leg.depth}",
                    bar_time,
                    leg.ctc_price,
                    f"+{self.ctc_threshold_pts:.1f} pts reached -> SL stepped to breakeven",
                    old_sl=old_sl,
                    new_sl=leg.sl_price,
                    trigger_price=leg.ctc_price,
                )

    def process_bar(
        self,
        chain: TradeChain,
        bar_time: dt.datetime,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
    ) -> Optional[TradeLeg]:
        """
        Processes a single price bar against the currently active leg in the chain.
        Returns the newly spawned leg if a transition occurred, else None.
        """
        leg = chain.current_leg
        if leg is None or not leg.is_open:
            return None

        # If order is pending limit entry, check if bar touches entry price
        if leg.is_pending:
            filled = False
            if leg.direction == "LONG" and low_p <= leg.entry_price:
                filled = True
            elif leg.direction == "SHORT" and high_p >= leg.entry_price:
                filled = True

            if filled:
                leg.is_pending = False
                leg.entry_time = bar_time
                chain.record_event(
                    "LIMIT_FILLED",
                    bar_time,
                    leg.entry_price,
                    f"Limit {leg.direction} filled @ {leg.entry_price:.2f}",
                    leg_id=leg.leg_id,
                    depth=leg.depth,
                )
            else:
                return None

        # Check for CTC threshold arming
        self._maybe_arm_ctc(chain, leg, bar_time, high_p, low_p)

        # Check for SL, CTC, or TP hits within the bar
        exit_event: Optional[Tuple[str, float]] = None

        if leg.direction == "LONG":
            sl_hit = low_p <= leg.sl_price
            tp_hit = high_p >= leg.tp_price

            if sl_hit and tp_hit:
                # Conservative quant rule: Assume SL/CTC hit first when ambiguous
                reason = "CTC" if (leg.ctc_armed and abs(leg.sl_price - leg.entry_price) < 1e-6) else "SL"
                exit_event = (reason, leg.sl_price - self.fee_model.base_slippage_pts)
            elif sl_hit:
                reason = "CTC" if (leg.ctc_armed and abs(leg.sl_price - leg.entry_price) < 1e-6) else "SL"
                exit_event = (reason, leg.sl_price - self.fee_model.base_slippage_pts)
            elif tp_hit:
                exit_event = ("TP", leg.tp_price - self.fee_model.base_slippage_pts)

        else:  # SHORT
            sl_hit = high_p >= leg.sl_price
            tp_hit = low_p <= leg.tp_price

            if sl_hit and tp_hit:
                reason = "CTC" if (leg.ctc_armed and abs(leg.sl_price - leg.entry_price) < 1e-6) else "SL"
                exit_event = (reason, leg.sl_price + self.fee_model.base_slippage_pts)
            elif sl_hit:
                reason = "CTC" if (leg.ctc_armed and abs(leg.sl_price - leg.entry_price) < 1e-6) else "SL"
                exit_event = (reason, leg.sl_price + self.fee_model.base_slippage_pts)
            elif tp_hit:
                exit_event = ("TP", leg.tp_price + self.fee_model.base_slippage_pts)

        if exit_event is None:
            return None

        reason, fill_price = exit_event

        # Calculate costs
        commission = self.fee_model.commission_per_unit * leg.units
        spread_cost = self.fee_model.spread_pts * leg.units * self.point_value
        slippage_cost = self.fee_model.base_slippage_pts * leg.units * self.point_value

        # Handle exit through the FSM sequencer
        next_leg = chain.handle_leg_exit(
            leg=leg,
            exit_time=bar_time,
            exit_price=fill_price,
            reason=reason,
            commission=commission,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            point_value=self.point_value,
        )

        return next_leg

