"""
Execution simulator for Range Sweep Version 2.

Handles limit decay (+15 pts unfilled), 5 PM IST limit cancel, CTC at +10 pts,
SL-flip on initial SL only (not CTC), and EOD forced exit.
"""

from dataclasses import dataclass
import datetime as dt
from typing import Optional, Tuple, Literal

from src.execution.simulator import FeeModel
from src.sequencer.fsm import TradeChain, TradeLeg
from src.execution.ambiguity_resolver import (
    is_ambiguous_bar, resolve_ambiguity, Confidence
)


@dataclass(frozen=True)
class RangeSweepV2SimConfig:
    ctc1_points: float = 10.0
    ctc2_points: float = 10.0
    decay_points: float = 15.0
    enable_flip: bool = True
    ctc_trigger_points: Optional[float] = None  # Backward compatibility fallback

    def get_ctc_points(self, depth: int) -> float:
        if self.ctc_trigger_points is not None:
            return self.ctc_trigger_points
        return self.ctc1_points if depth == 1 else self.ctc2_points


class RangeSweepV2Simulator:
    def __init__(
        self,
        fee_model: FeeModel | None = None,
        point_value: float = 1.0,
        config: RangeSweepV2SimConfig | None = None,
        df_1m: Optional[pd.DataFrame] = None,
    ):
        self.fee_model = fee_model or FeeModel()
        self.point_value = point_value
        self.config = config or RangeSweepV2SimConfig()
        self.df_1m = df_1m  # Optional 1-min data for ambiguity resolution

    def process_bar(
        self,
        chain: TradeChain,
        bar_time: dt.datetime,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        eval_price: Optional[float] = None,
        limit_cancel_utc: Optional[dt.datetime] = None,
    ) -> Optional[TradeLeg]:
        leg = chain.current_leg
        if leg is None or not leg.is_open:
            return None

        if leg.is_pending:
            return self._process_pending_bar(
                chain=chain,
                leg=leg,
                bar_time=bar_time,
                high_p=high_p,
                low_p=low_p,
                eval_price=eval_price,
                limit_cancel_utc=limit_cancel_utc,
            )

        self._maybe_arm_ctc(chain, leg, bar_time=bar_time, high_p=high_p, low_p=low_p)
        return self._process_active_bar(chain, leg, bar_time, high_p, low_p, open_p=open_p, close_p=close_p)

    def _process_pending_bar(
        self,
        chain: TradeChain,
        leg: TradeLeg,
        bar_time: dt.datetime,
        high_p: float,
        low_p: float,
        eval_price: Optional[float],
        limit_cancel_utc: Optional[dt.datetime],
    ) -> Optional[TradeLeg]:
        if limit_cancel_utc and bar_time >= limit_cancel_utc:
            chain.record_event(
                "EXPIRATION",
                time=bar_time,
                price=leg.entry_price,
                note="Expired at limit cutoff without fill",
            )
            return self._close_leg(chain, leg, bar_time, leg.entry_price, "EXPIRATION")

        if eval_price is not None:
            if leg.direction == "LONG" and high_p >= eval_price + self.config.decay_points:
                trigger_p = eval_price + self.config.decay_points
                chain.record_event(
                    "DECAY",
                    time=bar_time,
                    price=trigger_p,
                    eval_price=eval_price,
                    decay_pts=self.config.decay_points,
                    note=f"Price ran +{self.config.decay_points:.1f} pts favorably without filling limit",
                )
                return self._close_leg(chain, leg, bar_time, leg.entry_price, "DECAY")
            if leg.direction == "SHORT" and low_p <= eval_price - self.config.decay_points:
                trigger_p = eval_price - self.config.decay_points
                chain.record_event(
                    "DECAY",
                    time=bar_time,
                    price=trigger_p,
                    eval_price=eval_price,
                    decay_pts=self.config.decay_points,
                    note=f"Price ran -{self.config.decay_points:.1f} pts favorably without filling limit",
                )
                return self._close_leg(chain, leg, bar_time, leg.entry_price, "DECAY")

        if leg.direction == "LONG":
            if low_p <= leg.entry_price:
                leg.is_pending = False
                chain.record_event("LIMIT_FILLED", time=bar_time, price=leg.entry_price, depth=leg.depth)
                self._maybe_arm_ctc(chain, leg, bar_time=bar_time, high_p=high_p, low_p=low_p)
                return self._process_active_bar(chain, leg, bar_time, high_p, low_p, open_p=high_p, close_p=low_p)
        elif high_p >= leg.entry_price:
            leg.is_pending = False
            chain.record_event("LIMIT_FILLED", time=bar_time, price=leg.entry_price, depth=leg.depth)
            self._maybe_arm_ctc(chain, leg, bar_time=bar_time, high_p=high_p, low_p=low_p)
            return self._process_active_bar(chain, leg, bar_time, high_p, low_p, open_p=high_p, close_p=low_p)

        return None

    def _maybe_arm_ctc(self, chain: TradeChain, leg: TradeLeg, bar_time: dt.datetime, high_p: float, low_p: float) -> bool:
        if leg.ctc_armed:
            return True

        ctc_threshold = self.config.get_ctc_points(leg.depth)
        if ctc_threshold <= 0:
            return

        if leg.direction == "LONG":
            if high_p >= leg.entry_price + ctc_threshold:
                leg.ctc_time = bar_time
                leg.ctc_price = leg.entry_price + ctc_threshold
                old_sl = leg.sl_price
                leg.sl_price = leg.entry_price
                leg.ctc_armed = True
                chain.record_event(
                    f"CTC_STEP_L{leg.depth}",
                    time=bar_time,
                    trigger_price=leg.ctc_price,
                    old_sl=old_sl,
                    new_sl=leg.sl_price,
                    note=f"+{ctc_threshold:.1f} pts reached -> SL stepped to breakeven",
                )
        elif low_p <= leg.entry_price - ctc_threshold:
            leg.ctc_time = bar_time
            leg.ctc_price = leg.entry_price - ctc_threshold
            old_sl = leg.sl_price
            leg.sl_price = leg.entry_price
            leg.ctc_armed = True
            chain.record_event(
                f"CTC_STEP_L{leg.depth}",
                time=bar_time,
                trigger_price=leg.ctc_price,
                old_sl=old_sl,
                new_sl=leg.sl_price,
                note=f"+{ctc_threshold:.1f} pts reached -> SL stepped to breakeven",
            )

    def _process_active_bar(
        self,
        chain: TradeChain,
        leg: TradeLeg,
        bar_time: dt.datetime,
        high_p: float,
        low_p: float,
        open_p: float = 0.0,
        close_p: float = 0.0,
    ) -> Optional[TradeLeg]:
        ctc_threshold = self.config.get_ctc_points(leg.depth)

        # Check for ambiguity
        ambiguous, conflict_reasons = is_ambiguous_bar(
            direction=leg.direction,
            bar_high=high_p,
            bar_low=low_p,
            entry_price=leg.entry_price,
            sl_price=leg.sl_price,
            tp_price=leg.tp_price,
            ctc_threshold=ctc_threshold,
            ctc_armed=leg.ctc_armed,
        )

        if ambiguous:
            resolution = resolve_ambiguity(
                direction=leg.direction,
                bar_open=open_p if open_p != 0.0 else high_p,
                bar_high=high_p,
                bar_low=low_p,
                bar_close=close_p if close_p != 0.0 else low_p,
                bar_time=bar_time,
                entry_price=leg.entry_price,
                sl_price=leg.sl_price,
                tp_price=leg.tp_price,
                ctc_threshold=ctc_threshold,
                ctc_armed=leg.ctc_armed,
                conflict_reasons=conflict_reasons,
                df_1m=self.df_1m,
            )

            chain.confidence = resolution.tier.value
            chain.ambiguity_reasons = conflict_reasons

            if resolution.event_type == "TP_HIT":
                fill = resolution.price - self.fee_model.base_slippage_pts if leg.direction == "LONG" else resolution.price + self.fee_model.base_slippage_pts
                return self._close_leg(chain, leg, bar_time, fill, "TP")
            elif resolution.event_type == "SL_HIT":
                fill = leg.sl_price - self.fee_model.base_slippage_pts if leg.direction == "LONG" else leg.sl_price + self.fee_model.base_slippage_pts
                return self._close_leg(chain, leg, bar_time, fill, "SL")
            elif resolution.event_type == "CTC_SL_HIT":
                fill = leg.entry_price - self.fee_model.base_slippage_pts if leg.direction == "LONG" else leg.entry_price + self.fee_model.base_slippage_pts
                return self._close_leg(chain, leg, bar_time, fill, "CTC")
            elif resolution.event_type == "CTC_ARM":
                # CTC armed, trade survives
                if not leg.ctc_armed:
                    leg.ctc_armed = True
                    leg.ctc_time = bar_time
                    if leg.direction == "LONG":
                        leg.ctc_price = leg.entry_price + ctc_threshold
                    else:
                        leg.ctc_price = leg.entry_price - ctc_threshold
                    old_sl = leg.sl_price
                    leg.sl_price = leg.entry_price
                    chain.record_event(
                        f"CTC_STEP_L{leg.depth}",
                        time=bar_time,
                        trigger_price=leg.ctc_price,
                        old_sl=old_sl,
                        new_sl=leg.sl_price,
                        note=f"Ambiguity resolved ({resolution.tier.value}): CTC armed, trade survives",
                    )
            # SURVIVE: continue
            return None

        # No ambiguity — fast path
        exit_event: Optional[Tuple[Literal["TP", "SL", "CTC"], float]] = None

        if leg.direction == "LONG":
            sl_hit = low_p <= leg.sl_price
            tp_hit = high_p >= leg.tp_price

            if sl_hit and tp_hit:
                exit_event = self._resolve_sl_exit(leg, leg.sl_price - self.fee_model.base_slippage_pts)
            elif sl_hit:
                exit_event = self._resolve_sl_exit(leg, leg.sl_price - self.fee_model.base_slippage_pts)
            elif tp_hit:
                exit_event = ("TP", leg.tp_price - self.fee_model.base_slippage_pts)
        else:
            sl_hit = high_p >= leg.sl_price
            tp_hit = low_p <= leg.tp_price

            if sl_hit and tp_hit:
                exit_event = self._resolve_sl_exit(leg, leg.sl_price + self.fee_model.base_slippage_pts)
            elif sl_hit:
                exit_event = self._resolve_sl_exit(leg, leg.sl_price + self.fee_model.base_slippage_pts)
            elif tp_hit:
                exit_event = ("TP", leg.tp_price + self.fee_model.base_slippage_pts)

        if exit_event is None:
            return None

        reason, fill_price = exit_event
        return self._close_leg(chain, leg, bar_time, fill_price, reason)

    def _resolve_sl_exit(
        self,
        leg: TradeLeg,
        fill_price: float,
    ) -> Tuple[Literal["SL", "CTC"], float]:
        if leg.ctc_armed and abs(leg.sl_price - leg.entry_price) < 1e-6:
            return ("CTC", fill_price)
        return ("SL", fill_price)

    def _close_leg(
        self,
        chain: TradeChain,
        leg: TradeLeg,
        bar_time: dt.datetime,
        fill_price: float,
        reason: Literal["TP", "SL", "CTC", "EXPIRATION", "DECAY"],
    ) -> Optional[TradeLeg]:
        commission = self.fee_model.commission_per_unit * leg.units if reason not in ("EXPIRATION", "DECAY") else 0.0
        spread_cost = self.fee_model.spread_pts * leg.units * self.point_value if reason not in ("EXPIRATION", "DECAY") else 0.0
        slippage_cost = self.fee_model.base_slippage_pts * leg.units * self.point_value if reason not in ("EXPIRATION", "DECAY") else 0.0

        return chain.handle_leg_exit(
            leg=leg,
            exit_time=bar_time,
            exit_price=fill_price,
            reason=reason,
            commission=commission,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            point_value=self.point_value,
        )
