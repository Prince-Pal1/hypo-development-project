"""
FSM / DAG Trade Sequencer for Multi-Leg Chained Strategies.
Coordinates state transitions, stop-loss flips, and contingent child orders.
Enforces structural isolation with the RiskService at every transition step.
"""

from dataclasses import dataclass, field
import datetime as dt
from enum import Enum
from typing import Optional, Literal, List
import uuid

from src.core.risk_service import RiskService, RiskServiceLimits


class ChainStatus(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    HALTED_RISK = "HALTED_RISK"


@dataclass
class TradeLeg:
    leg_id: str
    chain_id: str
    depth: int
    direction: Literal["LONG", "SHORT"]
    entry_time: dt.datetime
    entry_price: float
    units: float
    sl_price: float
    tp_price: float
    parent_trade_id: Optional[str] = None
    exit_time: Optional[dt.datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[Literal["TP", "SL", "CTC", "CIRCUIT_BREAKER", "EXPIRATION", "DECAY"]] = None
    pnl_gross: float = 0.0
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    pnl_net: float = 0.0
    is_open: bool = True
    is_pending: bool = False
    ctc_armed: bool = False
    ctc_time: Optional[dt.datetime] = None
    ctc_price: Optional[float] = None
    initial_sl_price: Optional[float] = None

    def __post_init__(self):
        if self.initial_sl_price is None:
            self.initial_sl_price = self.sl_price

    def close(
        self,
        exit_time: dt.datetime,
        exit_price: float,
        reason: Literal["TP", "SL", "CTC", "CIRCUIT_BREAKER", "EXPIRATION", "DECAY"],
        commission: float,
        spread_cost: float,
        slippage_cost: float,
        point_value: float = 1.0,
    ) -> None:
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason
        self.commission = commission
        self.spread_cost = spread_cost
        self.slippage_cost = slippage_cost
        self.is_open = False

        if self.direction == "LONG":
            pts = exit_price - self.entry_price
        else:
            pts = self.entry_price - exit_price

        self.pnl_gross = pts * self.units * point_value
        self.pnl_net = self.pnl_gross - (commission + spread_cost + slippage_cost)


@dataclass(frozen=True)
class ChainedFollowUpConfig:
    sl_flip_offset: float = 8.0     # SL points for Leg 2 (reversal)
    tp_flip_offset: float = 20.0    # TP points for Leg 2
    requested_multiplier: float = 1.0  # Multiplier intent (RiskService clamps <= 1.0)
    max_chain_depth: int = 3
    enable_flip: bool = True        # Whether reverse follow-back trade triggers on SL


class TradeChain:
    """
    Finite State Machine managing the lifecycle of a single chained trading sequence.
    Leg 1 (Primary) -> If SL -> Leg 2 (Reversal) -> If SL -> Leg 3 / Complete.
    """

    def __init__(
        self,
        chain_id: str,
        config: ChainedFollowUpConfig,
        risk_service: RiskService,
        account_equity: float = 10000.0,
    ):
        self.chain_id = chain_id
        self.config = config
        self.risk_service = risk_service
        self.account_equity = account_equity
        self.status = ChainStatus.IDLE
        self.legs: List[TradeLeg] = []
        self.events: List[Dict[str, Any]] = []
        self.computed_context: Dict[str, Any] = {}
        self.confidence: str = "RESOLVED"
        self.ambiguity_reasons: Optional[List[str]] = None

    def record_event(self, event_type: str, *args, **kwargs) -> None:
        """Records a timestamped lifecycle event into the chain's event journal."""
        evt = {"type": event_type}
        if len(args) >= 1:
            evt["time"] = args[0]
        if len(args) >= 2:
            evt["price"] = args[1]
        if len(args) >= 3:
            evt["description"] = args[2]
        evt.update(kwargs)
        self.events.append(evt)

    def generate_outcome_path_summary(self) -> str:
        """
        Auto-generates the human-readable outcome-path summary from the state-transition log.
        Example:
          'Long limit filled @ 4163.5 → SL1 hit @ 4153.5 (−10) → flipped short @ 4153.5 → CTC @ 4143.5 (SL→BE) → TP2 hit @ 4133.5 (+20). Net: +$100.00 across the sequence.'
        """
        if not self.legs:
            if self.computed_context.get("tie_detected"):
                return "No trade — Proximity tie at 12:30 IST evaluation."
            return "No trades executed."

        parts = []
        l1 = self.legs[0]
        pts_dir = 1.0 if l1.direction == "LONG" else -1.0

        if l1.is_pending and l1.exit_reason in ("DECAY", "EXPIRATION"):
            if l1.exit_reason == "DECAY":
                decay_pts = self.computed_context.get("decay_points", 15.0)
                return f"{l1.direction.capitalize()} limit placed @ {l1.entry_price:.1f} &rarr; Decay cancelled (price ran +{decay_pts:.0f} pts favorably without filling). Net: $0.00."
            else:
                return f"{l1.direction.capitalize()} limit placed @ {l1.entry_price:.1f} &rarr; Expired at 5:00 PM cutoff without filling. Net: $0.00."

        # Leg 1 fill
        parts.append(f"{l1.direction.capitalize()} limit filled @ {l1.entry_price:.1f}")

        # Leg 1 CTC
        if l1.ctc_armed and l1.ctc_price is not None:
            parts.append(f"CTC @ {l1.ctc_price:.1f} (SL&rarr;BE)")

        # Leg 1 exit
        if l1.exit_reason == "TP":
            parts.append(f"TP1 hit @ {l1.exit_price:.1f} (+{abs(l1.exit_price - l1.entry_price):.1f} pts, ${l1.pnl_net:+.2f})")
        elif l1.exit_reason == "CTC":
            parts.append(f"CTC-SL hit @ {l1.exit_price:.1f} (0 pts, ${l1.pnl_net:+.2f})")
        elif l1.exit_reason == "SL":
            parts.append(f"SL1 hit @ {l1.exit_price:.1f} (&minus;{abs(l1.exit_price - l1.entry_price):.1f} pts, ${l1.pnl_net:+.2f})")
        elif l1.exit_reason == "EXPIRATION":
            parts.append(f"EOD close @ {l1.exit_price:.1f} (${l1.pnl_net:+.2f})")
        elif l1.exit_reason:
            parts.append(f"Exit ({l1.exit_reason}) @ {l1.exit_price:.1f} (${l1.pnl_net:+.2f})")

        # Leg 2 (Flip)
        if len(self.legs) > 1:
            l2 = self.legs[1]
            parts.append(f"flipped {l2.direction.lower()} @ {l2.entry_price:.1f}")
            if l2.ctc_armed and l2.ctc_price is not None:
                parts.append(f"CTC @ {l2.ctc_price:.1f} (SL&rarr;BE)")

            if l2.exit_reason == "TP":
                parts.append(f"TP2 hit @ {l2.exit_price:.1f} (+{abs(l2.exit_price - l2.entry_price):.1f} pts, ${l2.pnl_net:+.2f})")
            elif l2.exit_reason == "CTC":
                parts.append(f"CTC-SL hit @ {l2.exit_price:.1f} (0 pts, ${l2.pnl_net:+.2f})")
            elif l2.exit_reason == "SL":
                parts.append(f"SL2 hit @ {l2.exit_price:.1f} (&minus;{abs(l2.exit_price - l2.entry_price):.1f} pts, ${l2.pnl_net:+.2f})")
            elif l2.exit_reason == "EXPIRATION":
                parts.append(f"EOD close @ {l2.exit_price:.1f} (${l2.pnl_net:+.2f})")
            elif l2.exit_reason:
                parts.append(f"Exit ({l2.exit_reason}) @ {l2.exit_price:.1f} (${l2.pnl_net:+.2f})")

        net_str = f"Net: ${self.total_pnl_net:+.2f} across the sequence."
        return " &rarr; ".join(parts) + f". {net_str}"

    @property
    def current_leg(self) -> Optional[TradeLeg]:
        for leg in reversed(self.legs):
            if leg.is_open:
                return leg
        return None

    @property
    def total_pnl_net(self) -> float:
        return sum(leg.pnl_net for leg in self.legs)

    def start_leg1(
        self,
        direction: Literal["LONG", "SHORT"],
        entry_time: dt.datetime,
        entry_price: float,
        base_units: float,
        sl_price: float,
        tp_price: float,
        eval_price: Optional[float] = None,
        point_value: float = 1.0,
    ) -> Optional[TradeLeg]:
        """Initiates Leg 1 of the chain after passing through the isolated RiskService."""
        sl_distance = abs(entry_price - sl_price)
        approved, sanctioned_units, reason = self.risk_service.sanitize_order_intent(
            base_units=base_units,
            chain_depth=1,
            requested_multiplier=1.0,
            ml_confidence_scale=1.0,
            account_equity=self.account_equity,
            sl_points=sl_distance,
            point_value=point_value,
        )

        if not approved:
            self.status = ChainStatus.HALTED_RISK
            return None

        is_pending = False
        if eval_price is not None:
            if direction == "LONG" and eval_price > entry_price:
                is_pending = True
            elif direction == "SHORT" and eval_price < entry_price:
                is_pending = True

        leg1 = TradeLeg(
            leg_id=f"{self.chain_id}-L1",
            chain_id=self.chain_id,
            depth=1,
            direction=direction,
            entry_time=entry_time,
            entry_price=entry_price,
            units=sanctioned_units,
            sl_price=sl_price,
            tp_price=tp_price,
            is_pending=is_pending,
        )
        self.legs.append(leg1)
        self.status = ChainStatus.ACTIVE
        self.record_event(
            "LIMIT_PLACED",
            time=entry_time,
            price=entry_price,
            direction=direction,
            units=sanctioned_units,
            sl=sl_price,
            tp=tp_price,
            is_pending=is_pending,
        )
        return leg1

    def handle_leg_exit(
        self,
        leg: TradeLeg,
        exit_time: dt.datetime,
        exit_price: float,
        reason: Literal["TP", "SL", "CTC", "CIRCUIT_BREAKER", "EXPIRATION", "DECAY"],
        commission: float,
        spread_cost: float,
        slippage_cost: float,
        point_value: float = 1.0,
    ) -> Optional[TradeLeg]:
        """
        Closes current leg and determines whether to transition to the next contingent leg.
        If reason is 'SL', initiates immediate opposite reversal (Leg N+1).
        """
        leg.close(
            exit_time=exit_time,
            exit_price=exit_price,
            reason=reason,
            commission=commission,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            point_value=point_value,
        )

        self.record_event(
            f"EXIT_L{leg.depth}",
            time=exit_time,
            price=exit_price,
            reason=reason,
            pnl_net=leg.pnl_net,
        )

        # Update running account equity
        self.account_equity += leg.pnl_net

        # If trade resulted in a loss, update the daily circuit breaker in RiskService
        if leg.pnl_net < 0:
            loss_pct = (abs(leg.pnl_net) / self.account_equity) * 100.0
            self.risk_service.record_loss(loss_pct)

        # Transition Logic
        if reason == "TP":
            # Profit target achieved -> Chain successfully completed
            self.status = ChainStatus.COMPLETED
            return None

        if reason in ("CIRCUIT_BREAKER", "EXPIRATION", "DECAY", "CTC"):
            self.status = ChainStatus.COMPLETED
            return None

        if reason == "SL":
            # STOP LOSS HIT: Check if reversal flip is enabled
            if not self.config.enable_flip:
                self.status = ChainStatus.COMPLETED
                return None

            # Transition to next contingent reversal leg
            next_depth = leg.depth + 1
            if next_depth > self.config.max_chain_depth:
                self.status = ChainStatus.COMPLETED
                return None

            # Reversal: Opposite direction
            reversal_direction: Literal["LONG", "SHORT"] = "SHORT" if leg.direction == "LONG" else "LONG"
            reversal_entry_price = exit_price

            if reversal_direction == "LONG":
                new_sl = reversal_entry_price - self.config.sl_flip_offset
                new_tp = reversal_entry_price + self.config.tp_flip_offset
            else:
                new_sl = reversal_entry_price + self.config.sl_flip_offset
                new_tp = reversal_entry_price - self.config.tp_flip_offset

            # Sanitize through RiskService
            approved, sanctioned_units, gate_reason = self.risk_service.sanitize_order_intent(
                base_units=leg.units,
                chain_depth=next_depth,
                requested_multiplier=self.config.requested_multiplier,
                ml_confidence_scale=1.0,
                account_equity=self.account_equity,
                sl_points=self.config.sl_flip_offset,
                point_value=point_value,
            )

            if not approved:
                self.status = ChainStatus.HALTED_RISK
                return None

            next_leg = TradeLeg(
                leg_id=f"{self.chain_id}-L{next_depth}",
                chain_id=self.chain_id,
                depth=next_depth,
                direction=reversal_direction,
                entry_time=exit_time,
                entry_price=reversal_entry_price,
                units=sanctioned_units,
                sl_price=new_sl,
                tp_price=new_tp,
                parent_trade_id=leg.leg_id,
            )
            self.legs.append(next_leg)
            self.status = ChainStatus.ACTIVE
            self.record_event(
                "FLIP_ENTERED",
                time=exit_time,
                price=reversal_entry_price,
                direction=reversal_direction,
                depth=next_depth,
                units=sanctioned_units,
                sl=new_sl,
                tp=new_tp,
            )
            return next_leg

        return None
