"""
Range Scope Strategy Version 1 — 11:00 AM IST Range Scope Momentum Strategy.

Evaluates range from Sydney Open (or Tokyo Open) up to 11:00 AM IST.
- Proximity check: Nearer Range Low -> SHORT (continuation/momentum towards low).
                    Nearer Range High -> LONG (continuation/momentum towards high).
- Scope Gate: distance to target boundary (scope) must be >= X (configurable).
- Entry: Immediate market fill at 11:00 AM IST.
- Stop Loss: entry +/- sl_points.
- Take Profit (Two Selectable Modes):
    * Mode 1 (Dynamic 12:30 Offset):
        - <= 12:29 IST: TP = Range Low (Short) / Range High (Long)
        - At 12:30 IST: TP = min(Range Low + y, price_1230) for Short
                             max(Range High - y, price_1230) for Long
    * Mode 2 (Wait Until 12:30):
        - <= 12:29 IST: No TP (only SL active)
        - At 12:30 IST: Recompute range up to 12:30, then set TP to
                        min(low_range_1230 + y, price_1230) for Short
                        max(high_range_1230 - y, price_1230) for Long
"""

from dataclasses import dataclass
import datetime as dt
from typing import Optional, Literal
import pandas as pd

from src.session.engine import SessionEngine, SessionWindow
from src.core.risk_service import RiskService


@dataclass(frozen=True)
class RangeScopeConfig:
    session_start: str = "sydney"               # "sydney" or "tokyo"
    eval_hour_ist: int = 11                     # 11:00 AM IST default
    eval_minute_ist: int = 0
    eval_time_ist: str = "11:00"                # "HH:MM" in IST
    scope_min_x: float = 5.0                    # Scope gate: min distance (pts) to boundary
    sl_points: float = 10.0                     # Stop Loss in points
    ctc_points: float = 10.0                    # CTC threshold in points (0 to disable)
    tp_offset_y: float = 3.5                    # 12:30 TP offset buffer Y in points
    tp_mode: Literal["mode_1_dynamic", "mode_2_wait_1230"] = "mode_1_dynamic"
    base_units: float = 10.0
    tie_tolerance: float = 0.05                 # pts


@dataclass(frozen=True)
class RangeScopeSignal:
    date: dt.date
    eval_time_utc: dt.datetime
    eod_utc: dt.datetime
    direction: Literal["LONG", "SHORT"]
    eval_price: float
    entry_price: float
    sl_price: float
    initial_tp: float                           # Range Low (Short) or Range High (Long)
    scope_pts: float
    range_high: float
    range_low: float
    dist_to_low: float
    dist_to_high: float
    tp_offset_y: float
    tp_mode: str
    units: float
    risk_sanitized: bool
    risk_notes: str
    ctc_points: float = 10.0
    eval_time_ist: str = "11:00"

    @property
    def midpoint(self) -> float:
        return (self.range_high + self.range_low) / 2.0

    @property
    def range_pts(self) -> float:
        return self.range_high - self.range_low


class RangeScopeStrategy:
    def __init__(
        self,
        config: RangeScopeConfig | None = None,
        risk_service: RiskService | None = None,
    ):
        self.config = config or RangeScopeConfig()
        eval_time_str = getattr(self.config, "eval_time_ist", "11:00") or "11:00"
        parts = eval_time_str.strip().split(":")
        eval_h = int(parts[0]) if len(parts) > 0 and parts[0] else self.config.eval_hour_ist
        eval_m = int(parts[1]) if len(parts) > 1 and parts[1] else self.config.eval_minute_ist
        self.session_engine = SessionEngine(
            session_start=self.config.session_start,
            eval_hour_ist=eval_h,
            eval_minute_ist=eval_m,
        )
        self.risk_service = risk_service or RiskService()

    def evaluate_day_context(
        self,
        target_date: dt.date,
        day_df: pd.DataFrame,
        account_equity: float = 50000.0,
    ) -> tuple[Optional[RangeScopeSignal], dict]:
        """Evaluates 11:00 AM IST session interval and scope filter."""
        window: SessionWindow = self.session_engine.get_session_window(target_date)

        if pd.api.types.is_numeric_dtype(day_df["timestamp"]):
            dt_series = pd.to_datetime(day_df["timestamp"], unit="ms", utc=True)
        else:
            dt_series = pd.to_datetime(day_df["timestamp"], utc=True)

        # Prior completed bars strictly before 11:00 AM IST evaluation window
        prior_mask = (dt_series >= window.session_start_utc) & (dt_series < window.eval_time_utc)
        prior_bars = day_df.loc[prior_mask]
        eval_candle = day_df.loc[dt_series == window.eval_time_utc]

        if len(prior_bars) < 10 or len(eval_candle) == 0:
            return None, {"reason": "insufficient_data", "bars_count": len(prior_bars)}

        # Current price at 11:00 AM IST is the starting price (open) of the 11:00 candle
        eval_price = float(eval_candle.iloc[0]["open"])

        # Range at the instant 11:00:00 AM IST strictly uses prior completed bars + open price at 11:00
        # (Does NOT leak future high/low/close of the 11:00-11:05 bar)
        range_high = max(float(prior_bars["high"].max()), eval_price)
        range_low = min(float(prior_bars["low"].min()), eval_price)
        midpoint = (range_high + range_low) / 2.0
        range_pts = range_high - range_low

        dist_to_low = eval_price - range_low
        dist_to_high = range_high - eval_price
        is_tie = abs(dist_to_low - dist_to_high) <= self.config.tie_tolerance

        context = {
            "target_date": target_date,
            "window": window,
            "session_name": window.session_name,
            "range_high": range_high,
            "range_low": range_low,
            "midpoint": midpoint,
            "range_pts": range_pts,
            "eval_price": eval_price,
            "dist_to_low": dist_to_low,
            "dist_to_high": dist_to_high,
            "tie_detected": is_tie,
            "scope_min_x": self.config.scope_min_x,
            "sl_points": self.config.sl_points,
            "tp_offset_y": self.config.tp_offset_y,
            "tp_mode": self.config.tp_mode,
        }

        if is_tie:
            context["reason"] = "tie"
            return None, context

        # Nearer low -> SHORT (momentum towards range_low)
        # Nearer high -> LONG (momentum towards range_high)
        if dist_to_low < dist_to_high:
            direction: Literal["LONG", "SHORT"] = "SHORT"
            scope = dist_to_low
            initial_tp = range_low
            sl_price = eval_price + self.config.sl_points
        else:
            direction = "LONG"
            scope = dist_to_high
            initial_tp = range_high
            sl_price = eval_price - self.config.sl_points

        context["direction"] = direction
        context["scope"] = scope

        # Scope Gate: Scope must be >= scope_min_x
        # Round to 1 decimal place to match UI display (toFixed(1)) and prevent
        # floating-point precision disagreements (e.g. 4.97 displayed as "5.0")
        if round(scope, 1) < self.config.scope_min_x:
            context["reason"] = "insufficient_scope"
            return None, context

        approved, sanctioned_units, reason = self.risk_service.sanitize_order_intent(
            base_units=self.config.base_units,
            chain_depth=1,
            requested_multiplier=1.0,
            ml_confidence_scale=1.0,
            account_equity=account_equity,
            sl_points=self.config.sl_points,
        )

        if not approved:
            context["reason"] = "risk_rejected"
            context["risk_notes"] = reason
            return None, context

        signal = RangeScopeSignal(
            date=target_date,
            eval_time_utc=window.eval_time_utc,
            eod_utc=window.eod_utc,
            direction=direction,
            eval_price=eval_price,
            entry_price=eval_price,
            sl_price=round(sl_price, 3),
            initial_tp=round(initial_tp, 3),
            scope_pts=round(scope, 3),
            range_high=range_high,
            range_low=range_low,
            dist_to_low=dist_to_low,
            dist_to_high=dist_to_high,
            tp_offset_y=self.config.tp_offset_y,
            tp_mode=self.config.tp_mode,
            units=sanctioned_units,
            risk_sanitized=approved,
            risk_notes=reason,
            ctc_points=self.config.ctc_points,
            eval_time_ist=getattr(self.config, "eval_time_ist", "11:00"),
        )
        return signal, context
