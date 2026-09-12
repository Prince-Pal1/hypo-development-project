"""
Range Sweep Version 2 — Sydney-to-12:30 IST range fade with CTC, decay, and SL-flip.

Range window: Sydney open → 12:30 PM IST.
Proximity tie → no trade. Limit valid until 5:00 PM IST or decay (+15 pts unfilled).
CTC at +10 pts in favor on both Position 1 and flip (Position 2).
"""

from dataclasses import dataclass
import datetime as dt
from typing import Optional, Literal
import pandas as pd

from src.session.engine import SessionEngine, SessionWindow
from src.core.risk_service import RiskService


@dataclass(frozen=True)
class RangeSweepV2Config:
    x_offset: float = 3.5
    sl_points: float = 10.0          # sl1
    tp_points: float = 20.0          # tp1
    sl2_points: float = 10.0         # sl2 (reversal)
    tp2_points: float = 20.0         # tp2 (reversal)
    ctc1_points: float = 10.0        # ctc1 for Leg 1
    ctc2_points: float = 10.0        # ctc2 for Leg 2
    decay_points: float = 15.0
    enable_flip: bool = True         # trigger reversal trade on SL1 hit
    base_units: float = 10.0
    tie_tolerance: float = 0.05      # points — treat as tie if distances within this
    session_start: str = "sydney"    # "sydney" or "tokyo"
    limit_cancel_ist: str = "17:00"  # HH:MM IST cutoff for unfilled limit
    eval_time_ist: str = "12:30"     # HH:MM IST evaluation point
    ctc_trigger_points: Optional[float] = None  # Backward compatibility fallback


@dataclass(frozen=True)
class RangeSweepV2Signal:
    date: dt.date
    eval_time_utc: dt.datetime
    limit_cancel_utc: dt.datetime
    eod_utc: dt.datetime
    direction: Literal["LONG", "SHORT"]
    eval_price: float
    range_high: float
    range_low: float
    dist_to_low: float
    dist_to_high: float
    entry_price: float
    sl_price: float
    tp_price: float
    units: float
    risk_sanitized: bool
    risk_notes: str
    is_instant: bool = False

    @property
    def midpoint(self) -> float:
        return (self.range_high + self.range_low) / 2.0

    @property
    def range_pts(self) -> float:
        return self.range_high - self.range_low


class RangeSweepV2Strategy:
    def __init__(
        self,
        config: Optional[RangeSweepV2Config] = None,
        session_engine: Optional[SessionEngine] = None,
        risk_service: Optional[RiskService] = None,
    ):
        self.config = config or RangeSweepV2Config()
        if session_engine is None:
            eval_time_str = getattr(self.config, "eval_time_ist", "12:30") or "12:30"
            parts = eval_time_str.strip().split(":")
            eval_h = int(parts[0]) if len(parts) > 0 and parts[0] else 12
            eval_m = int(parts[1]) if len(parts) > 1 and parts[1] else 30
            self.session_engine = SessionEngine(
                session_start=self.config.session_start,
                eval_hour_ist=eval_h,
                eval_minute_ist=eval_m,
            )
            self.session_engine.set_limit_cancel_time_ist(self.config.limit_cancel_ist)
        else:
            self.session_engine = session_engine
        self.risk_service = risk_service or RiskService()

    def evaluate_day_context(
        self,
        target_date: dt.date,
        day_df: pd.DataFrame,
        account_equity: float = 10000.0,
    ) -> tuple[Optional[RangeSweepV2Signal], dict]:
        """Evaluates day and returns both the signal (if any) and the full computed_context."""
        window: SessionWindow = self.session_engine.get_session_window(target_date)

        if pd.api.types.is_numeric_dtype(day_df["timestamp"]):
            dt_series = pd.to_datetime(day_df["timestamp"], unit="ms", utc=True)
        else:
            dt_series = pd.to_datetime(day_df["timestamp"], utc=True)

        # Prior completed bars strictly before 12:30 IST evaluation window
        prior_mask = (dt_series >= window.session_start_utc) & (dt_series < window.eval_time_utc)
        prior_bars = day_df.loc[prior_mask]
        eval_candle = day_df.loc[dt_series == window.eval_time_utc]

        if len(prior_bars) < 15 or len(eval_candle) == 0:
            return None, {"reason": "insufficient_data", "bars_count": len(prior_bars)}

        # Current price at 12:30 IST is the starting price (open) of the 12:30 candle
        eval_price = float(eval_candle.iloc[0]["open"])

        # Range at the instant 12:30:00 IST strictly uses prior completed bars + open price at 12:30
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
            "decay_points": self.config.decay_points,
            "ctc1_points": self.config.ctc1_points,
            "ctc2_points": self.config.ctc2_points,
            "enable_flip": self.config.enable_flip,
            "x_offset": self.config.x_offset,
            "sl_points": self.config.sl_points,
            "tp_points": self.config.tp_points,
            "sl2_points": self.config.sl2_points,
            "tp2_points": self.config.tp2_points,
        }

        if is_tie:
            context["reason"] = "tie"
            return None, context

        signal = self.evaluate_day(target_date, day_df, account_equity=account_equity)
        return signal, context

    def evaluate_day(
        self,
        target_date: dt.date,
        day_df: pd.DataFrame,
        account_equity: float = 10000.0,
    ) -> Optional[RangeSweepV2Signal]:
        window: SessionWindow = self.session_engine.get_session_window(target_date)

        if pd.api.types.is_numeric_dtype(day_df["timestamp"]):
            dt_series = pd.to_datetime(day_df["timestamp"], unit="ms", utc=True)
        else:
            dt_series = pd.to_datetime(day_df["timestamp"], utc=True)

        # Prior completed bars strictly before 12:30 IST evaluation window
        prior_mask = (dt_series >= window.session_start_utc) & (dt_series < window.eval_time_utc)
        prior_bars = day_df.loc[prior_mask]
        eval_candle = day_df.loc[dt_series == window.eval_time_utc]

        if len(prior_bars) < 15 or len(eval_candle) == 0:
            return None

        # Current price at 12:30 IST is the starting price (open) of the 12:30 candle
        eval_price = float(eval_candle.iloc[0]["open"])

        # Range at the instant 12:30:00 IST strictly uses prior completed bars + open price at 12:30
        range_high = max(float(prior_bars["high"].max()), eval_price)
        range_low = min(float(prior_bars["low"].min()), eval_price)

        dist_to_low = eval_price - range_low
        dist_to_high = range_high - eval_price

        if abs(dist_to_low - dist_to_high) <= self.config.tie_tolerance:
            return None

        if dist_to_low < dist_to_high:
            direction: Literal["LONG", "SHORT"] = "LONG"
            entry_price = min(eval_price, range_low + self.config.x_offset)
            sl_price = entry_price - self.config.sl_points
            tp_price = entry_price + self.config.tp_points
        else:
            direction = "SHORT"
            entry_price = max(eval_price, range_high - self.config.x_offset)
            sl_price = entry_price + self.config.sl_points
            tp_price = entry_price - self.config.tp_points

        is_instant = abs(entry_price - eval_price) < 1e-5

        approved, sanctioned_units, reason = self.risk_service.sanitize_order_intent(
            base_units=self.config.base_units,
            chain_depth=1,
            requested_multiplier=1.0,
            ml_confidence_scale=1.0,
            account_equity=account_equity,
            sl_points=self.config.sl_points,
        )

        if not approved:
            return None

        return RangeSweepV2Signal(
            date=target_date,
            eval_time_utc=window.eval_time_utc,
            limit_cancel_utc=window.limit_cancel_utc,
            eod_utc=window.eod_utc,
            direction=direction,
            eval_price=eval_price,
            range_high=range_high,
            range_low=range_low,
            dist_to_low=dist_to_low,
            dist_to_high=dist_to_high,
            entry_price=round(entry_price, 3),
            sl_price=round(sl_price, 3),
            tp_price=round(tp_price, 3),
            units=sanctioned_units,
            risk_sanitized=approved,
            risk_notes=reason,
            is_instant=is_instant,
        )
