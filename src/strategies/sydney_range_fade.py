"""
Single Un-Chained Leg Strategy: Sydney Range Fade at 12:30 PM IST.
Phase 1 core deliverable. Evaluates range proximity and generates entry/SL/TP levels.
"""

from dataclasses import dataclass
import datetime as dt
from typing import Optional, Literal
import pandas as pd

from src.session.engine import SessionEngine, SessionWindow
from src.core.risk_service import RiskService


@dataclass(frozen=True)
class SydneyRangeFadeConfig:
    x_offset: float = 3.5          # points buffer near range extremes
    sl_points: float = 10.0        # stop loss distance in points
    tp_points: float = 18.0        # take profit distance in points
    base_units: float = 10.0       # e.g. 10 troy oz


@dataclass(frozen=True)
class OrderSignal:
    date: dt.date
    eval_time_utc: dt.datetime
    direction: Literal["LONG", "SHORT"]
    eval_price: float
    range_high: float
    range_low: float
    midpoint: float
    entry_price: float
    sl_price: float
    tp_price: float
    units: float
    risk_sanitized: bool
    risk_notes: str


class SydneyRangeFadeStrategy:
    def __init__(
        self,
        config: SydneyRangeFadeConfig | None = None,
        session_engine: SessionEngine | None = None,
        risk_service: RiskService | None = None,
    ):
        self.config = config or SydneyRangeFadeConfig()
        self.session_engine = session_engine or SessionEngine()
        self.risk_service = risk_service or RiskService()

    def evaluate_day(
        self,
        target_date: dt.date,
        day_df: pd.DataFrame,
        account_equity: float = 10000.0,
    ) -> Optional[OrderSignal]:
        """
        Evaluates a single day's 1-minute bars up to 12:30 PM IST.
        
        Args:
            target_date: calendar date to evaluate
            day_df: DataFrame containing at least ['timestamp', 'high', 'low', 'close']
                    where timestamp is UTC (either int ms or tz-aware datetime)
            account_equity: current simulated account equity for risk checks
            
        Returns:
            OrderSignal if evaluation is valid, None if incomplete data.
        """
        window: SessionWindow = self.session_engine.get_session_window(target_date)

        # Normalize timestamp to UTC datetime if in ms
        if pd.api.types.is_numeric_dtype(day_df["timestamp"]):
            dt_series = pd.to_datetime(day_df["timestamp"], unit="ms", utc=True)
        else:
            dt_series = pd.to_datetime(day_df["timestamp"], utc=True)

        # Filter bars strictly in [sydney_open_utc, eval_time_utc)
        prior_mask = (dt_series >= window.sydney_open_utc) & (dt_series < window.eval_time_utc)
        prior_bars = day_df.loc[prior_mask]
        eval_candle = day_df.loc[dt_series == window.eval_time_utc]

        if len(prior_bars) < 20 or len(eval_candle) == 0:
            # Need sufficient bars to establish a genuine session range
            return None

        # Current price at 12:30 PM IST (open of eval bar)
        eval_price = float(eval_candle.iloc[0]["open"])

        # Calculate Sydney-to-IST range (strictly prior bars + 12:30 open)
        range_high = max(float(prior_bars["high"].max()), eval_price)
        range_low = min(float(prior_bars["low"].min()), eval_price)
        midpoint = (range_high + range_low) / 2.0

        # Determine bias: Nearer to lower limit vs higher limit
        if eval_price < midpoint:
            # Nearer to lower limit -> Fade LONG
            direction: Literal["LONG", "SHORT"] = "LONG"
            entry_price = min(eval_price, range_low + self.config.x_offset)
            sl_price = entry_price - self.config.sl_points
            tp_price = entry_price + self.config.tp_points
        else:
            # Nearer to higher limit -> Fade SHORT
            direction = "SHORT"
            entry_price = max(eval_price, range_high - self.config.x_offset)
            sl_price = entry_price + self.config.sl_points
            tp_price = entry_price - self.config.tp_points

        # Sanitize order through isolated RiskService
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

        return OrderSignal(
            date=target_date,
            eval_time_utc=window.eval_time_utc,
            direction=direction,
            eval_price=eval_price,
            range_high=range_high,
            range_low=range_low,
            midpoint=midpoint,
            entry_price=round(entry_price, 3),
            sl_price=round(sl_price, 3),
            tp_price=round(tp_price, 3),
            units=sanctioned_units,
            risk_sanitized=approved,
            risk_notes=reason,
        )
