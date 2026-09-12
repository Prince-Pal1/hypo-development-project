"""
Execution simulator for Range Scope Strategy Version 1.

Simulates instant market entry at 11:00 AM IST, Stop Loss management,
and the two Take Profit modes:
  - Mode 1: Early TP at range boundary (<= 12:29 IST) + dynamic adjustment at 12:30 IST.
  - Mode 2: Wait until 12:30 IST (no early TP) + expanded 12:30 range TP calculation.
"""

from dataclasses import dataclass
import datetime as dt
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo
import pandas as pd

from src.execution.simulator import FeeModel
from src.strategies.range_scope_v1 import RangeScopeSignal
from src.execution.ambiguity_resolver import (
    is_ambiguous_bar, resolve_ambiguity, load_1min_dataframe, Confidence
)

TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_UTC = ZoneInfo("UTC")


@dataclass
class RangeScopeTradeResult:
    date: dt.date
    direction: str
    entry_time: dt.datetime
    entry_price: float
    units: float
    sl_price: float
    initial_tp: Optional[float]
    tp_mode: str
    tp_offset_y: float
    scope_pts: float
    range_high: float
    range_low: float
    ctc_armed: bool = False
    ctc_price: Optional[float] = None
    ctc_time: Optional[dt.datetime] = None
    exit_time: Optional[dt.datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None          # "TP", "TP_1230_LOCKIN", "SL", "CTC", "EOD"
    tp_at_1230: Optional[float] = None
    price_at_1230: Optional[float] = None
    pnl_points: float = 0.0
    pnl_gross: float = 0.0
    fee_usd: float = 0.0
    pnl_net: float = 0.0
    is_open: bool = True
    confidence: str = "RESOLVED"
    ambiguity_reasons: Optional[List[str]] = None
    events: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.events is None:
            self.events = []


class RangeScopeSimulator:
    def __init__(
        self,
        fee_model: Optional[FeeModel] = None,
        point_value: float = 1.0,
        df_1m: Optional[pd.DataFrame] = None,
    ):
        self.fee_model = fee_model or FeeModel()
        self.point_value = point_value
        self.df_1m = df_1m  # Optional 1-min data for ambiguity resolution

    def simulate_day(
        self,
        signal: RangeScopeSignal,
        day_df: pd.DataFrame,
        session_start_utc: dt.datetime,
    ) -> RangeScopeTradeResult:
        """Runs forward simulation for a single trading day from 11:00 AM IST."""
        if pd.api.types.is_numeric_dtype(day_df["timestamp"]):
            dt_series = pd.to_datetime(day_df["timestamp"], unit="ms", utc=True)
        else:
            dt_series = pd.to_datetime(day_df["timestamp"], utc=True)

        trade = RangeScopeTradeResult(
            date=signal.date,
            direction=signal.direction,
            entry_time=signal.eval_time_utc,
            entry_price=signal.entry_price,
            units=signal.units,
            sl_price=signal.sl_price,
            initial_tp=signal.initial_tp if signal.tp_mode == "mode_1_dynamic" else None,
            tp_mode=signal.tp_mode,
            tp_offset_y=signal.tp_offset_y,
            scope_pts=signal.scope_pts,
            range_high=signal.range_high,
            range_low=signal.range_low,
        )

        trade.events.append({
            "type": "ENTRY",
            "time": signal.eval_time_utc,
            "price": signal.entry_price,
            "direction": signal.direction,
            "note": f"Instant market fill at 11:00 AM IST | Scope: {signal.scope_pts:.1f} pts",
        })

        day_df = day_df.copy()
        if pd.api.types.is_numeric_dtype(day_df["timestamp"]):
            dt_series = pd.to_datetime(day_df["timestamp"], unit="ms", utc=True)
        else:
            dt_series = pd.to_datetime(day_df["timestamp"], utc=True)
        day_df["_dt"] = dt_series

        # Filter forward bars from 11:00 AM IST onwards
        forward_mask = day_df["_dt"] >= signal.eval_time_utc
        forward_bars = day_df.loc[forward_mask]

        if forward_bars.empty:
            trade.is_open = False
            trade.exit_price = signal.entry_price
            trade.exit_time = signal.eval_time_utc
            trade.exit_reason = "NO_DATA"
            return trade

        # 12:30 PM IST timestamp in UTC
        time_1230_utc = dt.datetime(signal.date.year, signal.date.month, signal.date.day, 12, 30, tzinfo=TZ_IST).astimezone(TZ_UTC)
        tp_updated_at_1230 = False
        current_tp = signal.initial_tp if signal.tp_mode == "mode_1_dynamic" else None
        ctc_threshold = getattr(signal, "ctc_points", 0.0) or 0.0

        for _, row in forward_bars.iterrows():
            bar_time = row["_dt"].to_pydatetime()
            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])

            # -------------------------------------------------------------
            # At 12:30 PM IST: Evaluate Dynamic Take Profit
            # -------------------------------------------------------------
            if not tp_updated_at_1230 and bar_time >= time_1230_utc:
                tp_updated_at_1230 = True
                price_1230 = bar_open
                trade.price_at_1230 = price_1230

                if signal.tp_mode == "mode_2_wait_1230":
                    # Mode 2: Recompute expanded range from session_start up to 12:30
                    mask_up_to_1230 = (day_df["_dt"] >= session_start_utc) & (day_df["_dt"] < bar_time)
                    bars_to_1230 = day_df.loc[mask_up_to_1230]
                    low_1230 = min(float(bars_to_1230["low"].min()), bar_open) if len(bars_to_1230) > 0 else bar_open
                    high_1230 = max(float(bars_to_1230["high"].max()), bar_open) if len(bars_to_1230) > 0 else bar_open

                    if signal.direction == "SHORT":
                        new_tp = min(low_1230 + signal.tp_offset_y, price_1230)
                        trade.tp_at_1230 = new_tp
                        if price_1230 <= low_1230 + signal.tp_offset_y:
                            return self._close_trade(trade, bar_time, price_1230, "TP_1230_LOCKIN", "12:30 immediate profit lock-in exit (Mode 2)")
                        current_tp = new_tp
                    else:  # LONG
                        new_tp = max(high_1230 - signal.tp_offset_y, price_1230)
                        trade.tp_at_1230 = new_tp
                        if price_1230 >= high_1230 - signal.tp_offset_y:
                            return self._close_trade(trade, bar_time, price_1230, "TP_1230_LOCKIN", "12:30 immediate profit lock-in exit (Mode 2)")
                        current_tp = new_tp

                else:
                    # Mode 1: Range low / high from session range
                    if signal.direction == "SHORT":
                        new_tp = min(signal.range_low + signal.tp_offset_y, price_1230)
                        trade.tp_at_1230 = new_tp
                        if price_1230 <= signal.range_low + signal.tp_offset_y:
                            return self._close_trade(trade, bar_time, price_1230, "TP_1230_LOCKIN", "12:30 immediate profit lock-in exit")
                        current_tp = new_tp
                    else:  # LONG
                        new_tp = max(signal.range_high - signal.tp_offset_y, price_1230)
                        trade.tp_at_1230 = new_tp
                        if price_1230 >= signal.range_high - signal.tp_offset_y:
                            return self._close_trade(trade, bar_time, price_1230, "TP_1230_LOCKIN", "12:30 immediate profit lock-in exit")
                        current_tp = new_tp

                tp_str = f"{current_tp:.2f}" if current_tp is not None else "None"
                trade.events.append({
                    "type": "TP_ADJUSTED_1230",
                    "time": bar_time,
                    "price": price_1230,
                    "new_tp": current_tp,
                    "note": f"TP dynamically adjusted at 12:30 IST to {tp_str}",
                })

            # -------------------------------------------------------------
            # Intrabar Execution Checks (SL, CTC, and active TP)
            # Uses 3-tier ambiguity resolution for conflicting events.
            #
            # IMPORTANT: CTC arming is NOT done before this check.
            # The ambiguity resolver needs to see the pre-CTC state to detect
            # CTC+SL conflicts. CTC is armed INSIDE the resolver (ambiguous path)
            # or AFTER SL/TP checks (non-ambiguous fast path).
            # -------------------------------------------------------------

            # Check for ambiguity BEFORE arming CTC
            ambiguous, conflict_reasons = is_ambiguous_bar(
                direction=signal.direction,
                bar_high=bar_high,
                bar_low=bar_low,
                entry_price=trade.entry_price,
                sl_price=trade.sl_price,
                tp_price=current_tp,
                ctc_threshold=ctc_threshold,
                ctc_armed=trade.ctc_armed,
            )

            if ambiguous:
                # Resolve using the 3-tier cascade
                resolution = resolve_ambiguity(
                    direction=signal.direction,
                    bar_open=bar_open,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    bar_close=bar_close,
                    bar_time=bar_time,
                    entry_price=trade.entry_price,
                    sl_price=trade.sl_price,
                    tp_price=current_tp,
                    ctc_threshold=ctc_threshold,
                    ctc_armed=trade.ctc_armed,
                    conflict_reasons=conflict_reasons,
                    df_1m=self.df_1m,
                )

                trade.confidence = resolution.tier.value
                trade.ambiguity_reasons = conflict_reasons

                if resolution.event_type == "TP_HIT":
                    return self._close_trade(trade, bar_time, resolution.price, "TP",
                        f"Take profit hit (resolved: {resolution.tier.value})")
                elif resolution.event_type == "SL_HIT":
                    return self._close_trade(trade, bar_time, resolution.price, "SL",
                        f"Stop loss hit (resolved: {resolution.tier.value})")
                elif resolution.event_type == "CTC_SL_HIT":
                    if not trade.ctc_armed:
                        trade.ctc_armed = True
                        trade.ctc_time = bar_time
                        if signal.direction == "LONG":
                            trade.ctc_price = round(trade.entry_price + ctc_threshold, 3)
                        else:
                            trade.ctc_price = round(trade.entry_price - ctc_threshold, 3)
                        trade.sl_price = trade.entry_price
                        trade.events.append({
                            "type": "CTC_ARMED",
                            "time": bar_time,
                            "price": trade.ctc_price,
                            "note": f"+{ctc_threshold:.1f} pts reached -> SL stepped to Breakeven (same bar as hit)",
                        })
                    return self._close_trade(trade, bar_time, trade.entry_price, "CTC",
                        f"CTC-SL hit at breakeven (resolved: {resolution.tier.value})")
                elif resolution.event_type == "CTC_ARM":
                    # CTC armed but trade survives — update state and continue
                    if not trade.ctc_armed:
                        trade.ctc_armed = True
                        trade.ctc_time = bar_time
                        if signal.direction == "LONG":
                            trade.ctc_price = round(trade.entry_price + ctc_threshold, 3)
                        else:
                            trade.ctc_price = round(trade.entry_price - ctc_threshold, 3)
                        trade.sl_price = trade.entry_price
                        trade.events.append({
                            "type": "CTC_ARMED",
                            "time": bar_time,
                            "price": trade.ctc_price,
                            "note": f"+{ctc_threshold:.1f} pts reached -> SL stepped to Breakeven (resolved: {resolution.tier.value})",
                        })
                # SURVIVE: trade continues to next bar
            else:
                # No ambiguity — fast path
                # 1. CTC arming (safe because no SL/TP conflict exists)
                if ctc_threshold > 0 and not trade.ctc_armed:
                    if signal.direction == "LONG" and bar_high >= trade.entry_price + ctc_threshold:
                        trade.ctc_armed = True
                        trade.ctc_time = bar_time
                        trade.ctc_price = round(trade.entry_price + ctc_threshold, 3)
                        trade.sl_price = trade.entry_price
                        trade.events.append({
                            "type": "CTC_ARMED",
                            "time": bar_time,
                            "price": trade.ctc_price,
                            "note": f"+{ctc_threshold:.1f} pts reached -> SL stepped to Breakeven",
                        })
                    elif signal.direction == "SHORT" and bar_low <= trade.entry_price - ctc_threshold:
                        trade.ctc_armed = True
                        trade.ctc_time = bar_time
                        trade.ctc_price = round(trade.entry_price - ctc_threshold, 3)
                        trade.sl_price = trade.entry_price
                        trade.events.append({
                            "type": "CTC_ARMED",
                            "time": bar_time,
                            "price": trade.ctc_price,
                            "note": f"+{ctc_threshold:.1f} pts reached -> SL stepped to Breakeven",
                        })

                # 2. SL check
                if signal.direction == "SHORT":
                    if bar_high >= trade.sl_price:
                        if trade.ctc_armed and abs(trade.sl_price - trade.entry_price) < 1e-4:
                            return self._close_trade(trade, bar_time, trade.entry_price, "CTC", "CTC-SL hit at breakeven (0 pts loss)")
                        return self._close_trade(trade, bar_time, trade.sl_price, "SL", "Stop loss hit")
                    if current_tp is not None and bar_low <= current_tp:
                        return self._close_trade(trade, bar_time, current_tp, "TP", "Take profit hit")

                elif signal.direction == "LONG":
                    if bar_low <= trade.sl_price:
                        if trade.ctc_armed and abs(trade.sl_price - trade.entry_price) < 1e-4:
                            return self._close_trade(trade, bar_time, trade.entry_price, "CTC", "CTC-SL hit at breakeven (0 pts loss)")
                        return self._close_trade(trade, bar_time, trade.sl_price, "SL", "Stop loss hit")
                    if current_tp is not None and bar_high >= current_tp:
                        return self._close_trade(trade, bar_time, current_tp, "TP", "Take profit hit")

            # EOD check
            if bar_time >= signal.eod_utc:
                return self._close_trade(trade, bar_time, bar_close, "EOD", "End-of-day forced close")

        # Fallback to last bar close if loop finishes
        last_row = forward_bars.iloc[-1]
        last_time = last_row["_dt"].to_pydatetime()
        return self._close_trade(trade, last_time, float(last_row["close"]), "EOD", "End of available bars")

    def _close_trade(
        self,
        trade: RangeScopeTradeResult,
        exit_time: dt.datetime,
        exit_price: float,
        reason: str,
        note: str,
    ) -> RangeScopeTradeResult:
        trade.is_open = False
        trade.exit_time = exit_time
        trade.exit_price = round(exit_price, 3)
        trade.exit_reason = reason

        if trade.direction == "SHORT":
            trade.pnl_points = trade.entry_price - trade.exit_price
        else:
            trade.pnl_points = trade.exit_price - trade.entry_price

        trade.pnl_gross = trade.pnl_points * trade.units * self.point_value
        commission = self.fee_model.commission_per_unit * trade.units
        spread_cost = self.fee_model.spread_pts * trade.units * self.point_value
        slippage_cost = self.fee_model.base_slippage_pts * trade.units * self.point_value
        trade.fee_usd = round(commission + spread_cost + slippage_cost, 2)
        trade.pnl_net = round(trade.pnl_gross - trade.fee_usd, 2)

        trade.events.append({
            "type": "EXIT",
            "time": exit_time,
            "price": trade.exit_price,
            "reason": reason,
            "pnl_points": round(trade.pnl_points, 2),
            "pnl_net": round(trade.pnl_net, 2),
            "note": note,
        })
        return trade
