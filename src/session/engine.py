"""
Session Engine with full IANA timezone awareness (zoneinfo).
Computes exact session anchors (e.g. Sydney Open) and evaluation triggers (e.g. IST 12:30)
with exact handling of Daylight Saving Time (AEDT <-> AEST) transitions.
"""

from dataclasses import dataclass
import datetime as dt
from zoneinfo import ZoneInfo

TZ_SYDNEY = ZoneInfo("Australia/Sydney")
TZ_TOKYO = ZoneInfo("Asia/Tokyo")
TZ_IST = ZoneInfo("Asia/Kolkata")
TZ_UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class SessionWindow:
    """Represents an exact time interval in UTC for a trading day."""
    date: dt.date
    session_start_utc: dt.datetime
    eval_time_utc: dt.datetime
    limit_cancel_utc: dt.datetime  # e.g. 5:00 PM IST — unfilled limit cutoff
    eod_utc: dt.datetime           # End-of-day forced exit
    duration_hours: float
    is_sydney_dst: bool  # True if AEDT (UTC+11), False if AEST (UTC+10)
    session_name: str = "sydney"

    @property
    def sydney_open_utc(self) -> dt.datetime:
        """Backward-compatible alias for interval start time."""
        return self.session_start_utc


class SessionEngine:
    def __init__(
        self,
        session_start: str = "sydney",
        sydney_open_hour: int = 7,
        sydney_open_minute: int = 0,
        tokyo_open_hour: int = 9,
        tokyo_open_minute: int = 0,
        eval_hour_ist: int = 12,
        eval_minute_ist: int = 30,
        limit_cancel_hour_ist: int = 17,
        limit_cancel_minute_ist: int = 0,
        eod_hour_utc: int = 21,
        eod_minute_utc: int = 0,
    ):
        self.session_start = session_start.lower().strip()
        self.sydney_open_hour = sydney_open_hour
        self.sydney_open_minute = sydney_open_minute
        self.tokyo_open_hour = tokyo_open_hour
        self.tokyo_open_minute = tokyo_open_minute
        self.eval_hour_ist = eval_hour_ist
        self.eval_minute_ist = eval_minute_ist
        self.limit_cancel_hour_ist = limit_cancel_hour_ist
        self.limit_cancel_minute_ist = limit_cancel_minute_ist
        self.eod_hour_utc = eod_hour_utc
        self.eod_minute_utc = eod_minute_utc

    def set_limit_cancel_time_ist(self, time_str: str) -> None:
        """Sets limit cancel cutoff from HH:MM format string (e.g. '17:00')."""
        parts = time_str.strip().split(":")
        self.limit_cancel_hour_ist = int(parts[0])
        self.limit_cancel_minute_ist = int(parts[1]) if len(parts) > 1 else 0

    def get_session_window(self, target_date: dt.date) -> SessionWindow:
        """
        Computes the exact UTC session window for a given calendar date.
        
        Session definition:
          - Evaluation Trigger: target_date at 12:30 PM IST.
          - Session Start:
            * Sydney: 07:00 AM local Sydney time on target_date (AEST/AEDT aware).
            * Tokyo: 09:00 AM JST on target_date (00:00 UTC / 05:30 IST).
        """
        is_dst = False
        if self.session_start == "tokyo":
            tokyo_local = dt.datetime(
                target_date.year,
                target_date.month,
                target_date.day,
                self.tokyo_open_hour,
                self.tokyo_open_minute,
                tzinfo=TZ_TOKYO,
            )
            start_open_utc = tokyo_local.astimezone(TZ_UTC)
            session_name = "tokyo"
        else:
            # Localize Sydney Open on target_date
            syd_local = dt.datetime(
                target_date.year,
                target_date.month,
                target_date.day,
                self.sydney_open_hour,
                self.sydney_open_minute,
                tzinfo=TZ_SYDNEY,
            )
            start_open_utc = syd_local.astimezone(TZ_UTC)
            is_dst = syd_local.dst() != dt.timedelta(0)
            session_name = "sydney"

        # Localize Evaluation Trigger in IST on target_date
        ist_local = dt.datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            self.eval_hour_ist,
            self.eval_minute_ist,
            tzinfo=TZ_IST,
        )
        eval_utc = ist_local.astimezone(TZ_UTC)

        limit_cancel_local = dt.datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            self.limit_cancel_hour_ist,
            self.limit_cancel_minute_ist,
            tzinfo=TZ_IST,
        )
        limit_cancel_utc = limit_cancel_local.astimezone(TZ_UTC)

        eod_utc = dt.datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            self.eod_hour_utc,
            self.eod_minute_utc,
            tzinfo=TZ_UTC,
        )

        # Calculate exact duration
        duration = (eval_utc - start_open_utc).total_seconds() / 3600.0

        return SessionWindow(
            date=target_date,
            session_start_utc=start_open_utc,
            eval_time_utc=eval_utc,
            limit_cancel_utc=limit_cancel_utc,
            eod_utc=eod_utc,
            duration_hours=duration,
            is_sydney_dst=is_dst,
            session_name=session_name,
        )
