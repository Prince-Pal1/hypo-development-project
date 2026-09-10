"""
Unit tests for DST transitions in the Session Engine.
Tests the exact shift across AEDT (UTC+11) <-> AEST (UTC+10) transition dates.
"""

import datetime as dt
import pytest
from src.session.engine import SessionEngine


def test_sydney_dst_transitions_2025():
    """
    In 2025:
      - DST ends (AEDT -> AEST): Sunday, April 6, 2025.
        * Friday April 4, 2025: AEDT (UTC+11). Sydney 07:00 is 20:00 UTC (April 3). Duration to 12:30 IST (07:00 UTC) = 11.0h.
        * Monday April 7, 2025: AEST (UTC+10). Sydney 07:00 is 21:00 UTC (April 6). Duration to 12:30 IST (07:00 UTC) = 10.0h.
      - DST starts (AEST -> AEDT): Sunday, October 5, 2025.
        * Friday October 3, 2025: AEST (UTC+10). Sydney 07:00 is 21:00 UTC (October 2). Duration to 12:30 IST = 10.0h.
        * Monday October 6, 2025: AEDT (UTC+11). Sydney 07:00 is 20:00 UTC (October 5). Duration to 12:30 IST = 11.0h.
    """
    engine = SessionEngine()

    # --- APRIL 2025 TRANSITION (AEDT -> AEST) ---
    pre_transition = engine.get_session_window(dt.date(2025, 4, 4))
    post_transition = engine.get_session_window(dt.date(2025, 4, 7))

    assert pre_transition.is_sydney_dst is True
    assert pre_transition.sydney_open_utc == dt.datetime(2025, 4, 3, 20, 0, tzinfo=dt.timezone.utc)
    assert pre_transition.eval_time_utc == dt.datetime(2025, 4, 4, 7, 0, tzinfo=dt.timezone.utc)
    assert pre_transition.duration_hours == 11.0

    assert post_transition.is_sydney_dst is False
    assert post_transition.sydney_open_utc == dt.datetime(2025, 4, 6, 21, 0, tzinfo=dt.timezone.utc)
    assert post_transition.eval_time_utc == dt.datetime(2025, 4, 7, 7, 0, tzinfo=dt.timezone.utc)
    assert post_transition.duration_hours == 10.0

    # --- OCTOBER 2025 TRANSITION (AEST -> AEDT) ---
    pre_october = engine.get_session_window(dt.date(2025, 10, 3))
    post_october = engine.get_session_window(dt.date(2025, 10, 6))

    assert pre_october.is_sydney_dst is False
    assert pre_october.sydney_open_utc == dt.datetime(2025, 10, 2, 21, 0, tzinfo=dt.timezone.utc)
    assert pre_october.eval_time_utc == dt.datetime(2025, 10, 3, 7, 0, tzinfo=dt.timezone.utc)
    assert pre_october.duration_hours == 10.0

    assert post_october.is_sydney_dst is True
    assert post_october.sydney_open_utc == dt.datetime(2025, 10, 5, 20, 0, tzinfo=dt.timezone.utc)
    assert post_october.eval_time_utc == dt.datetime(2025, 10, 6, 7, 0, tzinfo=dt.timezone.utc)
    assert post_october.duration_hours == 11.0


def test_sydney_dst_transitions_2024():
    """Verify 2024 transition dates (April 7, 2024 and October 6, 2024)."""
    engine = SessionEngine()

    # April 2024: Sunday April 7 was the transition
    fri_apr5 = engine.get_session_window(dt.date(2024, 4, 5))
    mon_apr8 = engine.get_session_window(dt.date(2024, 4, 8))

    assert fri_apr5.is_sydney_dst is True
    assert fri_apr5.duration_hours == 11.0
    assert mon_apr8.is_sydney_dst is False
    assert mon_apr8.duration_hours == 10.0
