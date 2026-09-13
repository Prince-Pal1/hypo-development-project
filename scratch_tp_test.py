import sys
from pathlib import Path
import pandas as pd
import datetime as dt
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.strategies.range_scope_v1 import RangeScopeConfig, RangeScopeStrategy, RangeScopeSignal
from src.core.risk_service import RiskService, RiskServiceLimits
from src.execution.range_scope_simulator import RangeScopeSimulator
from src.execution.simulator import FeeModel

def test_tp_modes():
    print("Testing tp_mode execution differences...")
    
    # Create fake data spanning 11:00 IST to 13:00 IST
    # 11:00 IST is 05:30 UTC
    tz_utc = ZoneInfo("UTC")
    base_time = dt.datetime(2026, 9, 1, 5, 30, tzinfo=tz_utc)
    
    # Create a downward trend to trigger a LONG trade at support, or something simple
    # Actually, we can just create a manual signal and feed fake forward data!
    
    signal_1 = RangeScopeSignal(
        date=dt.date(2026, 9, 1),
        direction="LONG",
        eval_time_utc=base_time,
        entry_price=100.0,
        sl_price=90.0,
        initial_tp=103.5, # 105.0 high - 1.5 offset
        tp_mode="mode_1_dynamic",
        tp_offset_y=1.5,
        scope_pts=5.0,
        range_high=105.0,
        range_low=95.0,
        units=1.0,
        ctc_points=15.0
    )
    
    signal_2 = RangeScopeSignal(
        date=dt.date(2026, 9, 1),
        direction="LONG",
        eval_time_utc=base_time,
        entry_price=100.0,
        sl_price=90.0,
        initial_tp=None, # Mode 2 doesn't have initial TP
        tp_mode="mode_2_wait_1230",
        tp_offset_y=1.5,
        scope_pts=5.0,
        range_high=105.0,
        range_low=95.0,
        units=1.0,
        ctc_points=15.0
    )

    # 12:30 IST is 07:00 UTC
    times = [base_time + dt.timedelta(minutes=m) for m in range(0, 150, 5)]
    
    # Let's make the price hover at 102.0 until 12:30.
    # From 11:00 to 12:30, price makes a new high of 106.0!
    # Then after 12:30 it hits 104.5.
    # Mode 1 dynamic will set TP at max(range_high(105) - 1.5, price) = 103.5
    # Mode 2 will expand range to 106.0, and set TP at 106.0 - 1.5 = 104.5
    
    data = []
    for t in times:
        if t < dt.datetime(2026, 9, 1, 7, 0, tzinfo=tz_utc):
            # Before 12:30
            high = 106.0 if t.minute == 0 else 102.0
            data.append({"timestamp": t, "open": 102.0, "high": high, "low": 100.0, "close": 102.0})
        else:
            # At/after 12:30
            data.append({"timestamp": t, "open": 104.5, "high": 104.5, "low": 104.5, "close": 104.5})

    df = pd.DataFrame(data)
    
    simulator = RangeScopeSimulator(fee_model=FeeModel(), point_value=1.0)
    
    res_1 = simulator.simulate_day(signal_1, df, base_time)
    res_2 = simulator.simulate_day(signal_2, df, base_time)
    
    print(f"Mode 1 Exit: {res_1.exit_reason} at {res_1.exit_price} (TP was {res_1.tp_at_1230})")
    print(f"Mode 2 Exit: {res_2.exit_reason} at {res_2.exit_price} (TP was {res_2.tp_at_1230})")
    
    if res_1.exit_price != res_2.exit_price:
        print("SUCCESS: Simulator outputs differently based on tp_mode.")
    else:
        print("FAIL: Simulator produced identical results.")

if __name__ == "__main__":
    test_tp_modes()
