import pandas as pd
import datetime as dt
from pathlib import Path
import sys

# Add parent to path if run from scripts
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.condition_generator import ConditionGenerator

def main():
    parquet_path = "/Users/prince/algo-trading/data/historical/XAUUSD_5m.parquet"
    cg = ConditionGenerator(db_path="hypotrader.db")
    
    print(f"Generating market conditions from {parquet_path}...")
    # Generate conditions for 2026 to capture the Jan-Jul and Aug-Sep windows
    # Note: If the parquet contains other years, we might want to bound it.
    start_dt = dt.date(2026, 1, 1)
    end_dt = dt.date(2026, 9, 7)
    
    # We use a broad slice and then filter
    df = cg.generate_conditions_from_parquet(parquet_path, start_date=start_dt, end_date=end_dt)
    print(f"Total trading days processed: {len(df)}")
    
    if len(df) == 0:
        print("No data processed!")
        return

    # Convert date strings to datetime for filtering
    df["date"] = pd.to_datetime(df["date"])
    
    # Define the two regimes
    regime_1 = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-07-31")]
    regime_2 = df[(df["date"] >= "2026-08-01") & (df["date"] <= "2026-09-07")]
    
    print(f"Regime 1 (Jan-Jul) count: {len(regime_1)}")
    print(f"Regime 2 (Aug-Sep) count: {len(regime_2)}")
    
    metrics = ["parkinson_volatility", "hurst_exponent", "adx", "vol_of_vol", "lag_1_autocorr", "lag_5_autocorr"]
    
    print("\n--- REGIME DIAGNOSTICS ---")
    for metric in metrics:
        m1 = regime_1[metric].dropna()
        m2 = regime_2[metric].dropna()
        
        print(f"\n{metric.upper()}:")
        if len(m1) > 0 and len(m2) > 0:
            print(f"  Jan-Jul 2026: Mean = {m1.mean():.4f}, Std = {m1.std():.4f}, Skew = {m1.skew():.4f}")
            print(f"  Aug-Sep 2026: Mean = {m2.mean():.4f}, Std = {m2.std():.4f}, Skew = {m2.skew():.4f}")
            print(f"  Difference (AugSep - JanJul): {(m2.mean() - m1.mean()):.4f}")
        else:
            print("  Not enough data to compute statistics.")

if __name__ == "__main__":
    main()
