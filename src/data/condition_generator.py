"""
Condition Query & Event Mining Engine.
Extracts structured market condition features (ranges, proximity, volatility) from multi-year
historical data and enables SQL-style condition filtering ("create data given some condition in market").
"""

from dataclasses import dataclass, asdict
import datetime as dt
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import pandas as pd
import numpy as np
import sqlite3
import warnings

from src.session.engine import SessionEngine, SessionWindow


@dataclass
class MarketConditionRecord:
    date: str
    sydney_open_utc: str
    eval_time_utc: str
    range_high: float
    range_low: float
    range_width_pts: float
    midpoint: float
    eval_price: float
    distance_to_high: float
    distance_to_low: float
    proximity_bias: str
    distance_to_extreme: float
    is_within_offset: bool
    is_event_day: bool
    
    # New regime features
    parkinson_volatility: Optional[float] = None
    hurst_exponent: Optional[float] = None
    adx: Optional[float] = None
    vol_of_vol: Optional[float] = None
    lag_1_autocorr: Optional[float] = None
    lag_2_autocorr: Optional[float] = None
    lag_3_autocorr: Optional[float] = None
    lag_4_autocorr: Optional[float] = None
    lag_5_autocorr: Optional[float] = None
    
    # New 11:00-12:30 IST regime specific properties
    regime_range_high: Optional[float] = None
    regime_range_low: Optional[float] = None
    regime_close: Optional[float] = None


class ConditionGenerator:
    def __init__(
        self,
        session_engine: Optional[SessionEngine] = None,
        db_path: Union[Path, str] = "hypotrader.db",
    ):
        self.session_engine = session_engine or SessionEngine()
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS market_conditions (
                date TEXT PRIMARY KEY,
                sydney_open_utc TEXT NOT NULL,
                eval_time_utc TEXT NOT NULL,
                range_high REAL NOT NULL,
                range_low REAL NOT NULL,
                range_width_pts REAL NOT NULL,
                midpoint REAL NOT NULL,
                eval_price REAL NOT NULL,
                distance_to_high REAL NOT NULL,
                distance_to_low REAL NOT NULL,
                proximity_bias TEXT NOT NULL,
                distance_to_extreme REAL NOT NULL,
                is_within_offset BOOLEAN NOT NULL
            );
            """)
            
            # Migration for new columns
            cursor = conn.execute("PRAGMA table_info(market_conditions)")
            cols = [col["name"] for col in cursor.fetchall()]
            new_cols = [
                "is_event_day", "regime_range_high", "regime_range_low", "regime_close",
                "parkinson_volatility", "hurst_exponent", "adx", 
                "vol_of_vol", "lag_1_autocorr", "lag_2_autocorr", 
                "lag_3_autocorr", "lag_4_autocorr", "lag_5_autocorr"
            ]
            for col in new_cols:
                if col not in cols:
                    if col == "is_event_day":
                        conn.execute(f"ALTER TABLE market_conditions ADD COLUMN {col} BOOLEAN")
                    else:
                        conn.execute(f"ALTER TABLE market_conditions ADD COLUMN {col} REAL")

    def generate_conditions_from_parquet(
        self,
        parquet_path: Union[Path, str],
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
        x_offset: float = 3.5,
    ) -> pd.DataFrame:
        """
        Parses multi-year 1m bars, calculates Sydney-to-IST session ranges,
        and builds the market conditions feature dataset.
        """
        df = pd.read_parquet(parquet_path)
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["utc_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["utc_time"] = pd.to_datetime(df["timestamp"], utc=True)

        min_dt = df["utc_time"].min().date()
        max_dt = df["utc_time"].max().date()

        start = start_date or min_dt
        end = end_date or max_dt

        # Hardcoded 2024-2026 FOMC and NFP logic for backtesting
        fomc_dates = {
            # 2024
            dt.date(2024, 1, 31), dt.date(2024, 3, 20), dt.date(2024, 5, 1),
            dt.date(2024, 6, 12), dt.date(2024, 7, 31), dt.date(2024, 9, 18),
            dt.date(2024, 11, 7), dt.date(2024, 12, 18),
            # 2025
            dt.date(2025, 1, 29), dt.date(2025, 3, 19), dt.date(2025, 5, 7),
            dt.date(2025, 6, 18), dt.date(2025, 7, 30), dt.date(2025, 9, 17),
            dt.date(2025, 10, 29), dt.date(2025, 12, 10),
            # 2026
            dt.date(2026, 1, 28), dt.date(2026, 3, 18), dt.date(2026, 4, 29),
            dt.date(2026, 6, 17), dt.date(2026, 7, 29), dt.date(2026, 9, 16),
            dt.date(2026, 11, 4), dt.date(2026, 12, 16)
        }
        
        def is_nfp(d: dt.date) -> bool:
            return d.weekday() == 4 and d.day <= 7

        # Find distinct trading dates (excluding weekends)
        dates_range = pd.date_range(start, end, freq="B").date

        records: List[MarketConditionRecord] = []

        for target_date in dates_range:
            is_event = (target_date in fomc_dates) or is_nfp(target_date)
            window: SessionWindow = self.session_engine.get_session_window(target_date)

            # Slice session bars
            mask = (df["utc_time"] >= window.sydney_open_utc) & (df["utc_time"] <= window.eval_time_utc)
            session_bars = df.loc[mask]

            if len(session_bars) < 30:
                continue

            range_high = float(session_bars["high"].max())
            range_low = float(session_bars["low"].min())
            range_width = round(range_high - range_low, 3)
            midpoint = round((range_high + range_low) / 2.0, 3)

            eval_price = round(float(session_bars.iloc[-1]["close"]), 3)
            dist_high = round(range_high - eval_price, 3)
            dist_low = round(eval_price - range_low, 3)

            if eval_price < midpoint:
                bias = "LONG"
                dist_extreme = dist_low
            else:
                bias = "SHORT"
                dist_extreme = dist_high

            is_within = dist_extreme <= x_offset
            
            # Slice 11:00-12:30 IST window (05:30-07:00 UTC) for regime calculations
            regime_start_utc = pd.Timestamp(dt.datetime.combine(target_date, dt.time(5, 30))).tz_localize("UTC")
            regime_end_utc = pd.Timestamp(dt.datetime.combine(target_date, dt.time(7, 0))).tz_localize("UTC")
            regime_mask = (df["utc_time"] >= regime_start_utc) & (df["utc_time"] <= regime_end_utc)
            regime_bars = df.loc[regime_mask]
            
            regime_high = None
            regime_low = None
            regime_close = None
            if not is_event and len(regime_bars) > 0:
                regime_high = float(regime_bars["high"].max())
                regime_low = float(regime_bars["low"].min())
                regime_close = float(regime_bars.iloc[-1]["close"])

            rec = MarketConditionRecord(
                date=target_date.isoformat(),
                sydney_open_utc=window.sydney_open_utc.isoformat(),
                eval_time_utc=window.eval_time_utc.isoformat(),
                range_high=range_high,
                range_low=range_low,
                range_width_pts=range_width,
                midpoint=midpoint,
                eval_price=eval_price,
                distance_to_high=dist_high,
                distance_to_low=dist_low,
                proximity_bias=bias,
                distance_to_extreme=dist_extreme,
                is_within_offset=is_within,
                is_event_day=is_event,
                regime_range_high=regime_high,
                regime_range_low=regime_low,
                regime_close=regime_close
            )
            records.append(rec)

        res_df = pd.DataFrame([asdict(r) for r in records])

        # Feature Engineering: compute rolling metrics (Phase 1)
        if not res_df.empty:
            # 1. Parkinson Volatility: sqrt( 1/(4 ln 2) * (ln(H/L))^2 )
            res_df["parkinson_volatility"] = np.sqrt(1.0 / (4.0 * np.log(2.0))) * np.log(pd.to_numeric(res_df["regime_range_high"]) / pd.to_numeric(res_df["regime_range_low"]))
            
            # 2. Vol-of-vol: 10-day rolling std of Parkinson volatility
            res_df["vol_of_vol"] = res_df["parkinson_volatility"].rolling(window=10, min_periods=5).std()
            
            # 3. Autocorrelation: Lag 1 to 5 of daily returns (using regime_close)
            daily_returns = pd.to_numeric(res_df["regime_close"]).pct_change()
            for lag in range(1, 6):
                # Use a rolling correlation to capture dynamic autocorrelation
                res_df[f"lag_{lag}_autocorr"] = daily_returns.rolling(window=10, min_periods=5).apply(
                    lambda x: pd.Series(x).autocorr(lag=lag) if len(x.dropna()) > lag else np.nan
                )
            
            # 4. Hurst Exponent (rolling 10-day window)
            try:
                from hurst import compute_Hc
                def calc_hurst(x):
                    x_clean = x.dropna()
                    if len(x_clean) < 10:
                        return np.nan
                    try:
                        H, c, data = compute_Hc(x_clean, kind='price', simplified=True)
                        return H
                    except Exception:
                        return 0.5 # Random walk assumption fallback
                
                res_df["hurst_exponent"] = pd.to_numeric(res_df["regime_close"]).rolling(window=10, min_periods=10).apply(calc_hurst)
            except ImportError:
                res_df["hurst_exponent"] = np.nan
                
            # 5. ADX (14-day)
            # Wilder's Smoothing
            def wilder_smooth(s, n=14):
                res = np.zeros_like(s.values)
                s_clean = s.dropna()
                res[0] = s_clean.values[0] if len(s_clean) > 0 else 0
                for i in range(1, len(s)):
                    if np.isnan(res[i-1]):
                        res[i] = s.values[i]
                    elif np.isnan(s.values[i]):
                        res[i] = res[i-1]
                    else:
                        res[i] = (res[i-1] * (n - 1) + s.values[i]) / n
                return pd.Series(res, index=s.index)

            # True Range based on regime window
            prev_close = pd.to_numeric(res_df["regime_close"]).shift(1)
            tr1 = pd.to_numeric(res_df["regime_range_high"]) - pd.to_numeric(res_df["regime_range_low"])
            tr2 = (pd.to_numeric(res_df["regime_range_high"]) - prev_close).abs()
            tr3 = (pd.to_numeric(res_df["regime_range_low"]) - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            up_move = pd.to_numeric(res_df["regime_range_high"]) - pd.to_numeric(res_df["regime_range_high"]).shift(1)
            down_move = pd.to_numeric(res_df["regime_range_low"]).shift(1) - pd.to_numeric(res_df["regime_range_low"])
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            atr = wilder_smooth(tr, 14)
            plus_di = 100 * (wilder_smooth(pd.Series(plus_dm, index=tr.index), 14) / atr)
            minus_di = 100 * (wilder_smooth(pd.Series(minus_dm, index=tr.index), 14) / atr)
            
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1))
            res_df["adx"] = wilder_smooth(dx, 14)

        # Update records with new computed features
        if not res_df.empty:
            res_df.replace({np.nan: None}, inplace=True)
            records = [MarketConditionRecord(**row) for row in res_df.to_dict("records")]

        # Persist to SQLite
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO market_conditions (
                    date, sydney_open_utc, eval_time_utc, range_high, range_low,
                    range_width_pts, midpoint, eval_price, distance_to_high,
                    distance_to_low, proximity_bias, distance_to_extreme, is_within_offset,
                    is_event_day, regime_range_high, regime_range_low, regime_close,
                    parkinson_volatility, hurst_exponent, adx, vol_of_vol,
                    lag_1_autocorr, lag_2_autocorr, lag_3_autocorr, lag_4_autocorr, lag_5_autocorr
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    r.date, r.sydney_open_utc, r.eval_time_utc, r.range_high, r.range_low,
                    r.range_width_pts, r.midpoint, r.eval_price, r.distance_to_high,
                    r.distance_to_low, r.proximity_bias, r.distance_to_extreme, r.is_within_offset,
                    r.is_event_day, r.regime_range_high, r.regime_range_low, r.regime_close,
                    r.parkinson_volatility, r.hurst_exponent, r.adx, r.vol_of_vol,
                    r.lag_1_autocorr, r.lag_2_autocorr, r.lag_3_autocorr, r.lag_4_autocorr, r.lag_5_autocorr
                ) for r in records
            ])

        return res_df

    def query_conditions(self, sql_where_clause: str) -> pd.DataFrame:
        """
        Executes a custom SQL query over the condition dataset to mine specific setups.
        Example: query_conditions("range_width_pts BETWEEN 15.0 AND 30.0 AND is_within_offset = 1")
        """
        query = f"SELECT * FROM market_conditions WHERE {sql_where_clause} ORDER BY date"
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn)
