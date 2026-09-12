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
                "parkinson_volatility", "hurst_exponent", "adx", 
                "vol_of_vol", "lag_1_autocorr", "lag_2_autocorr", 
                "lag_3_autocorr", "lag_4_autocorr", "lag_5_autocorr"
            ]
            for col in new_cols:
                if col not in cols:
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

        # Find distinct trading dates (excluding weekends)
        dates_range = pd.date_range(start, end, freq="B").date

        records: List[MarketConditionRecord] = []

        for target_date in dates_range:
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
            )
            records.append(rec)

        res_df = pd.DataFrame([asdict(r) for r in records])

        # Feature Engineering: compute rolling metrics (Phase 1)
        if not res_df.empty:
            # 1. Parkinson Volatility: sqrt( 1/(4 ln 2) * (ln(H/L))^2 )
            res_df["parkinson_volatility"] = np.sqrt(1.0 / (4.0 * np.log(2.0))) * np.log(res_df["range_high"] / res_df["range_low"])
            
            # 2. Vol-of-vol: 10-day rolling std of Parkinson volatility
            res_df["vol_of_vol"] = res_df["parkinson_volatility"].rolling(window=10, min_periods=5).std()
            
            # 3. Autocorrelation: Lag 1 to 5 of daily returns (using eval_price)
            daily_returns = res_df["eval_price"].pct_change()
            for lag in range(1, 6):
                # Use a rolling correlation to capture dynamic autocorrelation
                res_df[f"lag_{lag}_autocorr"] = daily_returns.rolling(window=10, min_periods=5).apply(
                    lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else np.nan
                )
            
            # 4. Hurst Exponent (rolling 10-day window)
            try:
                from hurst import compute_Hc
                def calc_hurst(x):
                    # hurst requires > 10 points typically, but we'll try on short windows, fallback to 0.5
                    # 'simplified' works for shorter series
                    try:
                        H, c, data = compute_Hc(x, kind='price', simplified=True)
                        return H
                    except Exception:
                        return 0.5 # Random walk assumption fallback
                
                res_df["hurst_exponent"] = res_df["eval_price"].rolling(window=10, min_periods=10).apply(calc_hurst)
            except ImportError:
                res_df["hurst_exponent"] = np.nan
                
            # 5. ADX (14-day)
            # Wilder's Smoothing
            def wilder_smooth(s, n=14):
                res = np.zeros_like(s.values)
                res[0] = s.dropna().values[0] if len(s.dropna()) > 0 else 0
                for i in range(1, len(s)):
                    if np.isnan(res[i-1]):
                        res[i] = s.values[i]
                    else:
                        res[i] = (res[i-1] * (n - 1) + s.values[i]) / n
                return pd.Series(res, index=s.index)

            # True Range
            prev_close = res_df["eval_price"].shift(1)
            tr1 = res_df["range_high"] - res_df["range_low"]
            tr2 = (res_df["range_high"] - prev_close).abs()
            tr3 = (res_df["range_low"] - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            up_move = res_df["range_high"] - res_df["range_high"].shift(1)
            down_move = res_df["range_low"].shift(1) - res_df["range_low"]
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            atr = wilder_smooth(tr, 14)
            plus_di = 100 * (wilder_smooth(pd.Series(plus_dm), 14) / atr)
            minus_di = 100 * (wilder_smooth(pd.Series(minus_dm), 14) / atr)
            
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
                    parkinson_volatility, hurst_exponent, adx, vol_of_vol,
                    lag_1_autocorr, lag_2_autocorr, lag_3_autocorr, lag_4_autocorr, lag_5_autocorr
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    r.date, r.sydney_open_utc, r.eval_time_utc, r.range_high, r.range_low,
                    r.range_width_pts, r.midpoint, r.eval_price, r.distance_to_high,
                    r.distance_to_low, r.proximity_bias, r.distance_to_extreme, r.is_within_offset,
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
