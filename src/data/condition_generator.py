"""
Condition Query & Event Mining Engine.
Extracts structured market condition features (ranges, proximity, volatility) from multi-year
historical data and enables SQL-style condition filtering ("create data given some condition in market").
"""

from dataclasses import dataclass, asdict
import datetime as dt
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import sqlite3

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


class ConditionGenerator:
    def __init__(
        self,
        session_engine: Optional[SessionEngine] = None,
        db_path: Path | str = "hypotrader.db",
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

    def generate_conditions_from_parquet(
        self,
        parquet_path: Path | str,
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

        # Persist to SQLite
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO market_conditions (
                    date, sydney_open_utc, eval_time_utc, range_high, range_low,
                    range_width_pts, midpoint, eval_price, distance_to_high,
                    distance_to_low, proximity_bias, distance_to_extreme, is_within_offset
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    r.date, r.sydney_open_utc, r.eval_time_utc, r.range_high, r.range_low,
                    r.range_width_pts, r.midpoint, r.eval_price, r.distance_to_high,
                    r.distance_to_low, r.proximity_bias, r.distance_to_extreme, r.is_within_offset
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
