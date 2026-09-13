"""
Trial Logger for Optuna Optimization Studies.
Logs EVERY evaluated trial (completed and pruned) into the optimizer_trials table
to ensure rigorous Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR) calculations.
"""

import sqlite3
import json
import datetime as dt
from pathlib import Path
from typing import Union, Optional, Dict, Any


class TrialLogger:
    def __init__(self, db_path: Union[Path, str] = "hypotrader.db"):
        self.db_path = str(db_path)
        self._init_table()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS optimizer_trials (
                trial_id INTEGER,
                study_name TEXT NOT NULL,
                x_offset REAL,
                sl_points REAL,
                tp_points REAL,
                eval_time_ist TEXT,
                cost_sl_ratio REAL,
                is_pruned BOOLEAN NOT NULL,
                prune_reason TEXT,
                sharpe_ratio REAL,
                annualized_return REAL,
                max_drawdown_pct REAL,
                trades_count INTEGER,
                sub_interval_id TEXT,
                sub_interval_start TEXT,
                sub_interval_end TEXT,
                params_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (study_name, trial_id)
            );
            """)
            
            # Migration for generalized engine
            cursor = conn.execute("PRAGMA table_info(optimizer_trials)")
            cols = [col["name"] for col in cursor.fetchall()]
            new_cols = [
                ("sub_interval_id", "TEXT"),
                ("sub_interval_start", "TEXT"),
                ("sub_interval_end", "TEXT"),
                ("params_json", "TEXT")
            ]
            for col_name, col_type in new_cols:
                if col_name not in cols:
                    conn.execute(f"ALTER TABLE optimizer_trials ADD COLUMN {col_name} {col_type}")

    def log_trial(
        self,
        trial_id: int,
        study_name: str,
        is_pruned: bool,
        prune_reason: Optional[str] = None,
        sharpe_ratio: Optional[float] = None,
        annualized_return: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        trades_count: Optional[int] = None,
        # Legacy/Optional fields
        x_offset: Optional[float] = None,
        sl_points: Optional[float] = None,
        tp_points: Optional[float] = None,
        eval_time_ist: Optional[str] = None,
        cost_sl_ratio: Optional[float] = None,
        # New generalized fields
        sub_interval_id: Optional[str] = None,
        sub_interval_start: Optional[str] = None,
        sub_interval_end: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        params_json = json.dumps(params) if params else None
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO optimizer_trials (
                    trial_id, study_name, x_offset, sl_points, tp_points, eval_time_ist,
                    cost_sl_ratio, is_pruned, prune_reason, sharpe_ratio,
                    annualized_return, max_drawdown_pct, trades_count,
                    sub_interval_id, sub_interval_start, sub_interval_end, params_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trial_id, study_name, x_offset, sl_points, tp_points, eval_time_ist,
                cost_sl_ratio, is_pruned, prune_reason, sharpe_ratio,
                annualized_return, max_drawdown_pct, trades_count,
                sub_interval_id, sub_interval_start, sub_interval_end, params_json
            ))

    def get_all_sharpe_ratios(self, study_name: str) -> list[float]:
        """Returns the distribution of all completed (non-pruned) trial Sharpe ratios."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT sharpe_ratio FROM optimizer_trials
                WHERE study_name = ? AND is_pruned = 0 AND sharpe_ratio IS NOT NULL
            """, (study_name,)).fetchall()
            return [float(r["sharpe_ratio"]) for r in rows]

    def get_total_trials_count(self, parent_study_prefix: str) -> int:
        """Returns the total number of evaluated trials N across ALL sub-intervals for a parent study."""
        with self._get_connection() as conn:
            # We match any study_name that starts with the parent_study_prefix
            row = conn.execute("""
                SELECT COUNT(*) as cnt FROM optimizer_trials WHERE study_name LIKE ?
            """, (f"{parent_study_prefix}%",)).fetchone()
            return int(row["cnt"]) if row else 0
