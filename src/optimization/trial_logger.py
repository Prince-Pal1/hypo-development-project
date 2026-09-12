"""
Trial Logger for Optuna Optimization Studies.
Logs EVERY evaluated trial (completed and pruned) into the optimizer_trials table
to ensure rigorous Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR) calculations.
"""

import sqlite3
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
                x_offset REAL NOT NULL,
                sl_points REAL NOT NULL,
                tp_points REAL NOT NULL,
                eval_time_ist TEXT NOT NULL,
                cost_sl_ratio REAL NOT NULL,
                is_pruned BOOLEAN NOT NULL,
                prune_reason TEXT,
                sharpe_ratio REAL,
                annualized_return REAL,
                max_drawdown_pct REAL,
                trades_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (study_name, trial_id)
            );
            """)

    def log_trial(
        self,
        trial_id: int,
        study_name: str,
        x_offset: float,
        sl_points: float,
        tp_points: float,
        eval_time_ist: str,
        cost_sl_ratio: float,
        is_pruned: bool,
        prune_reason: Optional[str] = None,
        sharpe_ratio: Optional[float] = None,
        annualized_return: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        trades_count: Optional[int] = None,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO optimizer_trials (
                    trial_id, study_name, x_offset, sl_points, tp_points, eval_time_ist,
                    cost_sl_ratio, is_pruned, prune_reason, sharpe_ratio,
                    annualized_return, max_drawdown_pct, trades_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trial_id,
                study_name,
                x_offset,
                sl_points,
                tp_points,
                eval_time_ist,
                cost_sl_ratio,
                is_pruned,
                prune_reason,
                sharpe_ratio,
                annualized_return,
                max_drawdown_pct,
                trades_count,
            ))

    def get_all_sharpe_ratios(self, study_name: str) -> list[float]:
        """Returns the distribution of all completed (non-pruned) trial Sharpe ratios."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT sharpe_ratio FROM optimizer_trials
                WHERE study_name = ? AND is_pruned = 0 AND sharpe_ratio IS NOT NULL
            """, (study_name,)).fetchall()
            return [float(r["sharpe_ratio"]) for r in rows]

    def get_total_trials_count(self, study_name: str) -> int:
        """Returns the total number of evaluated trials N (including pruned)."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as cnt FROM optimizer_trials WHERE study_name = ?
            """, (study_name,)).fetchone()
            return int(row["cnt"]) if row else 0
