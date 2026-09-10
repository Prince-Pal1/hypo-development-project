"""
SQLite Repository for Hypothesis Lifecycle, Manual Journals, and Chained Trade Logs.
Implements the relational schema: hypotheses -> strategy_templates -> backtest_runs -> trade_logs
with trade_logs.parent_trade_id as a self-referencing FK for chained sequences.
"""

import sqlite3
import json
import datetime as dt
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.cms.models import HypothesisCard, HypothesisStatus, ManualTradeRecord
from src.sequencer.fsm import TradeLeg, TradeChain


class DatabaseRepository:
    def __init__(self, db_path: Path | str = "hypotrader.db"):
        self.db_path = str(db_path)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Creates all necessary relational tables."""
        with self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                economic_rationale TEXT NOT NULL,
                asset_symbol TEXT NOT NULL,
                author TEXT NOT NULL,
                status TEXT NOT NULL,
                target_regimes TEXT NOT NULL,
                rules_summary TEXT,
                param_manifest TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                retirement_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS manual_trades (
                id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                date TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                units REAL NOT NULL,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl_net REAL NOT NULL,
                execution_rating INTEGER,
                notes TEXT,
                tags TEXT,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS backtest_runs (
                id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                param_set TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                total_pnl_net REAL NOT NULL,
                trades_count INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                sharpe_ratio REAL,
                max_drawdown_pct REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS trade_logs (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                chain_id TEXT NOT NULL,
                depth INTEGER NOT NULL,
                direction TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                units REAL NOT NULL,
                sl_price REAL NOT NULL,
                tp_price REAL NOT NULL,
                parent_trade_id TEXT,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl_gross REAL,
                commission REAL,
                spread_cost REAL,
                slippage_cost REAL,
                pnl_net REAL,
                FOREIGN KEY (run_id) REFERENCES backtest_runs (id) ON DELETE CASCADE,
                FOREIGN KEY (parent_trade_id) REFERENCES trade_logs (id)
            );

            CREATE TABLE IF NOT EXISTS optimizer_trials (
                trial_id INTEGER PRIMARY KEY,
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

    def save_hypothesis(self, hypo: HypothesisCard) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO hypotheses (
                    id, title, economic_rationale, asset_symbol, author, status,
                    target_regimes, rules_summary, param_manifest, created_at, updated_at, retirement_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    economic_rationale=excluded.economic_rationale,
                    status=excluded.status,
                    target_regimes=excluded.target_regimes,
                    rules_summary=excluded.rules_summary,
                    param_manifest=excluded.param_manifest,
                    updated_at=excluded.updated_at,
                    retirement_reason=excluded.retirement_reason
            """, (
                hypo.id,
                hypo.title,
                hypo.economic_rationale,
                hypo.asset_symbol,
                hypo.author,
                hypo.status.value,
                json.dumps(hypo.target_regimes),
                hypo.rules_summary,
                json.dumps(hypo.param_manifest),
                hypo.created_at.isoformat(),
                hypo.updated_at.isoformat(),
                hypo.retirement_reason,
            ))

    def get_hypothesis(self, hypothesis_id: str) -> Optional[HypothesisCard]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)).fetchone()
            if row is None:
                return None
            return HypothesisCard(
                id=row["id"],
                title=row["title"],
                economic_rationale=row["economic_rationale"],
                asset_symbol=row["asset_symbol"],
                author=row["author"],
                status=HypothesisStatus(row["status"]),
                target_regimes=json.loads(row["target_regimes"]),
                rules_summary=row["rules_summary"],
                param_manifest=json.loads(row["param_manifest"]) if row["param_manifest"] else {},
                created_at=dt.datetime.fromisoformat(row["created_at"]),
                updated_at=dt.datetime.fromisoformat(row["updated_at"]),
                retirement_reason=row["retirement_reason"],
            )

    def transition_hypothesis_status(
        self,
        hypothesis_id: str,
        new_status: HypothesisStatus,
        retirement_reason: Optional[str] = None,
    ) -> bool:
        """Validates and applies hypothesis lifecycle transitions."""
        hypo = self.get_hypothesis(hypothesis_id)
        if not hypo:
            return False

        hypo.status = new_status
        hypo.updated_at = dt.datetime.now(dt.timezone.utc)
        if new_status == HypothesisStatus.RETIRED:
            hypo.retirement_reason = retirement_reason

        self.save_hypothesis(hypo)
        return True

    def save_manual_trade(self, record: ManualTradeRecord) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO manual_trades (
                    id, hypothesis_id, date, direction, entry_time, entry_price, units,
                    exit_time, exit_price, exit_reason, pnl_net, execution_rating, notes, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.hypothesis_id,
                record.date.isoformat(),
                record.direction,
                record.entry_time.isoformat(),
                record.entry_price,
                record.units,
                record.exit_time.isoformat() if record.exit_time else None,
                record.exit_price,
                record.exit_reason,
                record.pnl_net,
                record.execution_rating,
                record.notes,
                json.dumps(record.tags),
            ))

    def save_trade_chain(self, chain: TradeChain, run_id: Optional[str] = None) -> None:
        """Persists all legs in a trade chain with parent-child linkage."""
        with self._get_connection() as conn:
            for leg in chain.legs:
                conn.execute("""
                    INSERT OR REPLACE INTO trade_logs (
                        id, run_id, chain_id, depth, direction, entry_time, entry_price,
                        units, sl_price, tp_price, parent_trade_id, exit_time, exit_price,
                        exit_reason, pnl_gross, commission, spread_cost, slippage_cost, pnl_net
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    leg.leg_id,
                    run_id,
                    leg.chain_id,
                    leg.depth,
                    leg.direction,
                    leg.entry_time.isoformat(),
                    leg.entry_price,
                    leg.units,
                    leg.sl_price,
                    leg.tp_price,
                    leg.parent_trade_id,
                    leg.exit_time.isoformat() if leg.exit_time else None,
                    leg.exit_price,
                    leg.exit_reason,
                    leg.pnl_gross,
                    leg.commission,
                    leg.spread_cost,
                    leg.slippage_cost,
                    leg.pnl_net,
                ))
