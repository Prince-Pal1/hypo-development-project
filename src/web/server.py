"""
Lightweight, zero-dependency HTTP server for the HypoTrader Backtest Web UI.
Serves the interactive strategy dashboard and handles REST API requests to run backtests on demand.
"""

import http.server
import json
from pathlib import Path
import socketserver
import sys
import urllib.parse
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.test_range_sweep_v2_aug_sept_2026 import (
    run_range_sweep_v2_backtest,
    DEFAULT_PARQUET_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_DB_PATH,
)
from scripts.test_range_scope_aug_sept_2026 import run_range_scope_backtest
from src.portfolio.portfolio import PortfolioEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"
PORT = 8050


class BacktestHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Connection", "close")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index_file = STATIC_DIR / "index.html"
            if index_file.exists():
                content = index_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "index.html not found")
                return

        elif path == "/api/config":
            payload = {
                "default_params": {
                    "strategy_mode": "both",
                    "session_start": "tokyo",
                    # Strategy 1 (Scope)
                    "scope_min_x": 5.0,
                    "scope_sl": 10.0,
                    "scope_ctc": 10.0,
                    "scope_tp_offset_y": 3.5,
                    "scope_tp_mode": "mode_1_dynamic",
                    "scope_eval_time": "11:00",
                    # Strategy 2 (Sweep)
                    "x_offset": 3.5,
                    "sl1": 10.0,
                    "tp1": 20.0,
                    "ctc1": 10.0,
                    "ctc2": 10.0,
                    "decay": 15.0,
                    "sl2": 10.0,
                    "tp2": 20.0,
                    "enable_flip": True,
                    "expiry_time": "17:00",
                    "sweep_eval_time": "12:30",
                    # Common
                    "units": 10.0,
                    "start_date": "2026-08-04",
                    "end_date": "2026-09-07",
                    "initial_equity": 50000.0,
                    "broker_mode": "xm_broker",
                    "zero_fees": True,
                },
                "dataset_exists": DEFAULT_PARQUET_PATH.exists(),
                "dataset_path": str(DEFAULT_PARQUET_PATH),
                "report_path": str(DEFAULT_REPORT_PATH),
            }
            self._send_json(payload)
            return

        elif path == "/api/report":
            if DEFAULT_REPORT_PATH.exists():
                content = DEFAULT_REPORT_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "Report not yet generated. Run backtest first.")
                return

        # Serve static assets if any
        file_path = STATIC_DIR / path.lstrip("/")
        if file_path.exists() and file_path.is_file():
            content = file_path.read_bytes()
            content_type = "text/plain"
            if path.endswith(".css"):
                content_type = "text/css"
            elif path.endswith(".js"):
                content_type = "application/javascript"
            elif path.endswith(".svg"):
                content_type = "image/svg+xml"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_error(404, "Not Found")

    def _serialize_metrics(self, metrics, daily_charts=None, trade_pnls=None) -> dict:
        base = {
            "initial_equity": metrics.initial_equity,
            "final_equity": metrics.final_equity,
            "total_net_pnl": metrics.total_net_pnl,
            "total_return_pct": metrics.total_return_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "profit_factor": metrics.profit_factor,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "max_drawdown_usd": metrics.max_drawdown_usd,
            "calmar_ratio": metrics.calmar_ratio,
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate_pct": metrics.win_rate_pct,
            "average_trade_pnl": getattr(metrics, "expectancy_usd", 0.0),
            "expectancy_usd": getattr(metrics, "expectancy_usd", 0.0),
        }

        # Calculate advanced win/loss statistics
        pnls = trade_pnls or []
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (abs(sum(losses)) / len(losses)) if losses else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)

        base["avg_win_usd"] = round(avg_win, 2)
        base["avg_loss_usd"] = round(avg_loss, 2)
        base["payoff_ratio"] = round(payoff_ratio, 2)

        # Directional & Streak stats from daily_charts
        long_trades = 0
        long_wins = 0
        long_pnl = 0.0
        short_trades = 0
        short_wins = 0
        short_pnl = 0.0

        best_day_usd = 0.0
        best_day_date = "—"
        worst_day_usd = 0.0
        worst_day_date = "—"

        curr_win_streak = 0
        max_win_streak = 0
        curr_loss_streak = 0
        max_loss_streak = 0

        if daily_charts:
            for d in daily_charts:
                dpnl = float(d.get("pnl_net", 0.0))
                ddate = d.get("date", "")
                if dpnl > best_day_usd:
                    best_day_usd = dpnl
                    best_day_date = ddate
                if dpnl < worst_day_usd:
                    worst_day_usd = dpnl
                    worst_day_date = ddate

                if dpnl > 0:
                    curr_win_streak += 1
                    curr_loss_streak = 0
                    if curr_win_streak > max_win_streak:
                        max_win_streak = curr_win_streak
                elif dpnl < 0:
                    curr_loss_streak += 1
                    curr_win_streak = 0
                    if curr_loss_streak > max_loss_streak:
                        max_loss_streak = curr_loss_streak

                for leg in d.get("legs", []):
                    if leg.get("exit_price") is not None and not leg.get("is_pending", False):
                        direction = str(leg.get("direction", "")).upper()
                        lpnl = float(leg.get("pnl_net", 0.0))
                        if direction == "LONG":
                            long_trades += 1
                            long_pnl += lpnl
                            if lpnl > 0:
                                long_wins += 1
                        elif direction == "SHORT":
                            short_trades += 1
                            short_pnl += lpnl
                            if lpnl > 0:
                                short_wins += 1

        base["long_trades"] = long_trades
        base["long_wins"] = long_wins
        base["long_win_rate_pct"] = round((long_wins / long_trades * 100.0), 1) if long_trades > 0 else 0.0
        base["long_pnl_usd"] = round(long_pnl, 2)

        base["short_trades"] = short_trades
        base["short_wins"] = short_wins
        base["short_win_rate_pct"] = round((short_wins / short_trades * 100.0), 1) if short_trades > 0 else 0.0
        base["short_pnl_usd"] = round(short_pnl, 2)

        base["best_day_usd"] = round(best_day_usd, 2)
        base["best_day_date"] = best_day_date
        base["worst_day_usd"] = round(worst_day_usd, 2)
        base["worst_day_date"] = worst_day_date
        base["max_consecutive_wins"] = max_win_streak
        base["max_consecutive_losses"] = max_loss_streak

        return base

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except Exception as e:
                self._send_json({"error": f"Invalid JSON payload: {str(e)}"}, status=400)
                return

            try:
                strategy_mode = str(data.get("strategy_mode", "both")).strip().lower()
                session_start = str(data.get("session_start", "tokyo")).lower().strip()
                start_date = str(data.get("start_date", "2026-08-04")).strip()
                end_date = str(data.get("end_date", "2026-09-07")).strip()
                initial_equity = float(data.get("initial_equity", 50000.0))
                units = float(data.get("units", 10.0))
                broker_mode = str(data.get("broker_mode", "xm_broker")).strip().lower()
                zero_fees = bool(data.get("zero_fees", True)) if "zero_fees" in data else (broker_mode == "xm_broker")

                # Strategy 2 (Sweep) parameters
                x_offset = float(data.get("x_offset", 3.5))
                sl1 = float(data.get("sl1", 10.0))
                tp1 = float(data.get("tp1", 20.0))
                ctc1 = float(data.get("ctc1", 10.0))
                ctc2 = float(data.get("ctc2", 10.0))
                decay = float(data.get("decay", 15.0))
                sl2 = float(data.get("sl2", 10.0))
                tp2 = float(data.get("tp2", 20.0))
                enable_flip = bool(data.get("enable_flip", True))
                expiry_time = str(data.get("expiry_time", "17:00")).strip()
                sweep_eval_time = str(data.get("sweep_eval_time", "12:30")).strip()

                # Strategy 1 (Scope) parameters
                scope_min_x = float(data.get("scope_min_x", 5.0))
                scope_sl = float(data.get("scope_sl", 10.0))
                scope_ctc = float(data.get("scope_ctc", 10.0))
                scope_tp_offset_y = float(data.get("scope_tp_offset_y", 3.5))
                scope_tp_mode = str(data.get("scope_tp_mode", "mode_1_dynamic")).strip()
                scope_eval_time = str(data.get("scope_eval_time", "11:00")).strip()

                if strategy_mode == "range_sweep":
                    result = run_range_sweep_v2_backtest(
                        x_offset=x_offset,
                        sl1=sl1,
                        tp1=tp1,
                        ctc1=ctc1,
                        ctc2=ctc2,
                        decay=decay,
                        sl2=sl2,
                        tp2=tp2,
                        enable_flip=enable_flip,
                        expiry_time=expiry_time,
                        session_start=session_start,
                        eval_time_ist=sweep_eval_time,
                        units=units,
                        start_date=start_date,
                        end_date=end_date,
                        initial_equity=initial_equity,
                        zero_fees=zero_fees,
                        broker_mode=broker_mode,
                        dataset_path=str(DEFAULT_PARQUET_PATH),
                        output_path=str(DEFAULT_REPORT_PATH),
                        db_path=str(DEFAULT_DB_PATH),
                    )
                    metrics = result["metrics"]
                    chains = result["executed_chains"]
                    trade_pnls = result.get("trade_pnls", [leg.pnl_net for c in chains for leg in c.legs if leg.exit_price is not None and not leg.is_pending])
                    flips_count = sum(1 for c in chains if len(c.legs) > 1)
                    ctc_count = sum(1 for c in chains for leg in c.legs if leg.exit_reason == "CTC")
                    decay_count = sum(1 for c in chains for leg in c.legs if leg.exit_reason == "DECAY")
                    expiry_count = sum(
                        1 for c in chains for leg in c.legs
                        if leg.exit_reason == "EXPIRATION" and leg.pnl_net == 0 and leg.is_pending is False
                    )

                    daily_charts = result["daily_chart_records"]
                    s2_tag = f"RS-{sweep_eval_time.replace(':', '')}"
                    for d in daily_charts:
                        d["strategy_tag"] = s2_tag
                        for l in d.get("legs", []):
                            l["strategy_tag"] = s2_tag

                    wins_count = sum(1 for d in daily_charts if d.get("pnl_net", 0.0) > 0)
                    losses_count = sum(1 for d in daily_charts if d.get("pnl_net", 0.0) < 0)
                    ties_count = sum(1 for d in daily_charts if d.get("day_type") in ("TIE", "SKIPPED"))
                    leg1_wins = sum(1 for c in chains if len(c.legs) > 0 and c.legs[0].exit_reason == "TP")
                    leg1_losses = sum(1 for c in chains if len(c.legs) > 0 and c.legs[0].exit_reason == "SL")
                    leg2_wins = sum(1 for c in chains if len(c.legs) > 1 and c.legs[1].exit_reason == "TP")
                    leg2_losses = sum(1 for c in chains if len(c.legs) > 1 and c.legs[1].exit_reason == "SL")

                    response_data = {
                        "status": "success",
                        "strategy_mode": "range_sweep",
                        "parameters": {
                            "strategy_mode": "range_sweep",
                            "session_start": session_start,
                            "x_offset": x_offset,
                            "sl1": sl1,
                            "tp1": tp1,
                            "ctc1": ctc1,
                            "ctc2": ctc2,
                            "decay": decay,
                            "sl2": sl2,
                            "tp2": tp2,
                            "enable_flip": enable_flip,
                            "expiry_time": expiry_time,
                            "sweep_eval_time": sweep_eval_time,
                            "units": units,
                            "start_date": start_date,
                            "end_date": end_date,
                            "initial_equity": initial_equity,
                            "zero_fees": zero_fees,
                            "broker_mode": broker_mode,
                        },
                        "metrics": self._serialize_metrics(metrics, daily_charts=daily_charts, trade_pnls=trade_pnls),
                        "rule_triggers": {
                            "all_days": len(daily_charts),
                            "traded_days": len(chains),
                            "wins": wins_count,
                            "losses": losses_count,
                            "ties": ties_count,
                            "flips_triggered": flips_count,
                            "leg1_wins": leg1_wins,
                            "leg1_losses": leg1_losses,
                            "leg2_wins": leg2_wins,
                            "leg2_losses": leg2_losses,
                            "ctc_exits": ctc_count,
                            "decay_cancels": decay_count,
                            "expiry_cancels": expiry_count,
                        },
                        "daily_equity": [
                            {"date": str(d), "equity": round(float(eq), 2)}
                            for d, eq in result["daily_equity_series"].items()
                        ],
                        "daily_charts": daily_charts,
                        "report_url": "/api/report",
                    }

                elif strategy_mode == "range_scope":
                    result = run_range_scope_backtest(
                        session_start=session_start,
                        eval_time_ist=scope_eval_time,
                        scope_min_x=scope_min_x,
                        sl_points=scope_sl,
                        ctc_points=scope_ctc,
                        tp_offset_y=scope_tp_offset_y,
                        tp_mode=scope_tp_mode,
                        units=units,
                        start_date=start_date,
                        end_date=end_date,
                        initial_equity=initial_equity,
                        zero_fees=zero_fees,
                        broker_mode=broker_mode,
                        dataset_path=str(DEFAULT_PARQUET_PATH),
                    )
                    metrics = result["metrics"]
                    trade_results = result["trade_results"]
                    trade_pnls = [t.pnl_net for t in trade_results if t.exit_price is not None]
                    tp_boundary_count = sum(1 for t in trade_results if t.exit_reason == "TP")
                    tp_lockin_count = sum(1 for t in trade_results if t.exit_reason == "TP_1230_LOCKIN")
                    tp_count = tp_boundary_count + tp_lockin_count
                    sl_count = sum(1 for t in trade_results if t.exit_reason == "SL")
                    ctc_count = sum(1 for t in trade_results if t.exit_reason == "CTC")
                    skip_count = sum(1 for r in result["daily_chart_records"] if r.get("day_type") == "SKIP")
                    eod_count = sum(1 for t in trade_results if t.exit_reason == "EOD")

                    daily_charts = result["daily_chart_records"]
                    s1_tag = f"SCOPE-{scope_eval_time.replace(':', '')}"
                    for d in daily_charts:
                        d["strategy_tag"] = s1_tag
                        for l in d.get("legs", []):
                            l["strategy_tag"] = s1_tag

                    wins_count = sum(1 for d in daily_charts if d.get("pnl_net", 0.0) > 0)
                    losses_count = sum(1 for d in daily_charts if d.get("pnl_net", 0.0) < 0)

                    response_data = {
                        "status": "success",
                        "strategy_mode": "range_scope",
                        "parameters": {
                            "strategy_mode": "range_scope",
                            "session_start": session_start,
                            "scope_min_x": scope_min_x,
                            "scope_sl": scope_sl,
                            "scope_ctc": scope_ctc,
                            "scope_tp_offset_y": scope_tp_offset_y,
                            "scope_tp_mode": scope_tp_mode,
                            "scope_eval_time": scope_eval_time,
                            "units": units,
                            "start_date": start_date,
                            "end_date": end_date,
                            "initial_equity": initial_equity,
                            "zero_fees": zero_fees,
                            "broker_mode": broker_mode,
                        },
                        "metrics": self._serialize_metrics(metrics, daily_charts=daily_charts, trade_pnls=trade_pnls),
                        "rule_triggers": {
                            "all_days": len(daily_charts),
                            "traded_days": len(trade_results),
                            "wins": wins_count,
                            "losses": losses_count,
                            "tp_hits": tp_count,
                            "tp_boundary_hits": tp_boundary_count,
                            "tp_1230_lockins": tp_lockin_count,
                            "sl_hits": sl_count,
                            "ctc_exits": ctc_count,
                            "scope_skips": skip_count,
                            "eod_exits": eod_count,
                            "strat1_trades": len(trade_results),
                            "strat1_wins": wins_count,
                            "strat1_losses": losses_count,
                        },
                        "daily_equity": [
                            {"date": str(d), "equity": round(float(eq), 2)}
                            for d, eq in result["daily_equity_series"].items()
                        ],
                        "daily_charts": daily_charts,
                        "report_url": "/api/report",
                    }

                else:  # "both" (Portfolio Mode)
                    res_sweep = run_range_sweep_v2_backtest(
                        x_offset=x_offset,
                        sl1=sl1,
                        tp1=tp1,
                        ctc1=ctc1,
                        ctc2=ctc2,
                        decay=decay,
                        sl2=sl2,
                        tp2=tp2,
                        enable_flip=enable_flip,
                        expiry_time=expiry_time,
                        session_start=session_start,
                        eval_time_ist=sweep_eval_time,
                        units=units,
                        start_date=start_date,
                        end_date=end_date,
                        initial_equity=initial_equity,
                        zero_fees=zero_fees,
                        broker_mode=broker_mode,
                        dataset_path=str(DEFAULT_PARQUET_PATH),
                        output_path=str(DEFAULT_REPORT_PATH),
                        db_path=str(DEFAULT_DB_PATH),
                    )
                    res_scope = run_range_scope_backtest(
                        session_start=session_start,
                        eval_time_ist=scope_eval_time,
                        scope_min_x=scope_min_x,
                        sl_points=scope_sl,
                        ctc_points=scope_ctc,
                        tp_offset_y=scope_tp_offset_y,
                        tp_mode=scope_tp_mode,
                        units=units,
                        start_date=start_date,
                        end_date=end_date,
                        initial_equity=initial_equity,
                        zero_fees=zero_fees,
                        broker_mode=broker_mode,
                        dataset_path=str(DEFAULT_PARQUET_PATH),
                    )

                    c_sweep_map = {r["date"]: r for r in res_sweep["daily_chart_records"]}
                    c_scope_map = {r["date"]: r for r in res_scope["daily_chart_records"]}
                    all_dates = sorted(list(set(list(c_sweep_map.keys()) + list(c_scope_map.keys()))))

                    cum_equity = initial_equity
                    combined_daily_equity = {}
                    combined_charts = []
                    all_trade_pnls = []

                    s1_tag = f"SCOPE-{scope_eval_time.replace(':', '')}"
                    s2_tag = f"RS-{sweep_eval_time.replace(':', '')}"

                    for d_str in all_dates:
                        c_sweep = c_sweep_map.get(d_str)
                        c_scope = c_scope_map.get(d_str)
                        pnl_sweep = c_sweep.get("pnl_net", 0.0) if c_sweep else 0.0
                        pnl_scope = c_scope.get("pnl_net", 0.0) if c_scope else 0.0
                        day_pnl = pnl_scope + pnl_sweep
                        cum_equity += day_pnl
                        combined_daily_equity[d_str] = cum_equity

                        legs = []
                        # Strategy 1 (Scope) legs first
                        if c_scope:
                            for leg in c_scope.get("legs", []):
                                lc = dict(leg)
                                lc["strategy_tag"] = s1_tag
                                legs.append(lc)
                                if lc.get("exit_price") is not None and lc.get("exit_reason") != "NO_TRADE":
                                    all_trade_pnls.append(lc.get("pnl_net", 0.0))
                        # Strategy 2 (Sweep & Flip) legs second
                        if c_sweep:
                            for leg in c_sweep.get("legs", []):
                                lc = dict(leg)
                                lc["strategy_tag"] = s2_tag
                                legs.append(lc)
                                if not lc.get("is_pending", False) and lc.get("exit_price") is not None:
                                    all_trade_pnls.append(lc.get("pnl_net", 0.0))

                        bars = (c_sweep and c_sweep.get("bars")) or (c_scope and c_scope.get("bars")) or []
                        sessions = dict((c_sweep and c_sweep.get("sessions")) or (c_scope and c_scope.get("sessions")) or {})
                        if c_scope and "sessions" in c_scope:
                            sessions["eval_1100"] = c_scope["sessions"].get("eval_1100") or c_scope["sessions"].get("eval_time")
                            sessions["eval_1230"] = c_scope["sessions"].get("eval_1230") or sessions.get("eval_1230")
                            sessions["eval_scope"] = c_scope["sessions"].get("eval_time")
                        if c_sweep and "sessions" in c_sweep:
                            sessions["eval_1230"] = c_sweep["sessions"].get("eval_1230") or c_sweep["sessions"].get("eval_time")
                            sessions["eval_1100"] = c_sweep["sessions"].get("eval_1100") or sessions.get("eval_1100")
                            sessions["eval_sweep"] = c_sweep["sessions"].get("eval_time")

                        p1100_val = c_scope.get("eval_price") if c_scope else (c_sweep.get("price_1100") if c_sweep else None)
                        p1230_val = c_sweep.get("eval_price") if c_sweep else (c_scope.get("price_1230") if c_scope else None)

                        sum_scope = c_scope.get("outcome_summary", "No trade") if c_scope else "N/A"
                        sum_sweep = c_sweep.get("outcome_summary", "No trade") if c_sweep else "N/A"
                        outcome_summary = f"[{s1_tag}]: {sum_scope} | [{s2_tag}]: {sum_sweep}"

                        day_type = "TRADE" if (pnl_scope != 0 or pnl_sweep != 0 or (c_scope and c_scope.get("day_type") == "TRADE") or (c_sweep and c_sweep.get("day_type") == "TRADE")) else "SKIP"

                        combined_charts.append({
                            "date": d_str,
                            "day_type": day_type,
                            "day_of_week": c_scope.get("day_of_week") if c_scope else (c_sweep.get("day_of_week") if c_sweep else ""),
                            "strategy_tag": "PORTFOLIO",
                            "range_high": c_sweep.get("range_high") if c_sweep else (c_scope.get("range_high") if c_scope else 0.0),
                            "range_low": c_sweep.get("range_low") if c_sweep else (c_scope.get("range_low") if c_scope else 0.0),
                            "range_1100_high": c_scope.get("range_high") if c_scope else None,
                            "range_1100_low": c_scope.get("range_low") if c_scope else None,
                            "range_1230_high": c_sweep.get("range_high") if c_sweep else None,
                            "range_1230_low": c_sweep.get("range_low") if c_sweep else None,
                            "eval_1100": p1100_val,
                            "eval_1230": p1230_val,
                            "price_1100": p1100_val,
                            "price_1230": p1230_val,
                            "eval_scope": p1100_val,
                            "eval_sweep": p1230_val,
                            "scope_1100": c_scope.get("scope_points") if c_scope else None,
                            "midpoint": c_sweep.get("midpoint") if c_sweep else (c_scope.get("midpoint") if c_scope else 0.0),
                            "range_pts": c_sweep.get("range_pts") if c_sweep else (c_scope.get("range_pts") if c_scope else 0.0),
                            "eval_price": c_sweep.get("eval_price") if c_sweep else (c_scope.get("eval_price") if c_scope else 0.0),
                            "pnl_net": round(day_pnl, 2),
                            "outcome_summary": outcome_summary,
                            "legs": legs,
                            "bars": bars,
                            "sessions": sessions,
                            "strat1": c_scope,
                            "strat2": c_sweep,
                        })

                    portfolio_engine = PortfolioEngine(initial_equity=initial_equity)
                    combined_equity_series = pd.Series(combined_daily_equity)
                    combined_metrics = portfolio_engine.compute_metrics(combined_equity_series, all_trade_pnls)

                    port_wins = sum(1 for d in combined_charts if d.get("pnl_net", 0.0) > 0)
                    port_losses = sum(1 for d in combined_charts if d.get("pnl_net", 0.0) < 0)

                    s_sweep_chains = res_sweep["executed_chains"]
                    s2_flips = sum(1 for c in s_sweep_chains if len(c.legs) > 1)
                    s2_ctc = sum(1 for c in s_sweep_chains for leg in c.legs if leg.exit_reason == "CTC")
                    s2_decay = sum(1 for c in s_sweep_chains for leg in c.legs if leg.exit_reason == "DECAY")
                    s1_trades = res_scope["trade_results"]
                    s1_lockins = sum(1 for t in s1_trades if t.exit_reason == "TP_1230_LOCKIN")
                    s1_ctc = sum(1 for t in s1_trades if t.exit_reason == "CTC")
                    s1_skips = sum(1 for r in res_scope["daily_chart_records"] if r.get("day_type") == "SKIP")

                    response_data = {
                        "status": "success",
                        "strategy_mode": "both",
                        "parameters": {
                            "strategy_mode": "both",
                            "session_start": session_start,
                            "x_offset": x_offset,
                            "sl1": sl1,
                            "tp1": tp1,
                            "ctc1": ctc1,
                            "ctc2": ctc2,
                            "decay": decay,
                            "sl2": sl2,
                            "tp2": tp2,
                            "enable_flip": enable_flip,
                            "expiry_time": expiry_time,
                            "sweep_eval_time": sweep_eval_time,
                            "scope_min_x": scope_min_x,
                            "scope_sl": scope_sl,
                            "scope_ctc": scope_ctc,
                            "scope_tp_offset_y": scope_tp_offset_y,
                            "scope_tp_mode": scope_tp_mode,
                            "scope_eval_time": scope_eval_time,
                            "units": units,
                            "start_date": start_date,
                            "end_date": end_date,
                            "initial_equity": initial_equity,
                            "zero_fees": zero_fees,
                            "broker_mode": broker_mode,
                        },
                        "metrics": self._serialize_metrics(combined_metrics, daily_charts=combined_charts, trade_pnls=all_trade_pnls),
                        "rule_triggers": {
                            "all_days": len(combined_charts),
                            "port_wins": port_wins,
                            "port_losses": port_losses,
                            "strat1_trades": res_scope["metrics"].total_trades,
                            "strat1_wins": res_scope["metrics"].winning_trades,
                            "strat1_losses": res_scope["metrics"].losing_trades,
                            "strat1_lockins": s1_lockins,
                            "strat1_ctc": s1_ctc,
                            "strat1_skips": s1_skips,
                            "strat1_net": res_scope["metrics"].total_net_pnl,
                            "strat2_trades": res_sweep["metrics"].total_trades,
                            "strat2_wins": res_sweep["metrics"].winning_trades,
                            "strat2_losses": res_sweep["metrics"].losing_trades,
                            "strat2_flips": s2_flips,
                            "strat2_ctc": s2_ctc,
                            "strat2_decay": s2_decay,
                            "strat2_net": res_sweep["metrics"].total_net_pnl,
                        },
                        "daily_equity": [
                            {"date": str(d), "equity": round(float(eq), 2)}
                            for d, eq in combined_daily_equity.items()
                        ],
                        "daily_equity_s1": [
                            {"date": str(d), "equity": round(float(eq), 2)}
                            for d, eq in res_scope["daily_equity_series"].items()
                        ],
                        "daily_equity_s2": [
                            {"date": str(d), "equity": round(float(eq), 2)}
                            for d, eq in res_sweep["daily_equity_series"].items()
                        ],
                        "daily_charts": combined_charts,
                        "report_url": "/api/report",
                    }

                self._send_json(response_data)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        self.send_error(404, "Not Found")

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        try:
            sys.stderr.write(f"[WebUI Server] {format % args}\n")
        except Exception:
            sys.stderr.write(f"[WebUI Server] {format}\n")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server(port: int = PORT) -> None:
    # Ensure static directory exists
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    server_address = ("", port)
    with ThreadedHTTPServer(server_address, BacktestHandler) as httpd:
        print("=" * 70)
        print(f"HYPOTRADER BACKTEST WEB UI SERVER RUNNING")
        print(f"URL: http://localhost:{port}")
        print("=" * 70)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.shutdown()


run_server = start_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    start_server(port)
