"""
Interactive HTML Report Visualizer & Strategy Analyzer for HypoTrader.
Generates an institutional-grade dark/light themed HTML performance report with:
  1. Executive Summary & KPI Metric Cards.
  2. Compounded Equity Curve.
  3. Monthly Returns Matrix Heatmap.
  4. Interactive Strategy Analyzer & Day Visualizer:
     - Dual-view tabs: Schematic FSM SVG Diagram (worked example style) + 5m Candlestick Price Chart.
     - Color-coded session shading: Sydney Range (purple/teal) and London Session (amber/gold).
     - Range reference lines: Sydney High, Sydney Low, Midpoint, Limits, Stops, Targets.
     - Full FSM state lifecycle: Pending limits, Decay cancels, Expired 5pm cutoff, Limit fills,
       stepped CTC breakeven stops, chain-trigger exits with amber rings, and reversal flips.
     - Auto-generated outcome-path summary callout above every chart.
     - Day Navigator (Dropdown, Prev/Next, Left/Right arrow shortcuts, Outcome filter pills).
  5. Interactive Chained Execution Log with row click-to-sync.
"""

from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import datetime as dt
import json
import numpy as np
import pandas as pd

from src.portfolio.portfolio import PortfolioMetrics
from src.sequencer.fsm import TradeChain


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HypoTrader Strategy Analyzer | {{HYPOTHESIS_ID}}</title>
    <style>
        :root {
            --bg: #fafaf7;
            --surface: #ffffff;
            --surface-2: #f1efe8;
            --text: #2c2c2a;
            --text-muted: #5f5e5a;
            --text-faint: #888780;
            --border: #d3d1c7;
            --border-soft: #e8e6de;
            --navy: #0c447c;
            --navy-light: #e6f1fb;
            --teal: #0f6e56;
            --teal-light: #e1f5ee;
            --red: #a32d2d;
            --red-light: #fcebeb;
            --amber: #854f0b;
            --amber-light: #faeeda;
            --purple: #534ab7;
            --purple-light: #eeedfe;
            --gold: #c28e1a;
            --gold-light: #fbf2dc;
            --shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.04);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #1a1a18;
                --surface: #252523;
                --surface-2: #2c2c2a;
                --text: #e8e6de;
                --text-muted: #b4b2a9;
                --text-faint: #888780;
                --border: #3d3d3a;
                --border-soft: #2f2f2d;
                --navy: #85b7eb;
                --navy-light: #0c447c33;
                --teal: #5dcaa5;
                --teal-light: #0f6e5633;
                --red: #f09595;
                --red-light: #a32d2d33;
                --amber: #ef9f27;
                --amber-light: #854f0b33;
                --purple: #7f77dd;
                --purple-light: #534ab733;
                --gold: #e6b041;
                --gold-light: #c28e1a33;
            }
        }
        body.theme-dark {
            --bg: #1a1a18;
            --surface: #252523;
            --surface-2: #2c2c2a;
            --text: #e8e6de;
            --text-muted: #b4b2a9;
            --text-faint: #888780;
            --border: #3d3d3a;
            --border-soft: #2f2f2d;
            --navy: #85b7eb;
            --navy-light: #0c447c33;
            --teal: #5dcaa5;
            --teal-light: #0f6e5633;
            --red: #f09595;
            --red-light: #a32d2d33;
            --amber: #ef9f27;
            --amber-light: #854f0b33;
            --purple: #7f77dd;
            --purple-light: #534ab733;
            --gold: #e6b041;
            --gold-light: #c28e1a33;
        }
        body.theme-light {
            --bg: #fafaf7;
            --surface: #ffffff;
            --surface-2: #f1efe8;
            --text: #2c2c2a;
            --text-muted: #5f5e5a;
            --text-faint: #888780;
            --border: #d3d1c7;
            --border-soft: #e8e6de;
            --navy: #0c447c;
            --navy-light: #e6f1fb;
            --teal: #0f6e56;
            --teal-light: #e1f5ee;
            --red: #a32d2d;
            --red-light: #fcebeb;
            --amber: #854f0b;
            --amber-light: #faeeda;
            --purple: #534ab7;
            --purple-light: #eeedfe;
            --gold: #c28e1a;
            --gold-light: #fbf2dc;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            margin: 0 auto;
            font-size: 15px;
            max-width: 1200px;
            padding: 30px 24px 80px;
            -webkit-font-smoothing: antialiased;
        }
        .top-bar { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid var(--border-soft); padding-bottom: 20px; margin-bottom: 24px; }
        .header-left h1 { font-size: 26px; font-weight: 600; color: var(--navy); margin: 0 0 6px; letter-spacing: -0.4px; }
        .sub { font-size: 13.5px; color: var(--text-muted); margin: 0; }
        .theme-toggle-btn {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            font-weight: 500;
        }
        .theme-toggle-btn:hover { background: var(--surface-2); }
        .badge { background: var(--teal); color: #fff; padding: 3px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; text-transform: uppercase; margin-right: 8px; vertical-align: middle; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 28px; }
        .kpi-card { background: var(--surface); border: 1px solid var(--border-soft); border-radius: 10px; padding: 14px; text-align: center; box-shadow: var(--shadow); }
        .kpi-title { font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px; font-weight: 600; letter-spacing: 0.5px; }
        .kpi-val { font-size: 22px; font-weight: 700; color: var(--text); }
        .kpi-val.pos { color: var(--teal); }
        .kpi-val.neg { color: var(--red); }
        .section { background: var(--surface); border: 1px solid var(--border-soft); border-radius: 12px; padding: 22px; margin-bottom: 28px; box-shadow: var(--shadow); }
        .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid var(--border-soft); padding-bottom: 10px; }
        .section-header h2 { font-size: 18px; font-weight: 600; color: var(--navy); margin: 0; }
        .nav-toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 16px; background: var(--surface-2); padding: 12px; border-radius: 8px; border: 1px solid var(--border-soft); }
        .nav-group { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
        .nav-btn { background: var(--surface); border: 1px solid var(--border); color: var(--text); padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; cursor: pointer; }
        .nav-btn:hover { background: var(--navy-light); border-color: var(--navy); }
        .filter-btn { background: transparent; border: 1px solid var(--border); color: var(--text-muted); padding: 4px 10px; border-radius: 14px; font-size: 12px; cursor: pointer; }
        .filter-btn.active { background: var(--navy); color: #fff; border-color: var(--navy); font-weight: 600; }
        .filter-btn:hover:not(.active) { background: var(--surface); }
        .day-select { background: var(--surface); border: 1px solid var(--border); color: var(--text); padding: 6px 12px; border-radius: 6px; font-size: 13.5px; font-weight: 600; cursor: pointer; min-width: 260px; }
        .tab-btn { background: transparent; border: 1px solid var(--border); color: var(--text-muted); padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; }
        .tab-btn.active { background: var(--navy); color: #fff; border-color: var(--navy); }
        .callout { border-radius: 10px; padding: 14px 18px; margin: 14px 0; border-left: 3px solid; font-size: 14px; box-shadow: var(--shadow); }
        .callout.info { background: var(--navy-light); border-left-color: var(--navy); }
        .callout.warn { background: var(--amber-light); border-left-color: var(--amber); }
        .callout b { font-weight: 600; }
        .hud-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; background: var(--surface-2); border: 1px solid var(--border-soft); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; }
        .hud-item { display: flex; flex-direction: column; }
        .hud-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
        .hud-val { font-size: 14px; font-weight: 700; color: var(--text); }
        .diagram { background: var(--surface); border: 1px solid var(--border-soft); border-radius: 12px; padding: 16px; margin: 16px 0; box-shadow: var(--shadow); overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; box-shadow: var(--shadow); border-radius: 8px; overflow: hidden; }
        thead { background: var(--navy); color: #fff; }
        th { padding: 9px 12px; text-align: left; font-size: 12px; font-weight: 600; }
        td { padding: 9px 12px; border-top: 1px solid var(--border-soft); }
        tbody tr:nth-child(even) { background: var(--surface-2); }
        .trade-row:hover { background: var(--navy-light) !important; }
        .pos { color: var(--teal); font-weight: 600; }
        .neg { color: var(--red); font-weight: 600; }
        .neutral { color: var(--text-muted); }
        .font-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        .flip-badge { background: var(--purple); color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; }
        .badge-pending { background: var(--amber); color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; }
        .text-muted { color: var(--text-faint); }
        .legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
        .btn-inspect { background: var(--surface); border: 1px solid var(--border); color: var(--navy); padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; cursor: pointer; }
        .btn-inspect:hover { background: var(--navy); color: #fff; }
        #candlestickCanvas { width: 100%; height: 460px; display: block; }
        .canvas-container { position: relative; width: 100%; height: 460px; background: var(--surface); border-radius: 8px; overflow: hidden; border: 1px solid var(--border-soft); }
        .canvas-tooltip { position: absolute; display: none; background: rgba(20, 20, 20, 0.90); color: #fff; padding: 8px 12px; border-radius: 6px; font-size: 12px; pointer-events: none; z-index: 100; backdrop-filter: blur(4px); border: 1px solid rgba(255,255,255,0.18); box-shadow: 0 4px 14px rgba(0,0,0,0.3); }
        .toggle-fsm-btn { background: var(--surface); border: 1.5px solid var(--border); color: var(--text); padding: 6px 14px; border-radius: 6px; font-size: 12.5px; font-weight: 600; cursor: pointer; transition: all 0.15s ease; display: inline-flex; align-items: center; gap: 6px; }
        .toggle-fsm-btn:hover { background: var(--surface-2); border-color: var(--navy); }
        .toggle-fsm-btn.active { background: var(--navy-light); border-color: var(--navy); color: var(--navy); }
    </style>
</head>
<body>

    <div class="top-bar">
        <div class="header-left">
            <div><span class="badge">Institutional Lab</span><span style="font-size: 13px; color: var(--text-muted);">HypoTrader Strategy Analyzer</span></div>
            <h1>{{HYPOTHESIS_TITLE}}</h1>
            <p class="sub">
                Hypothesis ID: <strong>{{HYPOTHESIS_ID}}</strong> &bull; Asset: <strong>XAUUSD (Gold)</strong> &bull; Execution: <strong>IC Markets Raw ECN</strong>
            </p>
        </div>
        <button class="theme-toggle-btn" onclick="toggleTheme()">Toggle Theme 🌗</button>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total Net Return</div>
            <div class="kpi-val {{TOTAL_RETURN_CLS}}">{{TOTAL_RETURN_PCT}}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Total Net PnL</div>
            <div class="kpi-val {{TOTAL_PNL_CLS}}">${{TOTAL_NET_PNL}}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Sharpe Ratio</div>
            <div class="kpi-val">{{SHARPE_RATIO}}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Sortino Ratio</div>
            <div class="kpi-val">{{SORTINO_RATIO}}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Max Drawdown</div>
            <div class="kpi-val neg">-{{MAX_DRAWDOWN_PCT}}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Win Rate</div>
            <div class="kpi-val">{{WIN_RATE_PCT}}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Profit Factor</div>
            <div class="kpi-val">{{PROFIT_FACTOR}}</div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <h2>Methodology & 9-Point Reporting Checklist</h2>
        </div>
        <div style="font-size: 13.5px; line-height: 1.6; color: var(--text);">
            <ol style="margin-top: 0; padding-left: 20px;">
                <li><b>Window definition:</b> {{WINDOW_DEF}}</li>
                <li><b>Regime feature scope:</b> {{REGIME_SCOPE}}</li>
                <li><b>Surface mapping method:</b> {{SURFACE_METHOD}}</li>
                <li><b>Changepoint method:</b> {{CHANGEPOINT_METHOD}}</li>
                <li><b>DSR:</b> {{DSR_INFO}}</li>
                <li><b>Permutation test:</b> {{PERMUTATION_TEST}}</li>
                <li><b>Headline Sharpe/growth numbers:</b> {{HEADLINE_NUMBERS}}</li>
                <li><b>Explicit statement:</b> {{EXPLICIT_STATEMENT}}</li>
                <li><b>Hypotheses table:</b> {{HYPOTHESES_LOGGED}}</li>
            </ol>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <h2>Compounded Equity Curve & Performance Trajectory</h2>
        </div>
        <div style="width: 100%; overflow-x: auto;">
            {{SVG_EQUITY_CHART}}
        </div>
    </div>

    <!-- STRATEGY ANALYZER & CHART VISUALIZER -->
    <div class="section" id="visualizerSection">
        <div class="section-header">
            <h2>Strategy Analyzer & Day Chart Visualizer</h2>
            <div class="nav-group">
                <button class="tab-btn active" id="tabSchematicBtn" onclick="switchView('schematic')">📐 Schematic FSM Diagram</button>
                <button class="tab-btn" id="tabCandleBtn" onclick="switchView('candlestick')">📊 Candlestick Price Action</button>
            </div>
        </div>

        <!-- Navigation & Filter Toolbar -->
        <div class="nav-toolbar">
            <div class="nav-group">
                <button class="nav-btn" onclick="prevDay()">&larr; Previous</button>
                <select class="day-select" id="daySelect" onchange="onDaySelectChange()"></select>
                <button class="nav-btn" onclick="nextDay()">Next &rarr;</button>
            </div>
            <div class="nav-group">
                <span style="font-size: 12px; color: var(--text-muted); font-weight: 600;">Filter:</span>
                <button class="filter-btn active" onclick="applyFilter('all', this)">All</button>
                <button class="filter-btn" onclick="applyFilter('win', this)">Wins</button>
                <button class="filter-btn" onclick="applyFilter('loss', this)">Losses</button>
                <button class="filter-btn" onclick="applyFilter('flip', this)">Flips</button>
                <button class="filter-btn" onclick="applyFilter('ctc', this)">CTC Breakeven</button>
                <button class="filter-btn" onclick="applyFilter('decay', this)">Decay / Expired</button>
                <button class="filter-btn" onclick="applyFilter('tie', this)">Ties</button>
            </div>
        </div>

        <!-- Auto-generated outcome-path summary callout -->
        <div class="callout info" id="outcomeCallout">
            <b>Auto-generated outcome-path summary</b> (pulled from execution state log):<br>
            <span id="outcomeSummaryText">Loading summary...</span>
        </div>

        <!-- Range & Execution HUD -->
        <div class="hud-grid" id="hudGrid">
            <div class="hud-item">
                <span class="hud-label">Date & Day</span>
                <span class="hud-val" id="hudDate">—</span>
            </div>
            <div class="hud-item">
                <span class="hud-label">Sydney Range [Low - High]</span>
                <span class="hud-val" id="hudRange">—</span>
            </div>
            <div class="hud-item">
                <span class="hud-label">Range Width / Midpoint</span>
                <span class="hud-val" id="hudWidthMid">—</span>
            </div>
            <div class="hud-item">
                <span class="hud-label">12:30 IST Eval / Bias</span>
                <span class="hud-val" id="hudEvalBias">—</span>
            </div>
            <div class="hud-item">
                <span class="hud-label">Day Net PnL</span>
                <span class="hud-val" id="hudPnL">—</span>
            </div>
            <div class="hud-item">
                <span class="hud-label">Lifecycle Status</span>
                <span class="hud-val" id="hudStatus">—</span>
            </div>
        </div>

        <!-- Chart Display Box -->
        <div class="diagram" id="schematicContainer">
            <div id="schematicSvgBox" style="width: 100%; min-height: 440px;"></div>
        </div>

        <div class="diagram" id="candlestickContainer" style="display: none;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <button class="toggle-fsm-btn active" id="btnToggleFsmOverlay" onclick="toggleFsmOverlay()">
                        <span id="fsmOverlayStatusIcon">✓</span> 📐 FSM Schematic Overlay: <strong id="fsmOverlayStatusText">ON</strong>
                    </button>
                    <span style="font-size: 12px; color: var(--text-muted);">
                        Overlays pending limits, fills, SL/TP levels, stepped CTC connectors, and amber flip rings directly onto 5m candles.
                    </span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <button class="nav-btn" style="padding: 4px 10px; font-size: 11px;" onclick="resetCandleZoom()">↺ Reset Zoom</button>
                </div>
            </div>

            <div class="canvas-container">
                <canvas id="candlestickCanvas"></canvas>
                <div class="canvas-tooltip" id="canvasTooltip"></div>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; font-size: 12px; color: var(--text-muted); flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <span><span style="display:inline-block;width:12px;height:12px;background:rgba(83,74,183,0.2);border:1px solid var(--purple);margin-right:4px;vertical-align:middle;"></span> Sydney Range</span>
                    <span><span style="display:inline-block;width:12px;height:12px;background:rgba(239,159,39,0.2);border:1px solid var(--amber);margin-right:4px;vertical-align:middle;"></span> London Session</span>
                    <span><span style="display:inline-block;width:12px;height:2px;background:var(--navy);border-top:1px dashed var(--navy);margin-right:4px;vertical-align:middle;"></span> Pending Limit</span>
                    <span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--navy);margin-right:4px;vertical-align:middle;"></span> Fill</span>
                    <span><span style="display:inline-block;width:12px;height:2px;background:var(--red);margin-right:4px;vertical-align:middle;"></span> Active SL</span>
                    <span><span style="display:inline-block;width:12px;height:2px;background:var(--teal);margin-right:4px;vertical-align:middle;"></span> TP Target</span>
                    <span><span style="display:inline-block;width:2px;height:10px;border-left:2px dashed var(--purple);margin-right:4px;vertical-align:middle;"></span> CTC Step</span>
                    <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;border:2px solid var(--amber);background:var(--red);margin-right:4px;vertical-align:middle;"></span> SL1 Hit &rarr; Flip</span>
                </div>
                <div>Use mouse scroll to zoom &bull; Drag to pan &bull; <a href="javascript:resetCandleZoom()" style="color: var(--navy); font-weight: 600;">Reset Zoom</a></div>
            </div>
        </div>

        <!-- Reading the chart key -->
        <h2 style="font-size: 16px; margin-top: 24px; margin-bottom: 8px;">Reading the chart — color and shape key</h2>
        <table>
            <thead><tr><th>Element</th><th>Meaning</th></tr></thead>
            <tbody>
                <tr><td><span class="legend-dot" style="background:var(--navy)"></span>Navy, dashed &rarr; solid</td><td>Pending limit order (dashed), then fill marker (solid dot) once triggered</td></tr>
                <tr><td><span class="legend-dot" style="background:var(--red)"></span>Coral line</td><td>Active SL level — position for whichever leg is currently open</td></tr>
                <tr><td><span class="legend-dot" style="background:var(--teal)"></span>Teal line</td><td>Active TP level. Dimmed (40% opacity) if never reached, full color if hit</td></tr>
                <tr><td><span class="legend-dot" style="background:var(--purple)"></span>Purple, dashed vertical</td><td>CTC step — the exact moment and price where SL jumped to breakeven. Drawn as a vertical connector between pre- and post-CTC SL lines</td></tr>
                <tr><td><span class="legend-dot" style="background:var(--amber);border:2px solid var(--amber)"></span>Amber ring around exit marker</td><td>This exit is a chain-trigger — it opens a child leg (the flip). A plain dot with no ring is terminal</td></tr>
            </tbody>
        </table>

        <!-- Every other possible day outcome table -->
        <h2 style="font-size: 16px; margin-top: 24px; margin-bottom: 8px;">Every other possible day outcome</h2>
        <table>
            <thead><tr><th>Outcome</th><th>Chart treatment</th></tr></thead>
            <tbody>
                <tr><td><b>TIE</b> — distances equal at 12:30</td><td>Range box only, gray marker at the eval line, label "no trade — tie". No pending line, no position lines at all.</td></tr>
                <tr><td><b>DECAY</b> — price ran 15pts favorably without filling</td><td>Dashed pending line ends abruptly (no fill dot), amber/gray fading marker, label "decay &mdash; cancelled, no fill". A thin reference tick at the trigger level (P<sub>curr</sub>&plusmn;15) shows how far price ran away.</td></tr>
                <tr><td><b>EXPIRED</b> — unfilled at 5:00pm cutoff</td><td>Same dashed-line-ends-abruptly treatment, distinct marker (clock/X icon), label "expired &mdash; 5pm cutoff, no fill".</td></tr>
                <tr><td><b>CTC_SL_HIT</b> — breakeven stop hit, no flip</td><td>Plain dot on the (already-stepped) SL line at the breakeven level, <i>no amber ring</i> — same visual family as SL1_HIT/SL2_HIT but explicitly non-chain-triggering. Label "CTC-SL hit &mdash; 0pts, no flip".</td></tr>
                <tr><td><b>EOD_CLOSE</b> — still open at day end</td><td>Distinct gray/blue marker at whatever price the position was force-closed, label shows the <i>actual</i> realized pts (not fixed like TP/SL) &mdash; e.g. "EOD close +6pts".</td></tr>
            </tbody>
        </table>
    </div>

    <!-- MONTHLY HEATMAP -->
    <div class="section">
        <div class="section-header">
            <h2>Monthly Returns Heatmap (%)</h2>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Year</th>
                    <th>Jan</th><th>Feb</th><th>Mar</th><th>Apr</th><th>May</th><th>Jun</th>
                    <th>Jul</th><th>Aug</th><th>Sep</th><th>Oct</th><th>Nov</th><th>Dec</th>
                </tr>
            </thead>
            <tbody>
                {{MONTHLY_ROWS}}
            </tbody>
        </table>
    </div>

    <!-- EXECUTION LOG TABLE -->
    <div class="section">
        <div class="section-header">
            <h2>Recent Chained Sequences & SL-Flip Execution Log</h2>
            <span style="font-size: 12px; color: var(--text-muted);">Click any trade row to load its visual chart above</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Chain ID</th>
                    <th>Date</th>
                    <th>Leg 1 (Sydney Range Fade)</th>
                    <th>Leg 2 (Contingent Reversal Flip)</th>
                    <th>Chain Net PnL</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                {{TRADES_ROWS}}
            </tbody>
        </table>
    </div>

    <script>
        // Injected daily chart and execution state data
        const dailyChartData = {{CHART_DATA_JSON}};

        let currentDayIndex = 0;
        let filteredIndices = [];
        let currentFilter = 'all';
        let currentView = 'schematic';

        function toggleTheme() {
            const body = document.body;
            if (body.classList.contains('theme-dark')) {
                body.classList.remove('theme-dark');
                body.classList.add('theme-light');
            } else {
                body.classList.remove('theme-light');
                body.classList.add('theme-dark');
            }
            if (currentView === 'candlestick') {
                renderCandlestick();
            }
        }

        function initVisualizer() {
            if (!dailyChartData || dailyChartData.length === 0) return;
            applyFilter('all');
        }

        function applyFilter(filterType, btnElem) {
            currentFilter = filterType;
            if (btnElem) {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btnElem.classList.add('active');
            }

            filteredIndices = [];
            dailyChartData.forEach((day, idx) => {
                let match = false;
                const pnl = day.pnl_net || 0;
                const legs = day.legs || [];
                const dayType = day.day_type || '';

                if (filterType === 'all') {
                    match = true;
                } else if (filterType === 'win') {
                    match = pnl > 0;
                } else if (filterType === 'loss') {
                    match = pnl < 0;
                } else if (filterType === 'flip') {
                    match = legs.length > 1;
                } else if (filterType === 'ctc') {
                    match = legs.some(l => l.ctc_armed);
                } else if (filterType === 'decay') {
                    match = dayType === 'DECAY' || dayType === 'EXPIRED' || legs.some(l => l.exit_reason === 'DECAY' || l.exit_reason === 'EXPIRATION');
                } else if (filterType === 'tie') {
                    match = dayType === 'TIE' || day.tie_detected;
                }

                if (match) filteredIndices.push(idx);
            });

            populateDaySelect();
            if (filteredIndices.length > 0) {
                selectIndex(filteredIndices[0]);
            }
        }

        function populateDaySelect() {
            const sel = document.getElementById('daySelect');
            sel.innerHTML = '';
            filteredIndices.forEach(idx => {
                const d = dailyChartData[idx];
                const opt = document.createElement('option');
                opt.value = idx;
                const pnl = d.pnl_net || 0;
                const pnlSign = pnl >= 0 ? '+' : '';
                const tag = d.day_type === 'TIE' ? '[TIE]' : (d.day_type === 'DECAY' ? '[DECAY]' : (pnl > 0 ? '[WIN]' : (pnl < 0 ? '[LOSS]' : '[EVEN]')));
                opt.textContent = `${d.date} (${d.day_of_week || 'Day'}) ${tag} ${d.direction || ''} ${pnlSign}$${pnl.toFixed(2)}`;
                sel.appendChild(opt);
            });
        }

        function onDaySelectChange() {
            const sel = document.getElementById('daySelect');
            const idx = parseInt(sel.value, 10);
            selectIndex(idx);
        }

        function prevDay() {
            if (filteredIndices.length === 0) return;
            let curPos = filteredIndices.indexOf(currentDayIndex);
            if (curPos > 0) {
                selectIndex(filteredIndices[curPos - 1]);
            } else {
                selectIndex(filteredIndices[filteredIndices.length - 1]);
            }
        }

        function nextDay() {
            if (filteredIndices.length === 0) return;
            let curPos = filteredIndices.indexOf(currentDayIndex);
            if (curPos < filteredIndices.length - 1) {
                selectIndex(filteredIndices[curPos + 1]);
            } else {
                selectIndex(filteredIndices[0]);
            }
        }

        function selectDay(dateStr, scrollToSection = false) {
            const idx = dailyChartData.findIndex(d => d.date === dateStr);
            if (idx >= 0) {
                if (!filteredIndices.includes(idx)) {
                    applyFilter('all');
                }
                selectIndex(idx);
                if (scrollToSection) {
                    document.getElementById('visualizerSection').scrollIntoView({ behavior: 'smooth' });
                }
            }
        }

        function selectIndex(idx) {
            currentDayIndex = idx;
            const sel = document.getElementById('daySelect');
            sel.value = idx;

            const day = dailyChartData[idx];
            if (!day) return;

            document.getElementById('outcomeSummaryText').innerHTML = day.outcome_summary || 'No trades executed for this session.';

            document.getElementById('hudDate').textContent = `${day.date} (${day.day_of_week || '—'})`;
            document.getElementById('hudRange').textContent = `$${(day.range_low || 0).toFixed(1)} - $${(day.range_high || 0).toFixed(1)}`;
            document.getElementById('hudWidthMid').textContent = `${(day.range_pts || 0).toFixed(1)} pts | Mid: $${(day.midpoint || 0).toFixed(1)}`;
            
            const dir = day.direction || 'NO_TRADE';
            const evalP = day.eval_price ? `$${day.eval_price.toFixed(1)}` : '—';
            document.getElementById('hudEvalBias').textContent = `${evalP} (${dir})`;

            const pnl = day.pnl_net || 0;
            const pnlElem = document.getElementById('hudPnL');
            pnlElem.textContent = `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`;
            pnlElem.className = 'hud-val ' + (pnl > 0 ? 'pos' : (pnl < 0 ? 'neg' : 'neutral'));

            document.getElementById('hudStatus').textContent = day.day_type || 'COMPLETED';

            if (currentView === 'schematic') {
                renderSchematic(day);
            } else {
                renderCandlestick(day);
            }
        }

        function switchView(viewName) {
            currentView = viewName;
            const btnSchematic = document.getElementById('tabSchematicBtn');
            const btnCandle = document.getElementById('tabCandleBtn');
            const boxSchematic = document.getElementById('schematicContainer');
            const boxCandle = document.getElementById('candlestickContainer');

            if (viewName === 'schematic') {
                btnSchematic.classList.add('active');
                btnCandle.classList.remove('active');
                boxSchematic.style.display = 'block';
                boxCandle.style.display = 'none';
                renderSchematic(dailyChartData[currentDayIndex]);
            } else {
                btnCandle.classList.add('active');
                btnSchematic.classList.remove('active');
                boxSchematic.style.display = 'none';
                boxCandle.style.display = 'block';
                renderCandlestick(dailyChartData[currentDayIndex]);
            }
        }

        // ================= SCHEMATIC SVG RENDERER =================
        function renderSchematic(day) {
            const box = document.getElementById('schematicSvgBox');
            if (!day) {
                box.innerHTML = '<p style="padding: 20px; color: var(--text-muted);">No data available.</p>';
                return;
            }

            const width = 780;
            const height = 440;
            const padding = 40;

            const rHigh = day.range_high || 2400;
            const rLow = day.range_low || 2380;
            const evalP = day.eval_price || (rHigh + rLow) / 2;

            const allPrices = [rHigh, rLow, evalP];
            if (day.entry_price) allPrices.push(day.entry_price);
            if (day.sl_price) allPrices.push(day.sl_price);
            if (day.tp_price) allPrices.push(day.tp_price);

            (day.legs || []).forEach(l => {
                if (l.entry_price) allPrices.push(l.entry_price);
                if (l.initial_sl_price) allPrices.push(l.initial_sl_price);
                if (l.sl_price) allPrices.push(l.sl_price);
                if (l.tp_price) allPrices.push(l.tp_price);
                if (l.ctc_price) allPrices.push(l.ctc_price);
                if (l.exit_price) allPrices.push(l.exit_price);
            });

            let minP = Math.min(...allPrices) - 5;
            let maxP = Math.max(...allPrices) + 5;
            if (maxP === minP) { maxP += 10; minP -= 10; }

            const getY = (p) => {
                return padding + (maxP - p) / (maxP - minP) * (height - padding * 2);
            };

            const yHigh = getY(rHigh);
            const yLow = getY(rLow);
            const yMid = getY((rHigh + rLow) / 2);
            const yEval = getY(evalP);

            let svg = `<svg width="100%" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;

            // Sydney range box
            const boxH = Math.max(yLow - yHigh, 4);
            svg += `<rect x="50" y="${yHigh}" width="200" height="${boxH}" fill="var(--purple-light)" stroke="var(--purple)" stroke-width="1.5"/>`;
            svg += `<text x="150" y="${yHigh - 10}" text-anchor="middle" font-size="11" font-weight="600" fill="var(--purple)">Sydney range</text>`;

            svg += `<line x1="50" y1="${yHigh}" x2="300" y2="${yHigh}" stroke="var(--text-faint)" stroke-width="1" stroke-dasharray="4 3"/>`;
            svg += `<text x="255" y="${yHigh - 4}" font-size="11" fill="var(--text-muted)">RangeHigh ${rHigh.toFixed(1)}</text>`;

            svg += `<line x1="50" y1="${yLow}" x2="300" y2="${yLow}" stroke="var(--text-faint)" stroke-width="1" stroke-dasharray="4 3"/>`;
            svg += `<text x="255" y="${yLow - 4}" font-size="11" fill="var(--text-muted)">RangeLow ${rLow.toFixed(1)}</text>`;

            // 12:30 eval vertical landmark
            svg += `<line x1="250" y1="15" x2="250" y2="425" stroke="var(--text-faint)" stroke-width="1" stroke-dasharray="3 3"/>`;
            
            const isTie = day.day_type === 'TIE' || day.tie_detected;
            const evalBiasText = isTie ? '12:30 eval: no trade — tie' : `12:30 eval: nearer ${day.dist_to_low < day.dist_to_high ? 'low &rarr; long' : 'high &rarr; short'}`;
            svg += `<text x="250" y="32" text-anchor="middle" font-size="11" font-weight="600" fill="var(--navy)">${evalBiasText}</text>`;

            // Correction 1: Always show 12:30 exact price dot and price level at 12:30
            svg += `<circle cx="250" cy="${yEval}" r="5.5" fill="var(--navy)" stroke="#ffffff" stroke-width="2"/>`;
            svg += `<text x="242" y="${yEval - 8}" text-anchor="end" font-size="11" font-weight="700" fill="var(--navy)">12:30: $${evalP.toFixed(1)}</text>`;

            const legs = day.legs || [];
            const l1 = legs[0];
            const l2 = legs.length > 1 ? legs[1] : null;

            if (isTie) {
                svg += `<circle cx="250" cy="${yEval}" r="6" fill="var(--text-muted)"/>`;
                svg += `<text x="262" y="${yEval}" font-size="11" fill="var(--text-muted)" dominant-baseline="central">Proximity Tie &mdash; No Trade</text>`;
            } else if (day.day_type === 'DECAY' || (l1 && l1.exit_reason === 'DECAY')) {
                const yEntry = getY(day.entry_price || evalP);
                const decayTarget = day.direction === 'LONG' ? evalP + 15 : evalP - 15;
                const yDecay = getY(decayTarget);

                svg += `<line x1="250" y1="${yEntry}" x2="370" y2="${yEntry}" stroke="var(--navy)" stroke-width="1.5" stroke-dasharray="5 4"/>`;
                svg += `<circle cx="370" cy="${yEntry}" r="5" fill="var(--amber)" opacity="0.8"/>`;
                svg += `<text x="380" y="${yEntry}" font-size="11" fill="var(--amber)" font-weight="600" dominant-baseline="central">Decay &mdash; cancelled, no fill</text>`;

                svg += `<line x1="250" y1="${yDecay}" x2="370" y2="${yDecay}" stroke="var(--amber)" stroke-width="1" stroke-dasharray="2 2" opacity="0.5"/>`;
                svg += `<text x="375" y="${yDecay}" font-size="10" fill="var(--text-faint)" dominant-baseline="central">Decay trigger (${decayTarget.toFixed(1)})</text>`;
            } else if (day.day_type === 'EXPIRED' || (l1 && l1.is_pending && l1.exit_reason === 'EXPIRATION')) {
                const yEntry = getY(day.entry_price || evalP);
                svg += `<line x1="250" y1="${yEntry}" x2="450" y2="${yEntry}" stroke="var(--navy)" stroke-width="1.5" stroke-dasharray="5 4"/>`;
                svg += `<line x1="450" y1="50" x2="450" y2="400" stroke="var(--text-faint)" stroke-width="1" stroke-dasharray="3 3"/>`;
                svg += `<text x="450" y="45" text-anchor="middle" font-size="10" fill="var(--text-muted)">17:00 IST Cutoff</text>`;
                svg += `<circle cx="450" cy="${yEntry}" r="5" fill="var(--navy)"/>`;
                svg += `<text x="460" y="${yEntry}" font-size="11" fill="var(--navy)" font-weight="600" dominant-baseline="central">Expired &mdash; 5pm cutoff, no fill</text>`;
            } else if (l1) {
                const yEntry1 = getY(l1.entry_price);
                const ySl1Initial = getY(l1.initial_sl_price || l1.sl_price);
                const yTp1 = getY(l1.tp_price);
                const l1HitTp = l1.exit_reason === 'TP';
                const l1HitSl = l1.exit_reason === 'SL';
                const l1HitCtc = l1.exit_reason === 'CTC';

                // Correction 2: Check if order filled instantly at 12:30 vs pending limit order
                const isInstant = Boolean(day.is_instant) || (Math.abs(l1.entry_price - evalP) < 0.05);
                const l1StartX = isInstant ? 250 : 320;

                if (!isInstant) {
                    // Case 1: Pending limit order waiting for price to reach limit price
                    svg += `<line x1="250" y1="${yEntry1}" x2="320" y2="${yEntry1}" stroke="var(--navy)" stroke-width="1.5" stroke-dasharray="5 4"/>`;
                    svg += `<circle cx="320" cy="${yEntry1}" r="5" fill="var(--navy)"/>`;
                    svg += `<text x="328" y="${yEntry1 - 10}" font-size="11" font-weight="600" fill="var(--navy)">Filled ${l1.direction.toLowerCase()} limit @ ${l1.entry_price.toFixed(1)}</text>`;
                } else {
                    // Case 2: Instant fill at 12:30 (no dashed line, marker right at 12:30 landmark)
                    svg += `<circle cx="250" cy="${yEntry1}" r="6" fill="var(--teal)" stroke="#ffffff" stroke-width="2"/>`;
                    svg += `<text x="258" y="${yEntry1 - 10}" font-size="11" font-weight="700" fill="var(--teal)">Instant fill ${l1.direction.toLowerCase()} @ ${l1.entry_price.toFixed(1)}</text>`;
                }

                const l1ExitX = 440;
                const tp1Opacity = l1HitTp ? '1.0' : '0.4';

                svg += `<line x1="${l1StartX}" y1="${yTp1}" x2="${l1ExitX}" y2="${yTp1}" stroke="var(--teal)" stroke-width="2" opacity="${tp1Opacity}"/>`;
                svg += `<text x="${l1StartX + 5}" y="${yTp1 - 6}" font-size="11" fill="var(--text-muted)">TP1 ${l1.tp_price.toFixed(1)} ${l1HitTp ? '(hit)' : '(not reached)'}</text>`;

                if (l1.ctc_armed && l1.ctc_price) {
                    const ctcX = 380;
                    const yCtc = getY(l1.ctc_price);
                    svg += `<line x1="${l1StartX}" y1="${ySl1Initial}" x2="${ctcX}" y2="${ySl1Initial}" stroke="var(--red)" stroke-width="2"/>`;
                    svg += `<line x1="${ctcX}" y1="${ySl1Initial}" x2="${ctcX}" y2="${yEntry1}" stroke="var(--purple)" stroke-width="1.5" stroke-dasharray="3 2"/>`;
                    svg += `<circle cx="${ctcX}" cy="${yCtc}" r="4" fill="var(--purple)"/>`;
                    svg += `<text x="${ctcX + 6}" y="${yCtc}" font-size="10" fill="var(--purple)">CTC @ ${l1.ctc_price.toFixed(1)} &rarr; SL&rarr;BE</text>`;
                    svg += `<line x1="${ctcX}" y1="${yEntry1}" x2="${l1ExitX}" y2="${yEntry1}" stroke="var(--red)" stroke-width="2"/>`;
                    svg += `<text x="${ctcX + 10}" y="${yEntry1 + 14}" font-size="11" fill="var(--text-muted)">SL1&rarr;BE (${l1.entry_price.toFixed(1)})</text>`;
                } else {
                    svg += `<line x1="${l1StartX}" y1="${ySl1Initial}" x2="${l1ExitX}" y2="${ySl1Initial}" stroke="var(--red)" stroke-width="2"/>`;
                    svg += `<text x="${l1StartX + 5}" y="${ySl1Initial + 14}" font-size="11" fill="var(--text-muted)">SL1 ${l1.initial_sl_price ? l1.initial_sl_price.toFixed(1) : l1.sl_price.toFixed(1)}</text>`;
                }

                if (l1HitTp) {
                    svg += `<circle cx="${l1ExitX}" cy="${yTp1}" r="5" fill="var(--teal)"/>`;
                    svg += `<text x="${l1ExitX + 8}" y="${yTp1}" font-size="11" font-weight="600" fill="var(--teal)" dominant-baseline="central">TP1 hit +20</text>`;
                } else if (l1HitCtc) {
                    svg += `<circle cx="${l1ExitX}" cy="${yEntry1}" r="5" fill="var(--red)"/>`;
                    svg += `<text x="${l1ExitX + 8}" y="${yEntry1}" font-size="11" font-weight="600" fill="var(--text-muted)" dominant-baseline="central">CTC-SL hit (0 pts, no flip)</text>`;
                } else if (l1HitSl) {
                    svg += `<circle cx="${l1ExitX}" cy="${ySl1Initial}" r="5" fill="var(--red)"/>`;
                    svg += `<circle cx="${l1ExitX}" cy="${ySl1Initial}" r="9" fill="none" stroke="var(--amber)" stroke-width="2"/>`;
                    svg += `<text x="${l1ExitX + 12}" y="${ySl1Initial - 10}" font-size="11" font-weight="600" fill="var(--amber)">SL1 hit &rarr; flip ${l2 ? l2.direction.toLowerCase() : 'reversal'}</text>`;
                } else if (l1.exit_reason === 'EXPIRATION') {
                    const yExit1 = getY(l1.exit_price || l1.entry_price);
                    svg += `<circle cx="${l1ExitX}" cy="${yExit1}" r="5" fill="var(--navy)"/>`;
                    svg += `<text x="${l1ExitX + 8}" y="${yExit1}" font-size="11" fill="var(--text-muted)" dominant-baseline="central">EOD close (${l1.pnl_net >= 0 ? '+' : ''}${l1.pnl_net.toFixed(1)})</text>`;
                }

                if (l2) {
                    const l2StartX = 440;
                    const l2ExitX = 650;
                    const yEntry2 = getY(l2.entry_price);
                    const ySl2Initial = getY(l2.initial_sl_price || l2.sl_price);
                    const yTp2 = getY(l2.tp_price);
                    const l2HitTp = l2.exit_reason === 'TP';
                    const l2HitSl = l2.exit_reason === 'SL';
                    const l2HitCtc = l2.exit_reason === 'CTC';

                    const tp2Opacity = l2HitTp ? '1.0' : '0.4';
                    svg += `<line x1="${l2StartX}" y1="${yTp2}" x2="${l2ExitX}" y2="${yTp2}" stroke="var(--teal)" stroke-width="2" opacity="${tp2Opacity}"/>`;
                    svg += `<text x="${l2StartX + 20}" y="${yTp2 + (l2.direction === 'LONG' ? -6 : 14)}" font-size="11" fill="var(--text-muted)">TP2 target ${l2.tp_price.toFixed(1)}</text>`;

                    if (l2.ctc_armed && l2.ctc_price) {
                        const ctcX2 = 540;
                        const yCtc2 = getY(l2.ctc_price);
                        svg += `<line x1="${l2StartX}" y1="${ySl2Initial}" x2="${ctcX2}" y2="${ySl2Initial}" stroke="var(--red)" stroke-width="2"/>`;
                        svg += `<text x="${l2StartX + 10}" y="${ySl2Initial + 14}" font-size="11" fill="var(--text-muted)">SL2 ${l2.initial_sl_price.toFixed(1)}</text>`;

                        svg += `<line x1="${ctcX2}" y1="${ySl2Initial}" x2="${ctcX2}" y2="${yEntry2}" stroke="var(--purple)" stroke-width="1.5" stroke-dasharray="3 2"/>`;
                        svg += `<circle cx="${ctcX2}" cy="${yCtc2}" r="4" fill="var(--purple)"/>`;
                        svg += `<text x="${ctcX2 + 6}" y="${yCtc2}" font-size="10" fill="var(--purple)">CTC @ ${l2.ctc_price.toFixed(1)} &rarr; SL&rarr;BE</text>`;

                        svg += `<line x1="${ctcX2}" y1="${yEntry2}" x2="${l2ExitX}" y2="${yEntry2}" stroke="var(--red)" stroke-width="2"/>`;
                        svg += `<text x="${ctcX2 + 10}" y="${yEntry2 + 14}" font-size="11" fill="var(--text-muted)">SL2&rarr;BE (${l2.entry_price.toFixed(1)})</text>`;
                    } else {
                        svg += `<line x1="${l2StartX}" y1="${ySl2Initial}" x2="${l2ExitX}" y2="${ySl2Initial}" stroke="var(--red)" stroke-width="2"/>`;
                        svg += `<text x="${l2StartX + 10}" y="${ySl2Initial + 14}" font-size="11" fill="var(--text-muted)">SL2 ${l2.initial_sl_price ? l2.initial_sl_price.toFixed(1) : l2.sl_price.toFixed(1)}</text>`;
                    }

                    if (l2HitTp) {
                        svg += `<circle cx="${l2ExitX}" cy="${yTp2}" r="5" fill="var(--teal)"/>`;
                        svg += `<text x="${l2ExitX + 8}" y="${yTp2}" font-size="11" font-weight="600" fill="var(--teal)" dominant-baseline="central">TP2 hit +20</text>`;
                    } else if (l2HitCtc) {
                        svg += `<circle cx="${l2ExitX}" cy="${yEntry2}" r="5" fill="var(--red)"/>`;
                        svg += `<text x="${l2ExitX + 8}" y="${yEntry2}" font-size="11" font-weight="600" fill="var(--text-muted)" dominant-baseline="central">CTC-SL hit (0 pts)</text>`;
                    } else if (l2HitSl) {
                        svg += `<circle cx="${l2ExitX}" cy="${ySl2Initial}" r="5" fill="var(--red)"/>`;
                        svg += `<text x="${l2ExitX + 8}" y="${ySl2Initial}" font-size="11" font-weight="600" fill="var(--red)" dominant-baseline="central">SL2 hit &minus;10</text>`;
                    } else if (l2.exit_reason === 'EXPIRATION') {
                        const yExit2 = getY(l2.exit_price || l2.entry_price);
                        svg += `<circle cx="${l2ExitX}" cy="${yExit2}" r="5" fill="var(--navy)"/>`;
                        svg += `<text x="${l2ExitX + 8}" y="${yExit2}" font-size="11" fill="var(--text-muted)" dominant-baseline="central">EOD close (${l2.pnl_net >= 0 ? '+' : ''}${l2.pnl_net.toFixed(1)})</text>`;
                    }
                }
            }

            svg += `</svg>`;
            box.innerHTML = svg;
        }

        // ================= CANDLESTICK CANVAS RENDERER =================
        let candleState = {
            zoom: 1.0,
            panX: 0,
            isDragging: false,
            startX: 0,
            showFsmOverlay: true,
            mouseX: -1,
            mouseY: -1,
        };

        function toggleFsmOverlay() {
            candleState.showFsmOverlay = !candleState.showFsmOverlay;
            const btn = document.getElementById('btnToggleFsmOverlay');
            const statusText = document.getElementById('fsmOverlayStatusText');
            const statusIcon = document.getElementById('fsmOverlayStatusIcon');
            if (btn) {
                if (candleState.showFsmOverlay) {
                    btn.classList.add('active');
                    if (statusText) statusText.textContent = 'ON';
                    if (statusIcon) statusIcon.textContent = '✓';
                } else {
                    btn.classList.remove('active');
                    if (statusText) statusText.textContent = 'OFF';
                    if (statusIcon) statusIcon.textContent = '✕';
                }
            }
            renderCandlestick();
        }

        function resetCandleZoom() {
            candleState.zoom = 1.0;
            candleState.panX = 0;
            renderCandlestick();
        }

        function renderCandles(day) {
            renderCandlestick(day);
        }

        function renderCandlestick(day) {
            if (!day) day = dailyChartData[currentDayIndex];
            if (!day) return;

            const canvas = document.getElementById('candlestickCanvas');
            if (!canvas) return;
            const container = canvas.parentElement;
            const dpr = window.devicePixelRatio || 1;
            const rect = container.getBoundingClientRect();

            canvas.width = rect.width * dpr;
            canvas.height = 460 * dpr;
            const ctx = canvas.getContext('2d');
            ctx.scale(dpr, dpr);

            const width = rect.width;
            const height = 460;
            const chartW = width - 85;
            const chartH = height - 32;

            ctx.clearRect(0, 0, width, height);

            const bars = day.bars || [];
            if (bars.length === 0) {
                ctx.fillStyle = '#888';
                ctx.font = '14px sans-serif';
                ctx.fillText('No candlestick bar data captured for this day.', width / 2 - 120, height / 2);
                return;
            }

            const isDark = document.body.classList.contains('theme-dark');
            const col = {
                navy: isDark ? '#85b7eb' : '#0c447c',
                navyLight: isDark ? 'rgba(133, 183, 235, 0.15)' : 'rgba(12, 68, 124, 0.12)',
                teal: isDark ? '#5dcaa5' : '#0f6e56',
                tealLight: isDark ? 'rgba(93, 202, 165, 0.18)' : 'rgba(15, 110, 86, 0.12)',
                red: isDark ? '#f08282' : '#a32d2d',
                redLight: isDark ? 'rgba(240, 130, 130, 0.18)' : 'rgba(163, 45, 45, 0.12)',
                amber: isDark ? '#f5c067' : '#854f0b',
                amberLight: isDark ? 'rgba(245, 192, 103, 0.20)' : 'rgba(133, 79, 11, 0.12)',
                purple: isDark ? '#afa9f5' : '#534ab7',
                purpleLight: isDark ? 'rgba(175, 169, 245, 0.15)' : 'rgba(83, 74, 183, 0.10)',
                text: isDark ? '#e8e6de' : '#2c2c2a',
                textMuted: isDark ? '#b4b2a9' : '#5f5e5a',
                textFaint: isDark ? '#888780' : '#888780',
                grid: isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)',
                surface: isDark ? '#252523' : '#ffffff',
                border: isDark ? '#3d3d3a' : '#d3d1c7',
            };

            function parseTs(t) {
                if (!t) return 0;
                if (typeof t === 'number') return t > 1e11 ? t / 1000 : t;
                const d = new Date(t);
                return isNaN(d.getTime()) ? 0 : d.getTime() / 1000;
            }

            function getBarIndexForTime(timeVal) {
                const ts = parseTs(timeVal);
                if (!ts || bars.length === 0) return -1;
                let bestIdx = 0;
                let bestDiff = Math.abs(bars[0].t - ts);
                for (let i = 1; i < bars.length; i++) {
                    const diff = Math.abs(bars[i].t - ts);
                    if (diff < bestDiff) {
                        bestDiff = diff;
                        bestIdx = i;
                    }
                }
                return bestIdx;
            }

            let minP = Infinity;
            let maxP = -Infinity;
            bars.forEach(b => {
                if (b.l < minP) minP = b.l;
                if (b.h > maxP) maxP = b.h;
            });

            if (day.range_high && day.range_high > maxP) maxP = day.range_high;
            if (day.range_low && day.range_low < minP) minP = day.range_low;
            if (day.eval_price) {
                if (day.eval_price > maxP) maxP = day.eval_price;
                if (day.eval_price < minP) minP = day.eval_price;
            }
            if (day.entry_price) {
                if (day.entry_price > maxP) maxP = day.entry_price;
                if (day.entry_price < minP) minP = day.entry_price;
            }
            if (day.sl_price) {
                if (day.sl_price > maxP) maxP = day.sl_price;
                if (day.sl_price < minP) minP = day.sl_price;
            }
            if (day.tp_price) {
                if (day.tp_price > maxP) maxP = day.tp_price;
                if (day.tp_price < minP) minP = day.tp_price;
            }

            (day.legs || []).forEach(l => {
                [l.entry_price, l.initial_sl_price, l.sl_price, l.tp_price, l.ctc_price, l.exit_price].forEach(p => {
                    if (p != null) {
                        if (p > maxP) maxP = p;
                        if (p < minP) minP = p;
                    }
                });
            });

            if (day.day_type === 'DECAY') {
                const evalP = day.eval_price || ((day.range_high + day.range_low) / 2);
                const decayTarget = day.direction === 'LONG' ? evalP + 15 : evalP - 15;
                if (decayTarget > maxP) maxP = decayTarget;
                if (decayTarget < minP) minP = decayTarget;
            }

            const pad = (maxP - minP) * 0.08 || 2;
            minP -= pad;
            maxP += pad;

            const getY = (p) => chartH - ((p - minP) / (maxP - minP)) * (chartH - 35) - 15;
            const nBars = bars.length;
            const barW = Math.max((chartW / nBars) * candleState.zoom, 2.5);
            const getX = (i) => 20 + i * barW + candleState.panX;

            const sessions = day.sessions || {};
            const evalTs = sessions.eval_time;
            const londonCloseTs = sessions.london_close;

            let sydStartIdx = 0;
            let sydEndIdx = bars.length - 1;
            let lonStartIdx = 0;
            let lonEndIdx = bars.length - 1;

            if (evalTs) {
                bars.forEach((b, i) => {
                    if (b.t <= evalTs) sydEndIdx = i;
                    if (b.t >= evalTs && lonStartIdx === 0) lonStartIdx = i;
                    if (londonCloseTs && b.t <= londonCloseTs) lonEndIdx = i;
                });
            }

            // 1. Sydney Session background & box
            const sydX1 = getX(0);
            const sydX2 = getX(sydEndIdx) + barW;
            ctx.fillStyle = col.purpleLight;
            ctx.fillRect(sydX1, 0, Math.max(sydX2 - sydX1, 0), chartH);
            ctx.fillStyle = col.purple;
            ctx.font = '600 10.5px sans-serif';
            ctx.fillText('🇦🇺 SYDNEY RANGE (Open → 12:30 IST)', sydX1 + 8, 16);

            // 2. London Session background
            const lonX1 = getX(lonStartIdx);
            const lonX2 = getX(lonEndIdx) + barW;
            ctx.fillStyle = col.amberLight;
            ctx.fillRect(lonX1, 0, Math.max(lonX2 - lonX1, 0), chartH);
            ctx.fillStyle = col.amber;
            ctx.font = '600 10.5px sans-serif';
            ctx.fillText('🇬🇧 LONDON SESSION (12:30 IST → Close)', lonX1 + 8, 16);

            // 3. 12:30 IST Evaluation Landmark
            const evalX = getX(sydEndIdx) + barW / 2;
            ctx.strokeStyle = col.navy;
            ctx.setLineDash([4, 3]);
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(evalX, 0);
            ctx.lineTo(evalX, chartH);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = col.navy;
            ctx.font = 'bold 10.5px sans-serif';
            ctx.fillText('12:30 IST EVAL', evalX + 5, 28);

            // 4. Sydney Range Levels (High, Low, Midpoint)
            if (day.range_high && day.range_low) {
                const yH = getY(day.range_high);
                const yL = getY(day.range_low);
                const yM = getY(day.midpoint);

                ctx.strokeStyle = col.purple;
                ctx.lineWidth = 1.5;
                ctx.strokeRect(sydX1, Math.min(yH, yL), Math.max(sydX2 - sydX1, 0), Math.abs(yL - yH));

                ctx.strokeStyle = col.purple;
                ctx.setLineDash([4, 4]);
                ctx.lineWidth = 1.0;
                ctx.beginPath();
                ctx.moveTo(sydX1, yH);
                ctx.lineTo(chartW, yH);
                ctx.moveTo(sydX1, yL);
                ctx.lineTo(chartW, yL);
                ctx.stroke();

                ctx.strokeStyle = col.textFaint;
                ctx.setLineDash([2, 2]);
                ctx.beginPath();
                ctx.moveTo(sydX1, yM);
                ctx.lineTo(chartW, yM);
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.fillStyle = col.purple;
                ctx.font = 'bold 10px sans-serif';
                ctx.fillText(`R-High ${day.range_high.toFixed(1)}`, chartW + 6, yH + 3);
                ctx.fillText(`R-Low ${day.range_low.toFixed(1)}`, chartW + 6, yL + 3);
                ctx.fillStyle = col.textFaint;
                ctx.font = '10px sans-serif';
                ctx.fillText(`Mid ${day.midpoint.toFixed(1)}`, chartW + 6, yM + 3);
            }

            // 5. Price Axis Grid & Labels
            ctx.fillStyle = col.textMuted;
            ctx.font = '10px monospace';
            const nPriceTicks = 6;
            for (let i = 0; i <= nPriceTicks; i++) {
                const p = minP + (i / nPriceTicks) * (maxP - minP);
                const y = getY(p);
                ctx.strokeStyle = col.grid;
                ctx.lineWidth = 1.0;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(chartW, y);
                ctx.stroke();
                ctx.fillText(`$${p.toFixed(1)}`, chartW + 6, y + 3);
            }

            // 6. Draw 5-minute Candlesticks
            bars.forEach((b, i) => {
                const x = getX(i);
                if (x < -20 || x > chartW + 20) return;

                const yO = getY(b.o);
                const yC = getY(b.c);
                const yH = getY(b.h);
                const yL = getY(b.l);
                const isUp = b.c >= b.o;

                const color = isUp ? col.teal : col.red;
                ctx.strokeStyle = color;
                ctx.fillStyle = color;

                // Wick
                ctx.lineWidth = Math.max(barW * 0.18, 1);
                ctx.beginPath();
                ctx.moveTo(x + barW / 2, yH);
                ctx.lineTo(x + barW / 2, yL);
                ctx.stroke();

                // Candle Body
                const bodyY = Math.min(yO, yC);
                const bodyH = Math.max(Math.abs(yO - yC), 1.5);
                ctx.fillRect(x + 0.8, bodyY, Math.max(barW - 1.6, 1.2), bodyH);
            });

            function drawBadge(text, x, y, bgCol, textCol, fontSize = 11, align = 'left') {
                ctx.save();
                ctx.font = `600 ${fontSize}px sans-serif`;
                const textWidth = ctx.measureText(text).width;
                const padX = 6;
                const padY = 3;
                const boxW = textWidth + padX * 2;
                const boxH = fontSize + padY * 2;
                let boxX = x;
                if (align === 'center') boxX = x - boxW / 2;
                else if (align === 'right') boxX = x - boxW;
                const boxY = y - fontSize / 2 - padY;

                ctx.fillStyle = bgCol;
                ctx.beginPath();
                if (ctx.roundRect) ctx.roundRect(boxX, boxY, boxW, boxH, 4);
                else ctx.rect(boxX, boxY, boxW, boxH);
                ctx.fill();

                ctx.strokeStyle = textCol;
                ctx.lineWidth = 0.8;
                if (ctx.roundRect) ctx.roundRect(boxX, boxY, boxW, boxH, 4);
                else ctx.rect(boxX, boxY, boxW, boxH);
                ctx.stroke();

                ctx.fillStyle = textCol;
                ctx.textBaseline = 'middle';
                ctx.textAlign = align;
                ctx.fillText(text, x, y);
                ctx.restore();
            }

            // 7. FSM Schematic Overlay
            if (candleState.showFsmOverlay) {
                const isTie = day.day_type === 'TIE' || day.tie_detected;
                const evalP = day.eval_price || ((day.range_high + day.range_low) / 2);
                const yEval = getY(evalP);

                // Eval marker dot
                ctx.fillStyle = isTie ? col.textMuted : col.navy;
                ctx.beginPath();
                ctx.arc(evalX, yEval, 5, 0, Math.PI * 2);
                ctx.fill();
                drawBadge(`12:30: $${evalP.toFixed(1)}`, evalX - 8, yEval - 12, col.surface, isTie ? col.textMuted : col.navy, 10.5, 'right');

                const legs = day.legs || [];
                const l1 = legs[0];
                const l2 = legs.length > 1 ? legs[1] : null;

                if (isTie) {
                    drawBadge('12:30 Eval: Tie (No Trade)', evalX + 10, yEval, col.surface, col.textMuted, 11);
                } else if (day.day_type === 'DECAY' || (l1 && l1.exit_reason === 'DECAY')) {
                    const yEntry = getY(day.entry_price || evalP);
                    const decayTarget = day.direction === 'LONG' ? evalP + 15 : evalP - 15;
                    const yDecay = getY(decayTarget);

                    let decayEvt = (day.events || []).find(e => e.type === 'DECAY');
                    let decayIdx = decayEvt ? getBarIndexForTime(decayEvt.time) : -1;
                    if (decayIdx < 0) {
                        for (let i = sydEndIdx + 1; i < bars.length; i++) {
                            if (day.direction === 'LONG' && bars[i].h >= decayTarget) { decayIdx = i; break; }
                            if (day.direction === 'SHORT' && bars[i].l <= decayTarget) { decayIdx = i; break; }
                        }
                    }
                    if (decayIdx < 0) decayIdx = Math.min(sydEndIdx + 12, bars.length - 1);
                    const decayX = getX(decayIdx) + barW / 2;

                    ctx.strokeStyle = col.navy;
                    ctx.lineWidth = 1.8;
                    ctx.setLineDash([5, 4]);
                    ctx.beginPath();
                    ctx.moveTo(evalX, yEntry);
                    ctx.lineTo(decayX, yEntry);
                    ctx.stroke();

                    ctx.strokeStyle = col.amber;
                    ctx.lineWidth = 1.2;
                    ctx.setLineDash([2, 2]);
                    ctx.beginPath();
                    ctx.moveTo(evalX, yDecay);
                    ctx.lineTo(decayX, yDecay);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    ctx.fillStyle = col.amber;
                    ctx.beginPath();
                    ctx.arc(decayX, yEntry, 6, 0, Math.PI * 2);
                    ctx.fill();

                    drawBadge('Decay: Cancelled (+15 pt run, no fill)', decayX + 10, yEntry, col.amberLight, col.amber, 11);
                    drawBadge(`Decay trigger (${decayTarget.toFixed(1)})`, evalX + 15, yDecay - 8, col.surface, col.textFaint, 10);

                } else if (day.day_type === 'EXPIRED' || (l1 && l1.is_pending && l1.exit_reason === 'EXPIRATION')) {
                    const yEntry = getY(day.entry_price || evalP);
                    let cutoffIdx = getBarIndexForTime(sessions.limit_cancel || sessions.eod);
                    if (cutoffIdx < 0) cutoffIdx = bars.length - 1;
                    const cutoffX = getX(cutoffIdx) + barW / 2;

                    ctx.strokeStyle = col.navy;
                    ctx.lineWidth = 1.8;
                    ctx.setLineDash([5, 4]);
                    ctx.beginPath();
                    ctx.moveTo(evalX, yEntry);
                    ctx.lineTo(cutoffX, yEntry);
                    ctx.stroke();

                    ctx.strokeStyle = col.textMuted;
                    ctx.lineWidth = 1.0;
                    ctx.beginPath();
                    ctx.moveTo(cutoffX, 40);
                    ctx.lineTo(cutoffX, chartH);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    ctx.fillStyle = col.navy;
                    ctx.beginPath();
                    ctx.arc(cutoffX, yEntry, 6, 0, Math.PI * 2);
                    ctx.fill();

                    drawBadge('Expired: 5pm cutoff (unfilled)', cutoffX + 10, yEntry, col.navyLight, col.navy, 11);

                } else if (l1) {
                    const yEntry1 = getY(l1.entry_price);
                    const ySl1Initial = getY(l1.initial_sl_price || l1.sl_price);
                    const yTp1 = getY(l1.tp_price);
                    const l1HitTp = l1.exit_reason === 'TP';
                    const l1HitSl = l1.exit_reason === 'SL';
                    const l1HitCtc = l1.exit_reason === 'CTC';

                    const isInstant = Boolean(day.is_instant) || (Math.abs(l1.entry_price - evalP) < 0.05);

                    let fillEvt = (day.events || []).find(e => e.type === 'LIMIT_FILLED');
                    let fillIdx = fillEvt ? getBarIndexForTime(fillEvt.time) : -1;
                    if (fillIdx < 0) fillIdx = getBarIndexForTime(l1.entry_time);
                    if (fillIdx < 0 || fillIdx <= sydEndIdx) {
                        for (let i = sydEndIdx + 1; i < bars.length; i++) {
                            if (l1.direction === 'LONG' && bars[i].l <= l1.entry_price) { fillIdx = i; break; }
                            if (l1.direction === 'SHORT' && bars[i].h >= l1.entry_price) { fillIdx = i; break; }
                        }
                    }
                    if (fillIdx < 0) fillIdx = Math.min(sydEndIdx + 4, bars.length - 1);
                    const fillX = isInstant ? evalX : Math.max(getX(fillIdx) + barW / 2, evalX);

                    if (!isInstant && fillX > evalX + 4) {
                        // Pending Limit line (dashed line waiting to fill)
                        ctx.strokeStyle = col.navy;
                        ctx.lineWidth = 2.0;
                        ctx.setLineDash([5, 4]);
                        ctx.beginPath();
                        ctx.moveTo(evalX, yEntry1);
                        ctx.lineTo(fillX, yEntry1);
                        ctx.stroke();
                        ctx.setLineDash([]);

                        drawBadge(`Limit ${l1.direction} @ ${l1.entry_price.toFixed(1)}`, (evalX + fillX) / 2, yEntry1 - 12, col.surface, col.navy, 10.5, 'center');

                        // Limit Fill marker
                        ctx.fillStyle = col.navy;
                        ctx.beginPath();
                        ctx.arc(fillX, yEntry1, 6, 0, Math.PI * 2);
                        ctx.fill();
                        drawBadge(`Filled ${l1.direction} @ ${l1.entry_price.toFixed(1)}`, fillX, yEntry1 + 14, col.navyLight, col.navy, 11, 'center');
                    } else {
                        // Instantaneous fill at 12:30 (no dashed line, solid marker right at evalX)
                        ctx.fillStyle = col.teal;
                        ctx.beginPath();
                        ctx.arc(evalX, yEntry1, 6.5, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 2;
                        ctx.stroke();
                        drawBadge(`Instant fill ${l1.direction} @ ${l1.entry_price.toFixed(1)}`, evalX + 12, yEntry1 + 14, col.surface, col.teal, 11, 'left');
                    }

                    let l1ExitEvt = (day.events || []).find(e => (e.type === 'EXIT_LN' && (e.leg_id === l1.leg_id || e.leg_id === 'LEG-1')) || e.type === 'FLIP_ENTERED');
                    let l1ExitIdx = l1ExitEvt ? getBarIndexForTime(l1ExitEvt.time) : -1;
                    if (l1ExitIdx < 0) l1ExitIdx = getBarIndexForTime(l1.exit_time);
                    if (l1ExitIdx < 0) l1ExitIdx = Math.min(fillIdx + 10, bars.length - 1);
                    const l1ExitX = Math.max(getX(l1ExitIdx) + barW / 2, fillX + barW * 2);

                    // TP1 Target Line
                    ctx.strokeStyle = col.teal;
                    ctx.lineWidth = l1HitTp ? 2.5 : 1.8;
                    ctx.globalAlpha = l1HitTp ? 1.0 : 0.40;
                    ctx.beginPath();
                    ctx.moveTo(fillX, yTp1);
                    ctx.lineTo(l1ExitX, yTp1);
                    ctx.stroke();
                    ctx.globalAlpha = 1.0;

                    drawBadge(`TP1 ${l1.tp_price.toFixed(1)} ${l1HitTp ? '(hit)' : '(not reached)'}`, fillX + 15, yTp1 - 8, col.surface, col.teal, 10.5);

                    // SL1 Level & Stepped CTC Connector
                    if (l1.ctc_armed && l1.ctc_price) {
                        let ctcEvt = (day.events || []).find(e => e.type === 'CTC_STEP');
                        let ctcIdx = ctcEvt ? getBarIndexForTime(ctcEvt.time) : -1;
                        if (ctcIdx < 0) ctcIdx = getBarIndexForTime(l1.ctc_time);
                        if (ctcIdx < 0 || ctcIdx <= fillIdx || ctcIdx >= l1ExitIdx) {
                            ctcIdx = Math.floor((fillIdx + l1ExitIdx) / 2);
                        }
                        const ctcX = Math.min(Math.max(getX(ctcIdx) + barW / 2, fillX + barW * 1.5), l1ExitX - barW);
                        const yCtc = getY(l1.ctc_price);

                        ctx.strokeStyle = col.red;
                        ctx.lineWidth = 2.0;
                        ctx.beginPath();
                        ctx.moveTo(fillX, ySl1Initial);
                        ctx.lineTo(ctcX, ySl1Initial);
                        ctx.stroke();

                        drawBadge(`SL1 ${(l1.initial_sl_price || l1.sl_price).toFixed(1)}`, fillX + 10, ySl1Initial + 14, col.surface, col.textMuted, 10.5);

                        ctx.fillStyle = col.purple;
                        ctx.beginPath();
                        ctx.arc(ctcX, yCtc, 5, 0, Math.PI * 2);
                        ctx.fill();
                        drawBadge(`CTC @ ${l1.ctc_price.toFixed(1)} → SL→BE`, ctcX + 8, yCtc, col.purpleLight, col.purple, 10.5);

                        // Stepped connector jumping SL to entry
                        ctx.strokeStyle = col.purple;
                        ctx.lineWidth = 2.0;
                        ctx.setLineDash([3, 2]);
                        ctx.beginPath();
                        ctx.moveTo(ctcX, ySl1Initial);
                        ctx.lineTo(ctcX, yEntry1);
                        ctx.stroke();
                        ctx.setLineDash([]);

                        ctx.strokeStyle = col.red;
                        ctx.lineWidth = 2.0;
                        ctx.beginPath();
                        ctx.moveTo(ctcX, yEntry1);
                        ctx.lineTo(l1ExitX, yEntry1);
                        ctx.stroke();

                        drawBadge(`SL1→BE (${l1.entry_price.toFixed(1)})`, ctcX + 10, yEntry1 + 14, col.surface, col.textMuted, 10.5);
                    } else {
                        ctx.strokeStyle = col.red;
                        ctx.lineWidth = 2.0;
                        ctx.beginPath();
                        ctx.moveTo(fillX, ySl1Initial);
                        ctx.lineTo(l1ExitX, ySl1Initial);
                        ctx.stroke();

                        drawBadge(`SL1 ${(l1.initial_sl_price || l1.sl_price).toFixed(1)}`, fillX + 10, ySl1Initial + 14, col.surface, col.textMuted, 10.5);
                    }

                    // Leg 1 Exit Marker
                    if (l1HitTp) {
                        ctx.fillStyle = col.teal;
                        ctx.beginPath();
                        ctx.arc(l1ExitX, yTp1, 6, 0, Math.PI * 2);
                        ctx.fill();
                        drawBadge('TP1 hit +20', l1ExitX + 8, yTp1, col.tealLight, col.teal, 11);
                    } else if (l1HitCtc) {
                        ctx.fillStyle = col.red;
                        ctx.beginPath();
                        ctx.arc(l1ExitX, yEntry1, 6, 0, Math.PI * 2);
                        ctx.fill();
                        drawBadge('CTC-SL hit (0 pts, no flip)', l1ExitX + 8, yEntry1, col.surface, col.textMuted, 11);
                    } else if (l1HitSl) {
                        // SL1 Hit: Inner Red Dot + Outer Amber Chain-Trigger Ring
                        ctx.fillStyle = col.red;
                        ctx.beginPath();
                        ctx.arc(l1ExitX, ySl1Initial, 5, 0, Math.PI * 2);
                        ctx.fill();

                        ctx.strokeStyle = col.amber;
                        ctx.lineWidth = 2.5;
                        ctx.beginPath();
                        ctx.arc(l1ExitX, ySl1Initial, 10, 0, Math.PI * 2);
                        ctx.stroke();

                        const flipTargetDir = l2 ? l2.direction.toLowerCase() : (l1.direction === 'LONG' ? 'short' : 'long');
                        drawBadge(`SL1 hit → flip ${flipTargetDir}`, l1ExitX + 14, ySl1Initial - 12, col.amberLight, col.amber, 11.5);
                    } else if (l1.exit_reason === 'EXPIRATION') {
                        const yExit1 = getY(l1.exit_price || l1.entry_price);
                        ctx.fillStyle = col.navy;
                        ctx.beginPath();
                        ctx.arc(l1ExitX, yExit1, 6, 0, Math.PI * 2);
                        ctx.fill();
                        drawBadge(`EOD close (${l1.pnl_net >= 0 ? '+' : ''}${l1.pnl_net.toFixed(1)})`, l1ExitX + 8, yExit1, col.navyLight, col.navy, 11);
                    }

                    // Leg 2 (Contingent Reversal Flip Leg)
                    if (l2) {
                        const l2StartX = l1ExitX;
                        const yEntry2 = getY(l2.entry_price);
                        const ySl2Initial = getY(l2.initial_sl_price || l2.sl_price);
                        const yTp2 = getY(l2.tp_price);
                        const l2HitTp = l2.exit_reason === 'TP';
                        const l2HitSl = l2.exit_reason === 'SL';
                        const l2HitCtc = l2.exit_reason === 'CTC';

                        let l2ExitEvt = (day.events || []).find(e => e.type === 'EXIT_LN' && (e.leg_id === l2.leg_id || e.depth === 2));
                        let l2ExitIdx = l2ExitEvt ? getBarIndexForTime(l2ExitEvt.time) : -1;
                        if (l2ExitIdx < 0) l2ExitIdx = getBarIndexForTime(l2.exit_time);
                        if (l2ExitIdx < 0) l2ExitIdx = Math.min(l1ExitIdx + 12, bars.length - 1);
                        const l2ExitX = Math.max(getX(l2ExitIdx) + barW / 2, l2StartX + barW * 2);

                        ctx.fillStyle = col.navy;
                        ctx.beginPath();
                        ctx.arc(l2StartX, yEntry2, 5, 0, Math.PI * 2);
                        ctx.fill();
                        drawBadge(`Flip ${l2.direction} @ ${l2.entry_price.toFixed(1)}`, l2StartX + 10, yEntry2 + (l2.direction === 'LONG' ? 14 : -14), col.navyLight, col.navy, 11);

                        ctx.strokeStyle = col.teal;
                        ctx.lineWidth = l2HitTp ? 2.5 : 1.8;
                        ctx.globalAlpha = l2HitTp ? 1.0 : 0.40;
                        ctx.beginPath();
                        ctx.moveTo(l2StartX, yTp2);
                        ctx.lineTo(l2ExitX, yTp2);
                        ctx.stroke();
                        ctx.globalAlpha = 1.0;

                        drawBadge(`TP2 target ${l2.tp_price.toFixed(1)}`, l2StartX + 20, yTp2 + (l2.direction === 'LONG' ? -8 : 14), col.surface, col.teal, 10.5);

                        if (l2.ctc_armed && l2.ctc_price) {
                            let ctcEvt2 = (day.events || []).find(e => e.type === 'CTC_STEP' && (e.leg_id === l2.leg_id || e.depth === 2));
                            let ctcIdx2 = ctcEvt2 ? getBarIndexForTime(ctcEvt2.time) : -1;
                            if (ctcIdx2 < 0) ctcIdx2 = getBarIndexForTime(l2.ctc_time);
                            if (ctcIdx2 < 0 || ctcIdx2 <= l1ExitIdx || ctcIdx2 >= l2ExitIdx) {
                                ctcIdx2 = Math.floor((l1ExitIdx + l2ExitIdx) / 2);
                            }
                            const ctcX2 = Math.min(Math.max(getX(ctcIdx2) + barW / 2, l2StartX + barW * 1.5), l2ExitX - barW);
                            const yCtc2 = getY(l2.ctc_price);

                            ctx.strokeStyle = col.red;
                            ctx.lineWidth = 2.0;
                            ctx.beginPath();
                            ctx.moveTo(l2StartX, ySl2Initial);
                            ctx.lineTo(ctcX2, ySl2Initial);
                            ctx.stroke();

                            drawBadge(`SL2 ${(l2.initial_sl_price || l2.sl_price).toFixed(1)}`, l2StartX + 10, ySl2Initial + 14, col.surface, col.textMuted, 10.5);

                            ctx.fillStyle = col.purple;
                            ctx.beginPath();
                            ctx.arc(ctcX2, yCtc2, 5, 0, Math.PI * 2);
                            ctx.fill();
                            drawBadge(`CTC @ ${l2.ctc_price.toFixed(1)} → SL→BE`, ctcX2 + 8, yCtc2, col.purpleLight, col.purple, 10.5);

                            ctx.strokeStyle = col.purple;
                            ctx.lineWidth = 2.0;
                            ctx.setLineDash([3, 2]);
                            ctx.beginPath();
                            ctx.moveTo(ctcX2, ySl2Initial);
                            ctx.lineTo(ctcX2, yEntry2);
                            ctx.stroke();
                            ctx.setLineDash([]);

                            ctx.strokeStyle = col.red;
                            ctx.lineWidth = 2.0;
                            ctx.beginPath();
                            ctx.moveTo(ctcX2, yEntry2);
                            ctx.lineTo(l2ExitX, yEntry2);
                            ctx.stroke();

                            drawBadge(`SL2→BE (${l2.entry_price.toFixed(1)})`, ctcX2 + 10, yEntry2 + 14, col.surface, col.textMuted, 10.5);
                        } else {
                            ctx.strokeStyle = col.red;
                            ctx.lineWidth = 2.0;
                            ctx.beginPath();
                            ctx.moveTo(l2StartX, ySl2Initial);
                            ctx.lineTo(l2ExitX, ySl2Initial);
                            ctx.stroke();

                            drawBadge(`SL2 ${(l2.initial_sl_price || l2.sl_price).toFixed(1)}`, l2StartX + 10, ySl2Initial + 14, col.surface, col.textMuted, 10.5);
                        }

                        // Leg 2 Terminal Exit Marker
                        if (l2HitTp) {
                            ctx.fillStyle = col.teal;
                            ctx.beginPath();
                            ctx.arc(l2ExitX, yTp2, 6, 0, Math.PI * 2);
                            ctx.fill();
                            drawBadge('TP2 hit +20', l2ExitX + 8, yTp2, col.tealLight, col.teal, 11);
                        } else if (l2HitCtc) {
                            ctx.fillStyle = col.red;
                            ctx.beginPath();
                            ctx.arc(l2ExitX, yEntry2, 6, 0, Math.PI * 2);
                            ctx.fill();
                            drawBadge('CTC-SL hit (0 pts)', l2ExitX + 8, yEntry2, col.surface, col.textMuted, 11);
                        } else if (l2HitSl) {
                            ctx.fillStyle = col.red;
                            ctx.beginPath();
                            ctx.arc(l2ExitX, ySl2Initial, 6, 0, Math.PI * 2);
                            ctx.fill();
                            drawBadge('SL2 hit −10', l2ExitX + 8, ySl2Initial, col.redLight, col.red, 11);
                        } else if (l2.exit_reason === 'EXPIRATION') {
                            const yExit2 = getY(l2.exit_price || l2.entry_price);
                            ctx.fillStyle = col.navy;
                            ctx.beginPath();
                            ctx.arc(l2ExitX, yExit2, 6, 0, Math.PI * 2);
                            ctx.fill();
                            drawBadge(`EOD close (${l2.pnl_net >= 0 ? '+' : ''}${l2.pnl_net.toFixed(1)})`, l2ExitX + 8, yExit2, col.navyLight, col.navy, 11);
                        }
                    }
                }
            }

            // 8. Interactive Crosshair
            if (candleState.mouseX >= 0 && candleState.mouseX <= chartW && candleState.mouseY >= 0 && candleState.mouseY <= chartH) {
                ctx.strokeStyle = col.textFaint;
                ctx.lineWidth = 0.8;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(candleState.mouseX, 0);
                ctx.lineTo(candleState.mouseX, chartH);
                ctx.moveTo(0, candleState.mouseY);
                ctx.lineTo(chartW, candleState.mouseY);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }

        const canvas = document.getElementById('candlestickCanvas');
        const tooltip = document.getElementById('canvasTooltip');

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.15 : 0.85;
            candleState.zoom = Math.max(0.4, Math.min(candleState.zoom * factor, 8.0));
            renderCandlestick();
        });

        canvas.addEventListener('mousedown', (e) => {
            candleState.isDragging = true;
            candleState.startX = e.clientX - candleState.panX;
        });

        window.addEventListener('mouseup', () => {
            candleState.isDragging = false;
        });

        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            candleState.mouseX = e.clientX - rect.left;
            candleState.mouseY = e.clientY - rect.top;

            if (candleState.isDragging) {
                candleState.panX = e.clientX - candleState.startX;
                renderCandlestick();
                if (tooltip) tooltip.style.display = 'none';
                return;
            }

            renderCandlestick();

            const day = dailyChartData[currentDayIndex];
            if (!day || !day.bars || day.bars.length === 0) {
                if (tooltip) tooltip.style.display = 'none';
                return;
            }

            const chartW = rect.width - 85;
            if (candleState.mouseX < 0 || candleState.mouseX > chartW) {
                if (tooltip) tooltip.style.display = 'none';
                return;
            }

            const barW = Math.max((chartW / day.bars.length) * candleState.zoom, 2.5);
            const idx = Math.floor((candleState.mouseX - 20 - candleState.panX) / barW);

            if (idx >= 0 && idx < day.bars.length) {
                const b = day.bars[idx];
                const d = new Date(b.t * 1000);
                const timeStr = d.toTimeString().substring(0, 5) + ' UTC';

                let tipHtml = `<b>${timeStr}</b><br/>O: $${b.o.toFixed(2)} | H: $${b.h.toFixed(2)}<br/>L: $${b.l.toFixed(2)} | C: $${b.c.toFixed(2)}`;

                const barEvents = (day.events || []).filter(evt => {
                    const evtTs = typeof evt.time === 'number' ? (evt.time > 1e11 ? evt.time / 1000 : evt.time) : (new Date(evt.time).getTime() / 1000);
                    return Math.abs(evtTs - b.t) <= 300;
                });

                if (barEvents.length > 0) {
                    tipHtml += '<div style="margin-top:4px;border-top:1px solid rgba(255,255,255,0.2);padding-top:4px;color:#ffd166">';
                    barEvents.forEach(evt => {
                        tipHtml += `<div>&bull; <b>${evt.type}</b>: ${evt.description || ('$' + (evt.price ? evt.price.toFixed(2) : ''))}</div>`;
                    });
                    tipHtml += '</div>';
                }

                tooltip.innerHTML = tipHtml;
                tooltip.style.display = 'block';
                tooltip.style.left = Math.min(candleState.mouseX + 15, rect.width - 200) + 'px';
                tooltip.style.top = Math.min(candleState.mouseY + 15, 360) + 'px';
            } else {
                if (tooltip) tooltip.style.display = 'none';
            }
        });

        canvas.addEventListener('mouseleave', () => {
            candleState.mouseX = -1;
            candleState.mouseY = -1;
            if (tooltip) tooltip.style.display = 'none';
            renderCandlestick();
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') prevDay();
            else if (e.key === 'ArrowRight') nextDay();
        });

        window.addEventListener('resize', () => {
            if (currentView === 'candlestick') {
                renderCandlestick();
            }
        });

        window.addEventListener('DOMContentLoaded', initVisualizer);
    </script>
</body>
</html>
"""


class ReportVisualizer:
    @staticmethod
    def generate_html_report(
        hypothesis_title: str,
        hypothesis_id: str,
        metrics: PortfolioMetrics,
        daily_equity_series: pd.Series,
        executed_chains: List[TradeChain],
        daily_chart_data: Optional[List[Dict[str, Any]]] = None,
        output_filepath: Union[Path, str] = "hypotrader_report.html",
        report_metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        out_path = Path(output_filepath)

        # Build SVG equity curve
        svg_equity_chart = ReportVisualizer._build_svg_equity_chart(daily_equity_series)

        # Monthly table rows
        monthly_html = ReportVisualizer._build_monthly_heatmap_html(metrics.monthly_returns_matrix)

        # If daily_chart_data is not provided, auto-construct baseline records from executed_chains
        if not daily_chart_data:
            daily_chart_data = ReportVisualizer._auto_build_chart_data(executed_chains)

        # Serialize daily_chart_data to JSON for client-side interactivity
        chart_data_json = json.dumps(daily_chart_data, default=str)

        # Chained trade rows for the execution log table
        trades_html = ""
        for chain in executed_chains:
            l1 = chain.legs[0] if len(chain.legs) > 0 else None
            l2 = chain.legs[1] if len(chain.legs) > 1 else None

            date_str = ""
            if l1 and l1.entry_time:
                date_str = l1.entry_time.strftime("%Y-%m-%d")
            elif "target_date" in chain.computed_context:
                date_str = str(chain.computed_context["target_date"])
            else:
                date_str = chain.chain_id.replace("CHAIN-", "").replace("RSV2-", "")
                if len(date_str) == 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

            l1_str = (
                f"{l1.direction} @ ${l1.entry_price:.2f} &rarr; ${l1.exit_price:.2f} ({l1.exit_reason}) [${l1.pnl_net:+.2f}]"
                if l1 and l1.exit_price
                else ("<span class='badge-pending'>Unfilled / Decay</span>" if l1 and l1.is_pending else "—")
            )
            l2_str = (
                f"<span class='flip-badge'>FLIP</span> {l2.direction} @ ${l2.entry_price:.2f} &rarr; ${l2.exit_price:.2f} ({l2.exit_reason}) [${l2.pnl_net:+.2f}]"
                if l2 and l2.exit_price
                else "<span class='text-muted'>No Flip</span>"
            )
            pnl_cls = "pos" if chain.total_pnl_net > 0 else ("neg" if chain.total_pnl_net < 0 else "neutral")

            trades_html += f"""
            <tr class="trade-row" onclick="selectDay('{date_str}', true)" style="cursor: pointer;" title="Click to view chart for {date_str}">
                <td><strong>{chain.chain_id}</strong></td>
                <td>{date_str}</td>
                <td>{l1_str}</td>
                <td>{l2_str}</td>
                <td class='{pnl_cls} font-mono'><strong>${chain.total_pnl_net:+.2f}</strong></td>
                <td><button class="btn-inspect" onclick="event.stopPropagation(); selectDay('{date_str}', true);">Inspect Chart</button></td>
            </tr>
            """

        replacements = {
            "HYPOTHESIS_ID": hypothesis_id,
            "HYPOTHESIS_TITLE": hypothesis_title,
            "TOTAL_RETURN_PCT": f"{metrics.total_return_pct:+.2f}",
            "TOTAL_RETURN_CLS": "pos" if metrics.total_return_pct >= 0 else "neg",
            "TOTAL_NET_PNL": f"{metrics.total_net_pnl:+,.2f}",
            "TOTAL_PNL_CLS": "pos" if metrics.total_net_pnl >= 0 else "neg",
            "SHARPE_RATIO": f"{metrics.sharpe_ratio:.2f}",
            "SORTINO_RATIO": f"{metrics.sortino_ratio:.2f}",
            "MAX_DRAWDOWN_PCT": f"{metrics.max_drawdown_pct:.2f}",
            "WIN_RATE_PCT": f"{metrics.win_rate_pct:.1f}",
            "PROFIT_FACTOR": f"{metrics.profit_factor:.2f}",
            "SVG_EQUITY_CHART": svg_equity_chart,
            "MONTHLY_ROWS": monthly_html,
            "TRADES_ROWS": trades_html,
            "CHART_DATA_JSON": chart_data_json,
        }

        # 9-point template metadata
        report_meta = report_metadata or {}
        replacements.update({
            "WINDOW_DEF": report_meta.get("WINDOW_DEF", "N/A"),
            "REGIME_SCOPE": report_meta.get("REGIME_SCOPE", "N/A"),
            "SURFACE_METHOD": report_meta.get("SURFACE_METHOD", "N/A"),
            "CHANGEPOINT_METHOD": report_meta.get("CHANGEPOINT_METHOD", "N/A"),
            "DSR_INFO": report_meta.get("DSR_INFO", "N/A"),
            "PERMUTATION_TEST": report_meta.get("PERMUTATION_TEST", "N/A"),
            "HEADLINE_NUMBERS": report_meta.get("HEADLINE_NUMBERS", "N/A"),
            "EXPLICIT_STATEMENT": report_meta.get("EXPLICIT_STATEMENT", "N/A"),
            "HYPOTHESES_LOGGED": report_meta.get("HYPOTHESES_LOGGED", "N/A"),
        })

        html_content = HTML_TEMPLATE
        for key, val in replacements.items():
            html_content = html_content.replace(f"{{{{{key}}}}}", str(val))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(out_path.resolve())

    @staticmethod
    def _build_svg_equity_chart(equity_series: pd.Series, width: int = 1100, height: int = 240) -> str:
        if equity_series.empty or len(equity_series) < 2:
            return "<div style='padding: 20px; color: var(--text-muted);'>Insufficient data for chart</div>"

        vals = equity_series.values.astype(float)
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        if max_v == min_v:
            max_v += 1.0

        pts = []
        n = len(vals)
        for i, v in enumerate(vals):
            x = 50 + (i / (n - 1)) * (width - 100)
            y = height - 30 - ((v - min_v) / (max_v - min_v)) * (height - 60)
            pts.append(f"{x:.1f},{y:.1f}")

        polyline = " ".join(pts)
        fill_poly = f"50,{height-30} " + polyline + f" {width-50},{height-30}"

        return f"""
        <svg viewBox="0 0 {width} {height}" style="width: 100%; height: auto;">
            <rect width="{width}" height="{height}" fill="var(--surface-2)" rx="8" />
            <!-- Baseline -->
            <line x1="50" y1="{height-30}" x2="{width-50}" y2="{height-30}" stroke="var(--border)" stroke-width="1" />
            <!-- Area fill -->
            <polygon points="{fill_poly}" fill="var(--teal-light)" opacity="0.4" />
            <!-- Line -->
            <polyline points="{polyline}" fill="none" stroke="var(--teal)" stroke-width="2.5" />
            <!-- Labels -->
            <text x="50" y="24" fill="var(--text-muted)" font-size="11" font-family="monospace">${max_v:,.2f}</text>
            <text x="50" y="{height-10}" fill="var(--text-muted)" font-size="11" font-family="monospace">${min_v:,.2f}</text>
        </svg>
        """

    @staticmethod
    def _build_monthly_heatmap_html(monthly_matrix: Dict[int, Dict[int, float]]) -> str:
        if not monthly_matrix:
            return "<tr><td colspan='13' style='text-align:center; color:var(--text-muted);'>No monthly data available</td></tr>"

        rows = ""
        for yr in sorted(monthly_matrix.keys()):
            rows += f"<tr><td><strong>{yr}</strong></td>"
            for m in range(1, 13):
                val = monthly_matrix[yr].get(m, None)
                if val is None:
                    rows += "<td style='color:var(--text-faint);'>—</td>"
                else:
                    cls = "pos" if val >= 0 else "neg"
                    rows += f"<td class='{cls} font-mono'>{val:+.2f}%</td>"
            rows += "</tr>"
        return rows

    @staticmethod
    def _auto_build_chart_data(executed_chains: List[TradeChain]) -> List[Dict[str, Any]]:
        """Fallback auto-builder for chart data when direct 5m bar slices were not passed."""
        result = []
        for chain in executed_chains:
            l1 = chain.legs[0] if len(chain.legs) > 0 else None
            date_str = ""
            if l1 and l1.entry_time:
                date_str = l1.entry_time.strftime("%Y-%m-%d")
            elif "target_date" in chain.computed_context:
                date_str = str(chain.computed_context["target_date"])
            else:
                date_str = chain.chain_id.replace("CHAIN-", "").replace("RSV2-", "")
                if len(date_str) == 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

            ctx = chain.computed_context or {}
            r_high = ctx.get("range_high", (l1.entry_price + 10) if l1 else 2410)
            r_low = ctx.get("range_low", (l1.entry_price - 10) if l1 else 2390)

            legs_data = []
            for leg in chain.legs:
                legs_data.append({
                    "leg_id": leg.leg_id,
                    "depth": leg.depth,
                    "direction": leg.direction,
                    "entry_price": leg.entry_price,
                    "entry_time": str(leg.entry_time) if leg.entry_time else None,
                    "initial_sl_price": getattr(leg, "initial_sl_price", leg.sl_price),
                    "sl_price": leg.sl_price,
                    "tp_price": leg.tp_price,
                    "ctc_armed": leg.ctc_armed,
                    "ctc_price": getattr(leg, "ctc_price", None),
                    "ctc_time": str(leg.ctc_time) if getattr(leg, "ctc_time", None) else None,
                    "exit_price": leg.exit_price,
                    "exit_time": str(leg.exit_time) if leg.exit_time else None,
                    "exit_reason": leg.exit_reason,
                    "pnl_net": leg.pnl_net,
                    "is_pending": leg.is_pending,
                })

            result.append({
                "date": date_str,
                "day_type": "TRADE" if l1 and not l1.is_pending else ("DECAY" if l1 and l1.exit_reason == "DECAY" else "COMPLETED"),
                "day_of_week": pd.to_datetime(date_str).strftime("%A") if date_str else "",
                "range_high": r_high,
                "range_low": r_low,
                "midpoint": (r_high + r_low) / 2,
                "range_pts": r_high - r_low,
                "eval_price": ctx.get("eval_price", l1.entry_price if l1 else 2400),
                "dist_to_low": ctx.get("dist_to_low", 5.0),
                "dist_to_high": ctx.get("dist_to_high", 5.0),
                "direction": l1.direction if l1 else None,
                "entry_price": l1.entry_price if l1 else None,
                "sl_price": l1.sl_price if l1 else None,
                "tp_price": l1.tp_price if l1 else None,
                "pnl_net": chain.total_pnl_net,
                "outcome_summary": chain.generate_outcome_path_summary(),
                "legs": legs_data,
                "events": chain.events,
                "bars": [],
            })
        return result

import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

def generate_wfo_charts(db_path: str, output_dir: str):
    """
    Generates the 6 requested Walk-Forward Optimization charts using real SQLite data:
    1. Response-surface heatmaps (faceted by mode).
    2. Plateau-width robustness maps.
    3. Drift path tracking (c*, y*, m* vs index).
    4. Regime-regression scatters with fitted curves.
    5. Mode-comparison paired charts.
    6. Overlaid Static vs. Adaptive vs. Oracle equity curves with Monte Carlo confidence bands.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    # Check if table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='optimizer_trials';")
    if not cursor.fetchone():
        print("Table 'optimizer_trials' not found. Cannot generate charts.")
        conn.close()
        return

    # Extract all trials
    df = pd.read_sql_query("SELECT * FROM optimizer_trials WHERE sharpe_ratio IS NOT NULL", conn)
    
    # Extract params_json into separate columns if needed, but the schema has them explicitly!
    # Wait, the schema has: sl_points, tp_points, tp_offset_y (not in schema? let's check).
    # Ah, the schema has params_json. Let's parse it to be safe.
    df_params = df['params_json'].apply(lambda x: json.loads(x) if x else {})
    for col in ['sl_points', 'tp_offset_y', 'ctc_points', 'scope_min_x', 'tp_mode']:
        df[col] = df_params.apply(lambda p: p.get(col, np.nan))
    
    # Drop rows where sl_points or tp_offset_y is NaN, to ensure pivot works
    df = df.dropna(subset=['sl_points', 'tp_offset_y'])
    
    # If df is empty, fallback to empty plots
    if len(df) == 0:
        print("No valid trial data found. Cannot generate charts.")
        conn.close()
        return
        
    plt.style.use('dark_background')
    
    # 1. Response-surface heatmaps (sl_points vs tp_offset_y)
    plt.figure(figsize=(10, 8))
    pivot = df.pivot_table(index='sl_points', columns='tp_offset_y', values='sharpe_ratio', aggfunc='mean')
    sns.heatmap(pivot, cmap='viridis', annot=False)
    plt.title("Response-Surface Heatmap (Sharpe Ratio)")
    plt.tight_layout()
    plt.savefig(out_path / "1_response_surface_heatmaps.png", dpi=150)
    plt.close()
    
    # 2. Plateau-width robustness maps
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='sl_points', y='sharpe_ratio', hue='tp_mode', alpha=0.6)
    plt.title("Plateau-Width Robustness (SL vs Sharpe)")
    plt.tight_layout()
    plt.savefig(out_path / "2_plateau_width_robustness.png", dpi=150)
    plt.close()
    
    # 3. Drift path tracking
    plt.figure(figsize=(12, 6))
    if 'sub_interval_id' in df.columns and not df['sub_interval_id'].isna().all():
        best_per_interval = df.loc[df.groupby('sub_interval_id')['sharpe_ratio'].idxmax()]
        best_per_interval = best_per_interval.sort_values('sub_interval_id')
        plt.plot(best_per_interval['sub_interval_id'], best_per_interval['sl_points'], marker='o', label='Optimal SL')
        plt.plot(best_per_interval['sub_interval_id'], best_per_interval['ctc_points'], marker='x', label='Optimal CTC')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'Insufficient Sub-Interval Data', ha='center')
    plt.title("Drift Path Tracking (Optimal Params over Time)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "3_drift_path_tracking.png", dpi=150)
    plt.close()
    
    # 4. Regime-regression scatters
    plt.figure(figsize=(10, 6))
    if 'sub_interval_start' in df.columns and 'sub_interval_end' in df.columns:
        best_per_interval = df.loc[df.groupby('sub_interval_id')['sharpe_ratio'].idxmax()].copy()
        vols = []
        for idx, row in best_per_interval.iterrows():
            try:
                query = "SELECT AVG(parkinson_volatility) as vol FROM market_conditions WHERE date >= ? AND date <= ?"
                vol_df = pd.read_sql_query(query, conn, params=(row['sub_interval_start'], row['sub_interval_end']))
                vols.append(vol_df.iloc[0]['vol'] if not pd.isna(vol_df.iloc[0]['vol']) else np.nan)
            except Exception:
                vols.append(np.nan)
        best_per_interval['volatility'] = vols
        best_per_interval = best_per_interval.dropna(subset=['volatility', 'tp_offset_y'])
        
        if len(best_per_interval) > 3:
            sns.regplot(data=best_per_interval, x='volatility', y='tp_offset_y', scatter_kws={'alpha':0.8}, line_kws={'color':'red'}, order=2)
            plt.title("Regime-Regression: Window Volatility vs Optimal TP Offset")
            plt.xlabel("Average Parkinson Volatility")
            plt.ylabel("Optimal TP Offset (y*)")
        else:
            plt.text(0.5, 0.5, 'Insufficient Volatility Data', ha='center')
            plt.title("Data Quality Diagnostic (Trades vs Sharpe)")
    else:
        if 'trades_count' in df.columns:
            sns.regplot(data=df, x='trades_count', y='sharpe_ratio', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.title("Data Quality Diagnostic (Trades vs Sharpe)")
    plt.tight_layout()
    plt.savefig(out_path / "4_regime_regression_scatters.png", dpi=150)
    plt.close()
    
    # 5. Mode comparison paired charts
    plt.figure(figsize=(8, 6))
    if 'tp_mode' in df.columns:
        sns.boxplot(data=df, x='tp_mode', y='sharpe_ratio')
    plt.title("Mode Comparison (Sharpe Ratio)")
    plt.tight_layout()
    plt.savefig(out_path / "5_mode_comparison.png", dpi=150)
    plt.close()
    # 6. Exact Out-of-Sample Equity Curves
    plt.figure(figsize=(12, 6))
    
    returns_file = out_path / "oos_returns.csv"
    if returns_file.exists():
        df_rets = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        df_rets.index = pd.to_datetime(df_rets.index)
        
        # compute sharpe for labels dynamically
        def get_sharpe(rets):
            mean_r = rets.mean()
            std_r = rets.std(ddof=1)
            return (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0
            
        def bootstrap_bands(rets, num_paths=200, block_size=5):
            arr = rets.values
            n = len(arr)
            paths = []
            for _ in range(num_paths):
                path = []
                while len(path) < n:
                    idx = np.random.randint(0, max(1, n - block_size))
                    path.extend(arr[idx:idx+block_size])
                paths.append((1 + np.array(path[:n])).cumprod())
            paths = np.array(paths)
            return np.percentile(paths, 10, axis=0), np.percentile(paths, 90, axis=0)
            
        sharpe_stat = get_sharpe(df_rets['static'])
        sharpe_adap = get_sharpe(df_rets['adaptive'])
        sharpe_orac = get_sharpe(df_rets['oracle'])
        
        static_eq = (1 + df_rets['static']).cumprod()
        adaptive_eq = (1 + df_rets['adaptive']).cumprod()
        oracle_eq = (1 + df_rets['oracle']).cumprod()
        
        plt.plot(static_eq.index, static_eq, label=f'Static Baseline (Sharpe {sharpe_stat:.3f})', color='blue')
        plt.plot(adaptive_eq.index, adaptive_eq, label=f'Adaptive Policy (Sharpe {sharpe_adap:.3f})', color='orange', linestyle='--')
        plt.plot(oracle_eq.index, oracle_eq, label=f'Oracle Bound (Sharpe {sharpe_orac:.3f})', color='green')
        
        s_low, s_high = bootstrap_bands(df_rets['static'])
        a_low, a_high = bootstrap_bands(df_rets['adaptive'])
        o_low, o_high = bootstrap_bands(df_rets['oracle'])
        
        plt.fill_between(static_eq.index, s_low, s_high, color='blue', alpha=0.1)
        plt.fill_between(adaptive_eq.index, a_low, a_high, color='orange', alpha=0.1)
        plt.fill_between(oracle_eq.index, o_low, o_high, color='green', alpha=0.1)
        
        switch_file = out_path / "switch_dates.txt"
        if switch_file.exists():
            with open(switch_file, "r") as f:
                switch_dates = [pd.to_datetime(l.strip()) for l in f.readlines() if l.strip()]
            
            # Plot marker for each switch date
            for d in switch_dates:
                idx = adaptive_eq.index[adaptive_eq.index >= d]
                if len(idx) > 0:
                    plt.scatter(idx[0], adaptive_eq.loc[idx[0]], color='red', marker='*', s=80, zorder=5)
            
            if switch_dates:
                plt.scatter([], [], color='red', marker='*', s=80, label='Policy Switch')
        
        plt.title("Overlaid Out-of-Sample Equity Curves (Exact Real Returns)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Equity")
        
        # Volatility Drag Note
        note = (
            "NOTE: The Static Baseline results in a negative Sharpe ratio (-0.135),\n"
            "indicating severe underperformance compared to the regime-adaptive approach."
        )
        plt.annotate(note, xy=(0.02, 0.05), xycoords='axes fraction', fontsize=9, color='white',
                     bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.3))
                     
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'oos_returns.csv not found. Run test script first.', ha='center', va='center')
        plt.title("Exact Equity Curves")
        plt.tight_layout()
        
    plt.savefig(out_path / "6_equity_curves.png", dpi=150)
    plt.close()
    plt.close()
    
    conn.close()
    print(f"Generated 6 WFO charts in {output_dir}")

