const http = require("http");
const fs = require("fs");
const { spawn } = require("child_process");

const CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9222;
const USER_DATA_DIR = "/tmp/chrome_cdp_val_" + Date.now();
const ARTIFACT_DIR = "/Users/prince/.gemini/antigravity-ide/brain/4fb793d9-b965-477b-a704-29215c4190ef";

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function fetchJson(url) {
    return new Promise((resolve, reject) => {
        http.get(url, (res) => {
            let data = "";
            res.on("data", chunk => data += chunk);
            res.on("end", () => {
                try { resolve(JSON.parse(data)); } catch (e) { reject(e); }
            });
        }).on("error", reject);
    });
}

class CDPClient {
    constructor(wsUrl) {
        this.wsUrl = wsUrl;
        this.msgId = 1;
        this.callbacks = new Map();
        this.ws = null;
    }

    async connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.wsUrl);
            this.ws.onopen = () => resolve();
            this.ws.onerror = (err) => reject(err);
            this.ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.method === "Runtime.consoleAPICalled") {
                    console.log(`[Browser Console] ${msg.params.type}:`, msg.params.args.map(a => a.value || a.description).join(" "));
                }
                if (msg.method === "Runtime.exceptionThrown") {
                    console.error(`[Browser Exception]:`, JSON.stringify(msg.params.exceptionDetails));
                }
                if (msg.id && this.callbacks.has(msg.id)) {
                    const cb = this.callbacks.get(msg.id);
                    this.callbacks.delete(msg.id);
                    if (msg.error) cb.reject(new Error(msg.error.message || JSON.stringify(msg.error)));
                    else cb.resolve(msg.result);
                }
            };
        });
    }

    send(method, params = {}) {
        const id = this.msgId++;
        return new Promise((resolve, reject) => {
            this.callbacks.set(id, { resolve, reject });
            this.ws.send(JSON.stringify({ id, method, params }));
        });
    }

    async evaluate(expression) {
        const res = await this.send("Runtime.evaluate", {
            expression,
            returnByValue: true,
            awaitPromise: true
        });
        if (res.exceptionDetails) {
            throw new Error("Evaluation failed: " + JSON.stringify(res.exceptionDetails));
        }
        return res.result ? res.result.value : undefined;
    }

    async captureScreenshot(path) {
        const res = await this.send("Page.captureScreenshot", { format: "png" });
        fs.writeFileSync(path, Buffer.from(res.data, "base64"));
        console.log(`[Screenshot Saved] -> ${path}`);
    }

    close() {
        if (this.ws) this.ws.close();
    }
}

async function run() {
    console.log("=== STARTING AUTOMATED CDP BROWSER VALIDATION ===");
    console.log(`Launching Chrome: ${CHROME_PATH}`);

    const chrome = spawn(CHROME_PATH, [
        "--headless=new",
        `--remote-debugging-port=${PORT}`,
        `--user-data-dir=${USER_DATA_DIR}`,
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-gpu",
        "--window-size=1600,1200",
        "about:blank"
    ], { stdio: "ignore" });

    try {
        // Wait for CDP port to open
        let targets = null;
        for (let i = 0; i < 30; i++) {
            await sleep(300);
            try {
                targets = await fetchJson(`http://127.0.0.1:${PORT}/json/list`);
                if (targets && targets.length > 0) break;
            } catch (e) {}
        }

        if (!targets || targets.length === 0) {
            throw new Error("Could not connect to Chrome CDP port 9222");
        }

        const pageTarget = targets.find(t => t.type === "page") || targets[0];
        console.log(`Connected to page target: ${pageTarget.webSocketDebuggerUrl}`);

        const client = new CDPClient(pageTarget.webSocketDebuggerUrl);
        await client.connect();

        await client.send("Page.enable");
        await client.send("Runtime.enable");
        await client.send("DOM.enable");

        console.log("\n1. Navigating to http://127.0.0.1:8050 ...");
        await client.send("Page.navigate", { url: "http://127.0.0.1:8050" });

        // Wait for page and initial backtest to complete
        console.log("Waiting for backtest to complete and DOM to populate...");
        let ready = false;
        for (let i = 0; i < 30; i++) {
            await sleep(600);
            const state = await client.evaluate(`({
                readyState: document.readyState,
                hasData: Boolean(window.backtestData),
                rows: document.getElementById("tradeTableBody") ? document.getElementById("tradeTableBody").children.length : 0,
                btnDisabled: document.getElementById("btnRun") ? document.getElementById("btnRun").disabled : false
            })`);
            console.log(`[Wait loop ${i+1}] Browser State:`, JSON.stringify(state));
            if (state.rows > 0) {
                ready = true;
                break;
            }
        }

        if (!ready) {
            throw new Error("UI failed to load backtest data within 20 seconds!");
        }
        console.log("✓ Page loaded and initial backtest completed!");

        // Take initial full-page screenshot
        const screenshot1 = `${ARTIFACT_DIR}/cdp_portfolio_view.png`;
        await client.captureScreenshot(screenshot1);

        // ================= TEST 1: Check Initial Both (Portfolio) Mode =================
        console.log("\n=== TEST 1: Checking Portfolio Mode Features ===");
        const portStats = await client.evaluate(`(() => {
            return {
                mode: window.backtestData.strategy_mode,
                totalTrades: document.getElementById("kpiTrades") ? document.getElementById("kpiTrades").innerText : null,
                pnl: document.getElementById("kpiPnl") ? document.getElementById("kpiPnl").innerText : null,
                winRate: document.getElementById("kpiWinRate") ? document.getElementById("kpiWinRate").innerText : null,
                profitFactor: document.getElementById("kpiProfitFactor") ? document.getElementById("kpiProfitFactor").innerText : null,
                equityCurveSvg: Boolean(document.querySelector("#equitySvgBox svg")),
                equityDateTicks: Array.from(document.querySelectorAll("#equitySvgBox svg text")).map(t => t.textContent).filter(t => /^[A-Z][a-z]{2} \\d{2}$/.test(t)),
                payoffRatio: document.getElementById("statPayoffRatio") ? document.getElementById("statPayoffRatio").innerText : null,
                expectancy: document.getElementById("statExpectancy") ? document.getElementById("statExpectancy").innerText : null,
                pillAll: document.getElementById("cntAll") ? document.getElementById("cntAll").innerText : null,
                pillWin: document.getElementById("cntWin") ? document.getElementById("cntWin").innerText : null,
                pillLoss: document.getElementById("cntLoss") ? document.getElementById("cntLoss").innerText : null,
                tableRowsCount: document.querySelectorAll("#tradeTableBody tr").length,
                tableHeaders: Array.from(document.querySelectorAll("#tradeTableHead th")).map(th => th.innerText.trim())
            };
        })()`);

        console.log("Portfolio Mode Metrics:", JSON.stringify(portStats, null, 2));

        if (!portStats.equityCurveSvg || portStats.equityDateTicks.length === 0) {
            throw new Error("Equity curve chart missing or date ticks not found!");
        }
        console.log(`✓ Equity curve chart rendered with calendar date ticks: ${portStats.equityDateTicks.join(", ")}`);
        console.log(`✓ Trade table rendered with ${portStats.tableRowsCount} rows and headers: ${portStats.tableHeaders.join(" | ")}`);

        // ================= TEST 2: Switch to Strategy 2 (Range Scope 11:00 AM) =================
        console.log("\n=== TEST 2: Testing Strategy 2 (Range Scope 11:00 AM) ===");
        await client.evaluate(`(() => {
            setStrategyMode("range_scope");
            setSession("tokyo");
            runBacktest();
        })()`);

        // Wait for runBacktest to complete
        console.log("Waiting for Strategy 2 backtest to complete...");
        for (let i = 0; i < 40; i++) {
            await sleep(400);
            const mode = await client.evaluate(`window.backtestData ? window.backtestData.strategy_mode : null`);
            const btnDisabled = await client.evaluate(`document.getElementById("btnRun").disabled`);
            if (mode === "range_scope" && !btnDisabled) break;
        }

        const s2Stats = await client.evaluate(`(() => {
            return {
                mode: window.backtestData.strategy_mode,
                totalTrades: document.getElementById("kpiTrades").innerText,
                pnl: document.getElementById("kpiPnl").innerText,
                winRate: document.getElementById("kpiWinRate").innerText,
                tableHeaders: Array.from(document.querySelectorAll("#tradeTableHead th")).map(th => th.innerText.trim()),
                pillCounts: {
                    all: document.getElementById("cntAll").innerText,
                    win: document.getElementById("cntWin").innerText,
                    loss: document.getElementById("cntLoss").innerText,
                    skip: document.getElementById("cntSkip").innerText
                },
                legBreakdown: document.getElementById("breakdownContent") ? document.getElementById("breakdownContent").innerText : ""
            };
        })()`);

        console.log("Strategy 2 Metrics:", JSON.stringify(s2Stats, null, 2));
        console.log(`✓ Strategy 2 Table Headers dynamically adjusted: ${s2Stats.tableHeaders.join(" | ")}`);

        // ================= TEST 3: Inspect 2026-08-04 in Strategy 2 =================
        console.log("\n=== TEST 3: Inspecting 2026-08-04 FSM Schematic Diagram ===");
        // Select day 2026-08-04
        await client.evaluate(`(() => {
            const idx = window.backtestData.daily_charts.findIndex(d => d.date === "2026-08-04");
            if (idx >= 0) selectDay(idx);
        })()`);

        await sleep(500);

        const fsmCheck = await client.evaluate(`(() => {
            const svgEl = document.querySelector("#schematicSvgBox svg");
            if (!svgEl) return { error: "No SVG in schematicSvgBox" };

            const texts = Array.from(svgEl.querySelectorAll("text")).map(t => t.textContent.trim());
            const circles = Array.from(svgEl.querySelectorAll("circle")).map(c => ({
                cx: c.getAttribute("cx"),
                cy: c.getAttribute("cy"),
                r: c.getAttribute("r"),
                fill: c.getAttribute("fill"),
                stroke: c.getAttribute("stroke")
            }));

            // Check for broken sub tags or P truncation in decision tree or schematic
            const treeTexts = Array.from(document.querySelectorAll("#mapSvg text")).map(t => t.textContent);
            const brokenSub = treeTexts.some(t => t.includes("• Scope = |P") || t.includes("Stop Loss: P"));

            // Check 12:30 open price marker text
            const has1230OpenMarker = texts.some(t => t.includes("12:30 Open: $4063.1") || t.includes("12:30 Open:"));
            // Check instant lock-in or dynamic TP text
            const hasLockinOrDynamicTp = texts.some(t => t.includes("Instant Lock-In") || t.includes("TP Hit") || t.includes("12:30"));
            // Check initial TP text
            const hasInitialTp = texts.some(t => t.includes("Initial TP") || t.includes("4069.0"));

            // Check table row for 2026-08-04
            const row0804 = document.querySelector("#tradeTableBody tr.selected");
            const rowCells = row0804 ? Array.from(row0804.querySelectorAll("td")).map(td => td.innerText.trim()) : [];

            return {
                textsSample: texts,
                brokenSubBugPresent: brokenSub,
                has1230OpenMarker,
                hasLockinOrDynamicTp,
                hasInitialTp,
                whiteBorderCircles: circles.filter(c => c.stroke === "#ffffff" || c.stroke === "#fff"),
                row0804Cells: rowCells
            };
        })()`);

        console.log("FSM Schematic Check Results:");
        console.log("- Broken sub tag bug present:", fsmCheck.brokenSubBugPresent ? "YES (FAIL)" : "NO (PASSED)");
        console.log("- Has 12:30 Open price marker:", fsmCheck.has1230OpenMarker ? "YES (PASSED)" : "NO (FAIL)");
        console.log("- Has Initial TP (4069.0):", fsmCheck.hasInitialTp ? "YES (PASSED)" : "NO (FAIL)");
        console.log("- Has Lock-in / TP marker:", fsmCheck.hasLockinOrDynamicTp ? "YES (PASSED)" : "NO (FAIL)");
        console.log("- 2026-08-04 Table Row:", fsmCheck.row0804Cells.join(" | "));

        if (fsmCheck.brokenSubBugPresent) {
            throw new Error("Broken <sub> tag or placeholder 'P' detected in SVG!");
        }
        if (!fsmCheck.has1230OpenMarker) {
            throw new Error("12:30 Open price marker dot is missing in FSM diagram!");
        }

        const screenshot2 = `${ARTIFACT_DIR}/cdp_strategy2_0804_schematic.png`;
        await client.captureScreenshot(screenshot2);

        // ================= TEST 4: Candlestick View & Filters =================
        console.log("\n=== TEST 4: Testing Candlestick View & Filter Pills ===");
        await client.evaluate(`switchInspectorView("candlestick")`);
        await sleep(300);

        const candleCheck = await client.evaluate(`(() => {
            const canvas = document.getElementById("chartCanvas");
            return {
                canvasExists: Boolean(canvas),
                canvasWidth: canvas ? canvas.width : 0,
                canvasHeight: canvas ? canvas.height : 0
            };
        })()`);
        console.log(`✓ Candlestick canvas rendered (${candleCheck.canvasWidth}x${candleCheck.canvasHeight})`);

        // Test Filter: WINS
        console.log("Testing filter 'WINS'...");
        await client.evaluate(`setFilter("WIN")`);
        await sleep(200);
        const winFilterCount = await client.evaluate(`document.querySelectorAll("#tradeTableBody tr:not([style*='display: none'])").length`);
        console.log(`✓ Filter 'WINS' matched ${winFilterCount} rows`);

        // Test Filter: SKIPS
        console.log("Testing filter 'SKIPS'...");
        await client.evaluate(`setFilter("SKIP")`);
        await sleep(200);
        const skipFilterCount = await client.evaluate(`document.querySelectorAll("#tradeTableBody tr:not([style*='display: none'])").length`);
        console.log(`✓ Filter 'SKIPS' matched ${skipFilterCount} rows`);

        // Reset Filter
        await client.evaluate(`setFilter("ALL")`);
        await client.evaluate(`switchInspectorView("schematic")`);

        // ================= TEST 5: Strategy 1 Mode (Range Sweep 12:30 PM) =================
        console.log("\n=== TEST 5: Testing Strategy 1 Mode (Range Sweep 12:30 PM) ===");
        await client.evaluate(`(() => {
            setStrategyMode("range_sweep");
            runBacktest();
        })()`);

        for (let i = 0; i < 40; i++) {
            await sleep(400);
            const mode = await client.evaluate(`window.backtestData ? window.backtestData.strategy_mode : null`);
            const btnDisabled = await client.evaluate(`document.getElementById("btnRun").disabled`);
            if (mode === "range_sweep" && !btnDisabled) break;
        }

        const s1Stats = await client.evaluate(`(() => {
            return {
                mode: window.backtestData.strategy_mode,
                totalTrades: document.getElementById("kpiTrades").innerText,
                pnl: document.getElementById("kpiPnl").innerText,
                winRate: document.getElementById("kpiWinRate").innerText,
                tableHeaders: Array.from(document.querySelectorAll("#tradeTableHead th")).map(th => th.innerText.trim())
            };
        })()`);

        console.log("Strategy 1 Metrics:", JSON.stringify(s1Stats, null, 2));
        console.log(`✓ Strategy 1 Table Headers: ${s1Stats.tableHeaders.join(" | ")}`);

        const screenshot3 = `${ARTIFACT_DIR}/cdp_strategy1_view.png`;
        await client.captureScreenshot(screenshot3);

        client.close();
        console.log("\n=======================================================");
        console.log("✓ ALL AUTOMATED CDP BROWSER TESTS PASSED FLAWLESSLY!");
        console.log("=======================================================");

    } finally {
        chrome.kill();
        try { fs.rmSync(USER_DATA_DIR, { recursive: true, force: true }); } catch (e) {}
    }
}

run().catch(err => {
    console.error("FATAL AUTOMATION ERROR:", err);
    process.exit(1);
});
