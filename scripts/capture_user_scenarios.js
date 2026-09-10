const http = require("http");
const fs = require("fs");
const { spawn } = require("child_process");

const CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9224;
const USER_DATA_DIR = "/tmp/chrome_scen_" + Date.now();
const ARTIFACT_DIR = "/Users/prince/.gemini/antigravity-ide/brain/4fb793d9-b965-477b-a704-29215c4190ef";

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function fetchJson(url) {
    return new Promise((resolve, reject) => {
        http.get(url, (res) => {
            let data = ""; res.on("data", c => data += c);
            res.on("end", () => resolve(JSON.parse(data)));
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
        if (res.exceptionDetails) throw new Error(JSON.stringify(res.exceptionDetails));
        return res.result.value;
    }

    async captureElementScreenshot(selector, outputPath) {
        const box = await this.evaluate(`
            (() => {
                const el = document.querySelector("${selector}");
                if (!el) return null;
                const rect = el.getBoundingClientRect();
                return {
                    x: rect.left + window.scrollX,
                    y: rect.top + window.scrollY,
                    width: rect.width,
                    height: rect.height
                };
            })()
        `);
        if (!box) throw new Error("Element not found: " + selector);

        const res = await this.send("Page.captureScreenshot", {
            format: "png",
            clip: {
                x: Math.max(0, box.x),
                y: Math.max(0, box.y),
                width: box.width,
                height: box.height,
                scale: 1
            },
            captureBeyondViewport: true
        });
        fs.writeFileSync(outputPath, Buffer.from(res.data, "base64"));
        console.log("[Screenshot Saved] -> " + outputPath);
    }
}

async function main() {
    console.log("Launching Chrome for User Scenario Captures...");
    const chromeProc = spawn(CHROME_PATH, [
        "--headless=new",
        `--remote-debugging-port=${PORT}`,
        `--user-data-dir=${USER_DATA_DIR}`,
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1600,1200"
    ]);

    try {
        await sleep(1500);
    const versionInfo = await fetchJson(`http://127.0.0.1:${PORT}/json/version`);
    const cdp = new CDPClient(versionInfo.webSocketDebuggerUrl);
    await cdp.connect();

    const targets = await fetchJson(`http://127.0.0.1:${PORT}/json`);
    const pageTarget = targets.find(t => t.type === "page");
    const pageCdp = new CDPClient(pageTarget.webSocketDebuggerUrl);
    await pageCdp.connect();

    await pageCdp.send("Page.enable");
    await pageCdp.send("Page.navigate", { url: "http://127.0.0.1:8050" });

    // Wait for data
    for (let i = 0; i < 40; i++) {
        await sleep(400);
        const hasData = await pageCdp.evaluate(`Boolean(window.backtestData && (window.backtestData.daily_charts || window.backtestData.daily_chart_records))`);
        if (hasData) break;
    }

        // SCENARIO 1: Portfolio Mode - 2026-08-19 (Bug 1 Proof: 11:00 is 4345.1, 12:30 is 4348.6)
        console.log("Capturing Portfolio Mode Day 2026-08-19 (Bug 1 verification: 11:00 vs 12:30)...");
        await pageCdp.evaluate(`
            (() => {
                const list = window.backtestData.daily_charts || window.backtestData.daily_chart_records;
                const idx = list.findIndex(d => d.date === "2026-08-19");
                if (idx >= 0) selectDay(idx);
            })()
        `);
        await sleep(600);
        await pageCdp.evaluate(`document.querySelector(".hud-grid").scrollIntoView({ behavior: "instant", block: "center" })`);
        await sleep(300);
        await pageCdp.captureElementScreenshot(".hud-grid", ARTIFACT_DIR + "/cdp_hud_0819.png");

        await pageCdp.evaluate(`document.getElementById("schematicSvgBox").scrollIntoView({ behavior: "instant", block: "center" })`);
        await sleep(300);
        await pageCdp.captureElementScreenshot("#schematicSvgBox", ARTIFACT_DIR + "/cdp_portfolio_0819_fsm.png");

        await pageCdp.evaluate(`switchInspectorView('candle')`);
        await sleep(500);
        await pageCdp.evaluate(`document.getElementById("candlestickContainer").scrollIntoView({ behavior: "instant", block: "center" })`);
        await sleep(300);
        await pageCdp.captureElementScreenshot("#candlestickContainer", ARTIFACT_DIR + "/cdp_chart_0819.png");
        await pageCdp.evaluate(`switchInspectorView('schematic')`);
        await sleep(300);

        // SCENARIO 2: Portfolio Mode - 2026-08-06 (S1 Scope Skipped at 11:00 + S2 Sweep Decay Limit Price @ 4250.4)
        console.log("Capturing Portfolio Mode Day 2026-08-06 (Decay @ 4250.4 label, S1 Skipped)...");
        await pageCdp.evaluate(`
            (() => {
                const list = window.backtestData.daily_charts || window.backtestData.daily_chart_records;
                const idx = list.findIndex(d => d.date === "2026-08-06");
                if (idx >= 0) selectDay(idx);
            })()
        `);
        await sleep(600);
        await pageCdp.evaluate(`document.querySelector(".hud-grid").scrollIntoView({ behavior: "instant", block: "center" })`);
        await sleep(300);
        await pageCdp.captureElementScreenshot(".hud-grid", ARTIFACT_DIR + "/cdp_hud_0806.png");

        await pageCdp.evaluate(`document.getElementById("schematicSvgBox").scrollIntoView({ behavior: "instant", block: "center" })`);
        await sleep(300);
        await pageCdp.captureElementScreenshot("#schematicSvgBox", ARTIFACT_DIR + "/cdp_portfolio_0806_fsm.png");

        // SCENARIO 3: Trade Table in Combined Portfolio Mode
        console.log("Capturing Portfolio Trade Table (Strategy 1 Scope, Strategy 2 Sweep & Flip)...");
        await pageCdp.evaluate(`document.getElementById("tableSection").scrollIntoView({ behavior: "instant", block: "start" })`);
        await sleep(400);
        await pageCdp.captureElementScreenshot("#tableSection", ARTIFACT_DIR + "/cdp_section_trade_table.png");

        // SCENARIO 4: XM Broker Mode Default Card & Zero Fees KPIs
        console.log("Capturing XM Broker Mode card & KPI analytics...");
        await pageCdp.evaluate(`window.scrollTo(0, 0)`);
        await sleep(400);
        await pageCdp.captureElementScreenshot(".studio-grid > .card:first-child", ARTIFACT_DIR + "/cdp_strategy_and_fee_modes.png");
        await pageCdp.captureElementScreenshot("#kpiContainer", ARTIFACT_DIR + "/cdp_zero_fees_kpi.png");

        console.log("✓ ALL USER SCENARIOS CAPTURED SUCCESSFULLY!");
    } finally {
        chromeProc.kill();
        try { fs.rmSync(USER_DATA_DIR, { recursive: true, force: true }); } catch (e) {}
    }
}

main().catch(err => {
    console.error("Fatal error:", err);
    process.exit(1);
});
