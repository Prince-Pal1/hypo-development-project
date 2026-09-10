const http = require("http");
const fs = require("fs");
const { spawn } = require("child_process");

const CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9223;
const USER_DATA_DIR = "/tmp/chrome_sec_" + Date.now();
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
        return res.result ? res.result.value : undefined;
    }

    async captureScreenshot(path) {
        const res = await this.send("Page.captureScreenshot", { format: "png" });
        fs.writeFileSync(path, Buffer.from(res.data, "base64"));
        console.log(`[Screenshot Saved] -> ${path}`);
    }

    close() { if (this.ws) this.ws.close(); }
}

async function run() {
    const chrome = spawn(CHROME_PATH, [
        "--headless=new",
        `--remote-debugging-port=${PORT}`,
        `--user-data-dir=${USER_DATA_DIR}`,
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-gpu",
        "--window-size=1600,1400",
        "about:blank"
    ], { stdio: "ignore" });

    try {
        let targets = null;
        for (let i = 0; i < 30; i++) {
            await sleep(300);
            try {
                targets = await fetchJson(`http://127.0.0.1:${PORT}/json/list`);
                if (targets && targets.length > 0) break;
            } catch (e) {}
        }
        const pageTarget = targets.find(t => t.type === "page") || targets[0];
        const client = new CDPClient(pageTarget.webSocketDebuggerUrl);
        await client.connect();

        await client.send("Page.enable");
        await client.send("Runtime.enable");
        await client.send("DOM.enable");

        await client.send("Page.navigate", { url: "http://127.0.0.1:8050" });

        // Wait for ready
        for (let i = 0; i < 30; i++) {
            await sleep(500);
            const r = await client.evaluate(`Boolean(window.backtestData && document.getElementById("tradeTableBody") && document.getElementById("tradeTableBody").children.length > 0)`);
            if (r) break;
        }

        // Switch to Strategy 2
        await client.evaluate(`(() => {
            setStrategyMode("range_scope");
            setSession("tokyo");
            runBacktest();
        })()`);

        for (let i = 0; i < 40; i++) {
            await sleep(400);
            const m = await client.evaluate(`window.backtestData ? window.backtestData.strategy_mode : null`);
            const dis = await client.evaluate(`document.getElementById("btnRun").disabled`);
            if (m === "range_scope" && !dis) break;
        }

        // Select 2026-08-04
        await client.evaluate(`(() => {
            const idx = window.backtestData.daily_charts.findIndex(d => d.date === "2026-08-04");
            if (idx >= 0) selectDay(idx);
        })()`);
        await sleep(400);

        // 1. Scroll to Equity Curve & Analytics Section and capture
        await client.evaluate(`document.getElementById("equitySection").scrollIntoView({ behavior: "instant", block: "start" })`);
        await sleep(300);
        await client.captureScreenshot(`${ARTIFACT_DIR}/cdp_section_equity_analytics.png`);

        // 2. Scroll to Day Inspector FSM Schematic and capture
        await client.evaluate(`document.getElementById("inspectorSection").scrollIntoView({ behavior: "instant", block: "start" })`);
        await sleep(300);
        await client.captureScreenshot(`${ARTIFACT_DIR}/cdp_section_fsm_schematic.png`);

        // 3. Scroll to Trade Table and capture
        await client.evaluate(`document.getElementById("tableSection").scrollIntoView({ behavior: "instant", block: "start" })`);
        await sleep(300);
        await client.captureScreenshot(`${ARTIFACT_DIR}/cdp_section_trade_table.png`);

        client.close();
        console.log("✓ ALL TARGETED SECTION SCREENSHOTS CAPTURED!");
    } finally {
        chrome.kill();
        try { fs.rmSync(USER_DATA_DIR, { recursive: true, force: true }); } catch (e) {}
    }
}

run().catch(console.error);
