from pathlib import Path
from time import perf_counter
from typing import List
import logging
from datetime import datetime, timezone
import uuid
import csv
import random

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.inference import predict_from_contents
from src.utils import configure_logging


class PredictRequest(BaseModel):
    content: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    content: str
    anomaly: int


class BatchPredictRequest(BaseModel):
    contents: List[str] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    total: int
    anomaly_count: int
    anomaly_rate: float
    predictions: List[PredictResponse]


app = FastAPI(title="AI Log Anomaly Detection API", version="0.1.0")
project_root = Path(__file__).resolve().parent.parent
configure_logging()
logger = logging.getLogger("api")

REQUEST_COUNTER = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "path", "status_code"],
)
INFERENCE_REQUEST_COUNTER = Counter(
    "inference_requests_total",
    "Total inference requests by endpoint",
    ["endpoint"],
)
INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"],
)
ANOMALY_PREDICTIONS_TOTAL = Counter(
    "anomaly_predictions_total",
    "Total anomaly predictions produced by API",
)
BATCH_SIZE_HISTOGRAM = Histogram(
    "batch_size",
    "Batch size distribution for prediction endpoint",
    buckets=(1, 2, 5, 10, 20, 50, 100, 250, 500, 1000),
)

runtime_metrics = {
    "requests_total": 0,
    "predict_requests": 0,
    "batch_predict_requests": 0,
    "total_inference_ms": 0.0,
    "last_request_ms": 0.0,
    "last_endpoint": "-",
    "last_anomaly_count": 0,
    "last_anomaly_rate": 0.0,
    "last_updated_at": "-",
}


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI Log Anomaly Monitoring</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b1020;
            --panel: rgba(16, 24, 40, 0.82);
            --panel-strong: #111a33;
            --text: #ecf2ff;
            --muted: #9fb0d0;
            --accent: #6ee7b7;
            --accent-2: #60a5fa;
            --danger: #fb7185;
            --border: rgba(148, 163, 184, 0.2);
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            min-height: 100vh;
            font-family: 'Inter', 'Aptos', sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.3), transparent 28%),
                radial-gradient(circle at top right, rgba(110, 231, 183, 0.18), transparent 24%),
                linear-gradient(135deg, #08101f 0%, #0b1020 48%, #111a33 100%);
        }

        .shell {
            width: min(1200px, calc(100% - 32px));
            margin: 0 auto;
            padding: 32px 0 48px;
        }

        .hero {
            display: flex;
            justify-content: space-between;
            gap: 24px;
            align-items: flex-end;
            margin-bottom: 24px;
        }

        .title {
            font-size: clamp(2rem, 4vw, 3.5rem);
            letter-spacing: -0.03em;
            font-weight: 700;
        }

        .subtitle {
            margin: 10px 0 0;
            color: var(--muted);
            max-width: 760px;
            line-height: 1.55;
            font-size: 0.95rem;
        }

        .status {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 16px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: rgba(17, 26, 51, 0.7);
            color: var(--muted);
            white-space: nowrap;
            font-size: 0.9rem;
        }

        .pulse {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 0 0 rgba(110, 231, 183, 0.6);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(110, 231, 183, 0.55); }
            70% { box-shadow: 0 0 0 16px rgba(110, 231, 183, 0); }
            100% { box-shadow: 0 0 0 0 rgba(110, 231, 183, 0); }
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
            padding: 20px;
        }

        .metric { grid-column: span 3; min-height: 136px; }
        .wide { grid-column: span 6; }
        .full { grid-column: span 12; }

        .label {
            color: var(--muted);
            font-size: 0.8rem;
            margin-bottom: 10px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .value {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.04em;
        }

        .meta {
            margin-top: 12px;
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.4;
        }

        .bars { display: grid; gap: 14px; }
        .bar-row { display: grid; gap: 8px; }
        .bar-head { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.88rem; }
        .bar-track { width: 100%; height: 8px; border-radius: 999px; background: rgba(148, 163, 184, 0.12); overflow: hidden; }
        .bar-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--accent-2), var(--accent)); transition: width 0.35s ease; }

        /* ── Stream Section ────────────────────────────────────────── */
        .stream-section {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
            overflow: hidden;
        }

        .stream-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 24px;
            border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
            gap: 12px;
        }

        .stream-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
            font-size: 1rem;
        }

        .rec-dot {
            width: 9px; height: 9px; border-radius: 50%;
            background: var(--danger);
            animation: pulse-red 1.5s infinite;
        }

        @keyframes pulse-red {
            0%  { box-shadow: 0 0 0 0   rgba(251,113,133,0.7); }
            70% { box-shadow: 0 0 0 8px rgba(251,113,133,0);   }
            100%{ box-shadow: 0 0 0 0   rgba(251,113,133,0);   }
        }

        .stream-stats {
            display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
        }

        .stat-pill {
            display: flex; align-items: center; gap: 6px;
            background: rgba(148,163,184,0.07);
            border: 1px solid var(--border);
            border-radius: 999px; padding: 4px 14px;
            font-size: 0.82rem; color: var(--muted);
        }
        .stat-pill .num { color: var(--text); font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-left: 2px; }
        .stat-pill.danger .num { color: var(--danger); }
        .stat-pill.accent  .num { color: var(--accent);  }

        .btn-pause {
            background: rgba(148,163,184,0.09);
            border: 1px solid var(--border);
            color: var(--muted); border-radius: 8px;
            padding: 6px 16px; cursor: pointer;
            font-size: 0.85rem; transition: all 0.2s; font-family: inherit;
        }
        .btn-pause:hover { background: rgba(148,163,184,0.2); color: var(--text); }

        /* Log feed */
        .log-feed {
            height: 360px;
            overflow-y: auto;
            padding: 8px 0;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 0.8rem;
            scroll-behavior: smooth;
        }
        .log-feed::-webkit-scrollbar { width: 4px; }
        .log-feed::-webkit-scrollbar-track { background: transparent; }
        .log-feed::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.18); border-radius: 2px; }

        .log-entry {
            display: grid;
            grid-template-columns: 88px 1fr 108px;
            align-items: center;
            gap: 10px;
            padding: 6px 24px;
            border-left: 2px solid transparent;
            animation: slideIn 0.28s ease;
            transition: background 0.15s;
        }
        .log-entry:hover { background: rgba(148,163,184,0.04); }
        .log-entry.anomaly { background: rgba(251,113,133,0.055); border-left-color: var(--danger); }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-5px); }
            to   { opacity: 1; transform: translateY(0);    }
        }

        .log-time    { color: var(--muted); font-size: 0.75rem; white-space: nowrap; }
        .log-content { color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; opacity: 0.88; }
        .log-entry.anomaly .log-content { color: #fca5a5; }

        .badge {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 2px 10px; border-radius: 999px;
            font-size: 0.73rem; font-weight: 600; white-space: nowrap; justify-content: center;
        }
        .badge.normal  { background: rgba(110,231,183,0.13); color: var(--accent); border: 1px solid rgba(110,231,183,0.28); }
        .badge.anomaly { background: rgba(251,113,133,0.13); color: var(--danger); border: 1px solid rgba(251,113,133,0.28); }

        .stream-loading {
            display: flex; align-items: center; justify-content: center;
            height: 200px; color: var(--muted); gap: 12px; font-size: 0.9rem;
        }
        .spinner {
            width: 20px; height: 20px;
            border: 2px solid rgba(148,163,184,0.18);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .stream-footer {
            padding: 12px 24px;
            border-top: 1px solid var(--border);
            display: flex; align-items: center; gap: 14px;
        }
        .rate-label  { font-size: 0.78rem; color: var(--muted); white-space: nowrap; }
        .rate-track  { flex: 1; height: 5px; background: rgba(148,163,184,0.1); border-radius: 999px; overflow: hidden; }
        .rate-fill   { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--accent), var(--danger)); transition: width 0.5s ease; }
        .rate-value  { font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; color: var(--text); white-space: nowrap; min-width: 46px; text-align: right; }

        @media (max-width: 960px) {
            .metric, .wide { grid-column: span 12; }
            .hero { flex-direction: column; align-items: flex-start; }
            .log-entry { grid-template-columns: 76px 1fr 90px; }
        }
    </style>
</head>
<body>
<main class="shell">

    <section class="hero">
        <div>
            <h1 class="title">AI Log Anomaly Monitoring</h1>
            <p class="subtitle">Real-time anomaly detection on HDFS server logs using Isolation Forest. Stream below shows live inference on actual dataset logs.</p>
        </div>
        <div class="status"><span class="pulse"></span><span id="statusText">Live</span></div>
    </section>

    <section class="grid">
        <article class="card metric">
            <div class="label">Requests total</div>
            <div class="value" id="requestsTotal">0</div>
            <div class="meta">All API calls handled by the service.</div>
        </article>
        <article class="card metric">
            <div class="label">Predict requests</div>
            <div class="value" id="predictRequests">0</div>
            <div class="meta">Single-log anomaly scoring calls.</div>
        </article>
        <article class="card metric">
            <div class="label">Batch requests</div>
            <div class="value" id="batchRequests">0</div>
            <div class="meta">Bulk inference calls against log groups.</div>
        </article>
        <article class="card metric">
            <div class="label">Avg inference ms</div>
            <div class="value" id="avgInferenceMs">0.000</div>
            <div class="meta">Mean runtime across prediction requests.</div>
        </article>

        <article class="card wide">
            <div class="label">Latest observation</div>
            <div class="bars">
                <div class="bar-row">
                    <div class="bar-head"><span>Last endpoint</span><strong id="lastEndpoint">-</strong></div>
                    <div class="bar-track"><div class="bar-fill" id="latencyBar" style="width:0%"></div></div>
                </div>
                <div class="bar-row">
                    <div class="bar-head"><span>Last request latency</span><strong id="lastRequestMs">0.000 ms</strong></div>
                    <div class="bar-track"><div class="bar-fill" id="requestRateBar" style="width:0%"></div></div>
                </div>
                <div class="bar-row">
                    <div class="bar-head"><span>Last anomaly rate</span><strong id="lastAnomalyRate">0.000</strong></div>
                    <div class="bar-track"><div class="bar-fill" id="anomalyBar" style="width:0%;background:linear-gradient(90deg,#fb7185,#f59e0b)"></div></div>
                </div>
            </div>
            <div class="meta" id="lastUpdatedAt">Waiting for the first request...</div>
        </article>

        <article class="card wide">
            <div class="label">Operational snapshot</div>
            <div class="meta" id="snapshotText">The service is ready. Live stream is loading HDFS logs...</div>
        </article>
    </section>

    <!-- Live Stream -->
    <section class="stream-section">
        <div class="stream-header">
            <div class="stream-title">
                <span class="rec-dot"></span>
                Live Log Analysis Stream
            </div>
            <div class="stream-stats">
                <div class="stat-pill">
                    <span>Analyzed</span>
                    <span class="num" id="sessionTotal">0</span>
                </div>
                <div class="stat-pill danger">
                    <span>&#9888; Anomalies</span>
                    <span class="num" id="sessionAnomalies">0</span>
                </div>
                <div class="stat-pill accent">
                    <span>Rate</span>
                    <span class="num" id="sessionRate">0.00%</span>
                </div>
            </div>
            <button class="btn-pause" id="btnPause" onclick="togglePause()">&#9208; Pause</button>
        </div>

        <div class="log-feed" id="logFeed">
            <div class="stream-loading" id="streamLoading">
                <div class="spinner"></div>
                <span>Loading HDFS log samples...</span>
            </div>
        </div>

        <div class="stream-footer">
            <span class="rate-label">Session anomaly rate</span>
            <div class="rate-track"><div class="rate-fill" id="rateFill" style="width:0%"></div></div>
            <span class="rate-value" id="rateValueFooter">0.00%</span>
        </div>
    </section>

</main>

<script>
    // ── Helpers ──────────────────────────────────────────────────────
    const clamp = (v,mn,mx) => Math.min(Math.max(v,mn),mx);
    const setText  = (id,v) => { const el=document.getElementById(id); if(el) el.textContent=v; };
    const setWidth = (id,v) => { const el=document.getElementById(id); if(el) el.style.width=`${clamp(v,0,100)}%`; };

    // ── Metrics polling ──────────────────────────────────────────────
    async function refreshMetrics() {
        try {
            const data = await fetch('/runtime-metrics',{cache:'no-store'}).then(r=>r.json());
            setText('requestsTotal',  data.requests_total ?? 0);
            setText('predictRequests',data.predict_requests ?? 0);
            setText('batchRequests',  data.batch_predict_requests ?? 0);
            setText('avgInferenceMs', Number(data.avg_inference_ms ?? 0).toFixed(3));
            setText('lastEndpoint',   data.last_endpoint ?? '-');
            setText('lastRequestMs',  `${Number(data.last_request_ms ?? 0).toFixed(3)} ms`);
            setText('lastAnomalyRate',Number(data.last_anomaly_rate ?? 0).toFixed(3));
            setText('lastUpdatedAt',  data.last_updated_at ?? '-');
            const lat = Number(data.last_request_ms ?? 0);
            const ano = Number(data.last_anomaly_rate ?? 0);
            setWidth('latencyBar',    lat > 0 ? Math.min((lat/1000)*100,100) : 0);
            setWidth('requestRateBar',lat > 0 ? Math.min((lat/250)*100,100)  : 0);
            setWidth('anomalyBar',    Math.min(ano*100,100));
            setText('snapshotText', `Last request on ${data.last_endpoint??'-'} took ${Number(data.last_request_ms??0).toFixed(3)} ms.`);
            setText('statusText','Live');
        } catch { setText('statusText','Offline'); }
    }
    refreshMetrics();
    setInterval(refreshMetrics, 2000);

    // ── Live Stream ──────────────────────────────────────────────────
    let logs = [], idx = 0, paused = false;
    let sessionTotal = 0, sessionAnomalies = 0;

    function togglePause() {
        paused = !paused;
        document.getElementById('btnPause').innerHTML = paused ? '&#9654; Resume' : '&#9208; Pause';
    }

    function nowTime() {
        return new Date().toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
    }

    function addEntry(content, isAnomaly) {
        const feed = document.getElementById('logFeed');
        const el   = document.createElement('div');
        el.className = `log-entry ${isAnomaly ? 'anomaly' : 'normal'}`;
        const short  = content.length > 75 ? content.slice(0,75) + '\u2026' : content;
        el.innerHTML = `
            <span class="log-time">${nowTime()}</span>
            <span class="log-content" title="${content.replace(/"/g,'&quot;')}">${short}</span>
            <span class="badge ${isAnomaly?'anomaly':'normal'}">${isAnomaly?'&#9888; ANOMALY':'&#10003; NORMAL'}</span>`;
        feed.appendChild(el);
        while (feed.children.length > 100) feed.removeChild(feed.firstChild);
        feed.scrollTop = feed.scrollHeight;
    }

    function updateStats(isAnomaly) {
        sessionTotal++;
        if (isAnomaly) sessionAnomalies++;
        const rate = sessionTotal > 0 ? (sessionAnomalies / sessionTotal) * 100 : 0;
        setText('sessionTotal',    sessionTotal);
        setText('sessionAnomalies',sessionAnomalies);
        setText('sessionRate',     rate.toFixed(2) + '%');
        setText('rateValueFooter', rate.toFixed(2) + '%');
        setWidth('rateFill', Math.min(rate * 4, 100));
    }

    async function processNext() {
        if (paused || !logs.length) return;
        const content = logs[idx % logs.length];
        idx++;
        try {
            const res  = await fetch('/predict', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({content})
            });
            const data = await res.json();
            addEntry(content, data.anomaly === 1);
            updateStats(data.anomaly === 1);
        } catch {
            addEntry(content, false);
            updateStats(false);
        }
    }

    async function startStream() {
        try {
            const data = await fetch('/demo-logs').then(r=>r.json());
            logs = (data.logs || []).sort(() => Math.random() - 0.5);
        } catch {
            logs = [
                "Receiving block blk_-1608999687919862906 src: /10.250.10.6:51553 dest: /10.250.11.85:50010",
                "PacketResponder 1 for block blk_-1608999687919862906 terminating",
                "BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.11.85:50010 is added to blk_-1608999687919862906",
                "Exception in receiveBlock for block blk_7503483334871736937",
                "writeBlock blk_7503483334871736937 received exception java.io.IOException",
            ];
        }
        const loading = document.getElementById('streamLoading');
        if (loading) loading.remove();
        setInterval(processNext, 950);
    }

    startStream();
</script>
</body>
</html>
"""


@app.middleware("http")
async def log_request_response(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = perf_counter()
    response = None
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        logger.exception(
            "Unhandled API exception",
            extra={
                "event": "request_error",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        response = JSONResponse(status_code=500, content={"detail": "Internal server error"})

    elapsed_ms = round((perf_counter() - start) * 1000, 3)
    response.headers["x-request-id"] = request_id
    REQUEST_COUNTER.labels(
        method=request.method,
        path=request.url.path,
        status_code=str(status_code),
    ).inc()
    runtime_metrics["requests_total"] += 1
    runtime_metrics["last_request_ms"] = elapsed_ms
    runtime_metrics["last_endpoint"] = request.url.path
    runtime_metrics["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Request completed",
        extra={
            "event": "request_completed",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "latency_ms": elapsed_ms,
        },
    )
    return response


@app.get("/", response_class=HTMLResponse)
def dashboard_root():
    return DASHBOARD_HTML


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/runtime-metrics")
def runtime_stats():
    request_count = runtime_metrics["predict_requests"] + runtime_metrics["batch_predict_requests"]
    avg_ms = (runtime_metrics["total_inference_ms"] / request_count) if request_count else 0.0
    return {
        **runtime_metrics,
        "avg_inference_ms": round(avg_ms, 3),
    }


@app.get("/demo-logs")
def demo_logs():
    """Return a shuffled sample of real HDFS log lines for the live stream demo."""
    data_path = project_root / "data" / "raw" / "HDFS_2k.log_structured.csv"
    logs: list[str] = []
    if data_path.exists():
        try:
            with open(data_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    content = (row.get("Content") or "").strip()
                    if content:
                        logs.append(content)
        except Exception:
            pass
    if not logs:
        logs = [
            "Receiving block blk_-1608999687919862906 src: /10.250.10.6:51553 dest: /10.250.11.85:50010",
            "PacketResponder 1 for block blk_-1608999687919862906 terminating",
            "BLOCK* NameSystem.addStoredBlock: blockMap updated",
            "Exception in receiveBlock for block blk_7503483334871736937",
            "writeBlock blk_7503483334871736937 received exception java.io.IOException",
        ]
    sample = random.sample(logs, min(120, len(logs)))
    return {"logs": sample, "total": len(sample)}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    runtime_metrics["predict_requests"] += 1
    INFERENCE_REQUEST_COUNTER.labels(endpoint="predict").inc()
    BATCH_SIZE_HISTOGRAM.observe(1)
    start = perf_counter()
    try:
        pred, _ = predict_from_contents([payload.content], project_root)
    except Exception as exc:
        logger.exception(
            "Prediction failed",
            extra={"event": "predict_failed", "batch_size": 1},
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    elapsed = perf_counter() - start
    INFERENCE_LATENCY_SECONDS.labels(endpoint="predict").observe(elapsed)
    runtime_metrics["total_inference_ms"] += elapsed * 1000
    runtime_metrics["last_anomaly_count"] = int(pred[0])
    runtime_metrics["last_anomaly_rate"] = float(pred[0])
    ANOMALY_PREDICTIONS_TOTAL.inc(int(pred[0]))
    logger.info(
        "Prediction success",
        extra={
            "event": "predict_success",
            "batch_size": 1,
            "anomaly_count": int(pred[0]),
        },
    )
    return PredictResponse(content=payload.content, anomaly=pred[0])


@app.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest):
    runtime_metrics["batch_predict_requests"] += 1
    INFERENCE_REQUEST_COUNTER.labels(endpoint="predict_batch").inc()
    BATCH_SIZE_HISTOGRAM.observe(len(payload.contents))
    start = perf_counter()
    try:
        pred, anomaly_rate = predict_from_contents(payload.contents, project_root)
    except Exception as exc:
        logger.exception(
            "Batch prediction failed",
            extra={"event": "predict_batch_failed", "batch_size": len(payload.contents)},
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    elapsed = perf_counter() - start
    INFERENCE_LATENCY_SECONDS.labels(endpoint="predict_batch").observe(elapsed)
    runtime_metrics["total_inference_ms"] += elapsed * 1000

    predictions = [
        PredictResponse(content=content, anomaly=label)
        for content, label in zip(payload.contents, pred)
    ]
    anomaly_count = sum(pred)
    runtime_metrics["last_anomaly_count"] = anomaly_count
    runtime_metrics["last_anomaly_rate"] = round(anomaly_rate, 4)
    ANOMALY_PREDICTIONS_TOTAL.inc(anomaly_count)
    logger.info(
        "Batch prediction success",
        extra={
            "event": "predict_batch_success",
            "batch_size": len(payload.contents),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_rate, 4),
        },
    )
    return BatchPredictResponse(
        total=len(payload.contents),
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_rate, 4),
        predictions=predictions,
    )
