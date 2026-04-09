from pathlib import Path
from time import perf_counter
from typing import List
import logging
from datetime import datetime, timezone
import uuid

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

        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: "Aptos", "Trebuchet MS", sans-serif;
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
            align-items: end;
            margin-bottom: 24px;
        }

        .title {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.5rem);
            letter-spacing: -0.03em;
        }

        .subtitle {
            margin: 10px 0 0;
            color: var(--muted);
            max-width: 760px;
            line-height: 1.55;
        }

        .status {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: rgba(17, 26, 51, 0.7);
            color: var(--muted);
            white-space: nowrap;
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
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
            padding: 18px;
        }

        .metric {
            grid-column: span 3;
            min-height: 136px;
        }

        .wide { grid-column: span 6; }
        .full { grid-column: span 12; }

        .label {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 10px;
            letter-spacing: 0.02em;
        }

        .value {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.04em;
        }

        .meta {
            margin-top: 12px;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.4;
        }

        .bars {
            display: grid;
            gap: 12px;
        }

        .bar-row {
            display: grid;
            gap: 8px;
        }

        .bar-head {
            display: flex;
            justify-content: space-between;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .bar-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.12);
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--accent-2), var(--accent));
            transition: width 0.35s ease;
        }

        @media (max-width: 960px) {
            .metric, .wide { grid-column: span 12; }
            .hero { flex-direction: column; align-items: start; }
        }
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <div>
                <h1 class="title">AI Log Anomaly Monitoring</h1>
                <p class="subtitle">Live dashboard for API traffic, inference latency, and anomaly activity. Metrics refresh automatically every 2 seconds.</p>
            </div>
            <div class="status"><span class="pulse"></span><span id="statusText">Live</span></div>
        </section>

        <section class="grid">
            <article class="card metric"><div class="label">Requests total</div><div class="value" id="requestsTotal">0</div><div class="meta">All API calls handled by the service.</div></article>
            <article class="card metric"><div class="label">Predict requests</div><div class="value" id="predictRequests">0</div><div class="meta">Single-log anomaly scoring calls.</div></article>
            <article class="card metric"><div class="label">Batch requests</div><div class="value" id="batchRequests">0</div><div class="meta">Bulk inference calls against log groups.</div></article>
            <article class="card metric"><div class="label">Avg inference ms</div><div class="value" id="avgInferenceMs">0.000</div><div class="meta">Mean runtime across prediction requests.</div></article>

            <article class="card wide">
                <div class="label">Latest observation</div>
                <div class="bars">
                    <div class="bar-row">
                        <div class="bar-head"><span>Last endpoint</span><strong id="lastEndpoint">-</strong></div>
                        <div class="bar-track"><div class="bar-fill" id="latencyBar" style="width: 0%;"></div></div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-head"><span>Last request latency</span><strong id="lastRequestMs">0.000 ms</strong></div>
                        <div class="bar-track"><div class="bar-fill" id="requestRateBar" style="width: 0%;"></div></div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-head"><span>Last anomaly rate</span><strong id="lastAnomalyRate">0.000</strong></div>
                        <div class="bar-track"><div class="bar-fill" id="anomalyBar" style="width: 0%; background: linear-gradient(90deg, #fb7185, #f59e0b);"></div></div>
                    </div>
                </div>
                <div class="meta" id="lastUpdatedAt">Waiting for the first request...</div>
            </article>

            <article class="card wide">
                <div class="label">Operational snapshot</div>
                <div class="meta" id="snapshotText">The service is ready. Send traffic to see metrics update live.</div>
            </article>
        </section>
    </main>

    <script>
        const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

        function setText(id, value) {
            document.getElementById(id).textContent = value;
        }

        function setWidth(id, value) {
            document.getElementById(id).style.width = `${clamp(value, 0, 100)}%`;
        }

        async function refreshMetrics() {
            try {
                const response = await fetch('/runtime-metrics', { cache: 'no-store' });
                const data = await response.json();

                setText('requestsTotal', data.requests_total ?? 0);
                setText('predictRequests', data.predict_requests ?? 0);
                setText('batchRequests', data.batch_predict_requests ?? 0);
                setText('avgInferenceMs', Number(data.avg_inference_ms ?? 0).toFixed(3));
                setText('lastEndpoint', data.last_endpoint ?? '-');
                setText('lastRequestMs', `${Number(data.last_request_ms ?? 0).toFixed(3)} ms`);
                setText('lastAnomalyRate', Number(data.last_anomaly_rate ?? 0).toFixed(3));
                setText('lastUpdatedAt', data.last_updated_at ?? '-');

                const latencyValue = Number(data.last_request_ms ?? 0);
                const anomalyValue = Number(data.last_anomaly_rate ?? 0);

                setWidth('latencyBar', latencyValue > 0 ? Math.min((latencyValue / 1000) * 100, 100) : 0);
                setWidth('requestRateBar', latencyValue > 0 ? Math.min((latencyValue / 250) * 100, 100) : 0);
                setWidth('anomalyBar', Math.min(anomalyValue * 100, 100));

                const summary = `Last request on ${data.last_endpoint ?? '-'} took ${Number(data.last_request_ms ?? 0).toFixed(3)} ms.`;
                setText('snapshotText', summary);
                setText('statusText', 'Live');
            } catch (error) {
                setText('statusText', 'Offline');
            }
        }

        refreshMetrics();
        setInterval(refreshMetrics, 2000);
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
