from pathlib import Path
from time import perf_counter
from typing import List
import logging
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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

runtime_metrics = {
    "requests_total": 0,
    "predict_requests": 0,
    "batch_predict_requests": 0,
    "total_inference_ms": 0.0,
}


@app.middleware("http")
async def log_request_response(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = perf_counter()

    try:
        response = await call_next(request)
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
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    elapsed_ms = round((perf_counter() - start) * 1000, 3)
    response.headers["x-request-id"] = request_id
    logger.info(
        "Request completed",
        extra={
            "event": "request_completed",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
        },
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    request_count = runtime_metrics["predict_requests"] + runtime_metrics["batch_predict_requests"]
    avg_ms = (runtime_metrics["total_inference_ms"] / request_count) if request_count else 0.0
    return {
        **runtime_metrics,
        "avg_inference_ms": round(avg_ms, 3),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    runtime_metrics["requests_total"] += 1
    runtime_metrics["predict_requests"] += 1
    start = perf_counter()
    try:
        pred, _ = predict_from_contents([payload.content], project_root)
    except Exception as exc:
        logger.exception(
            "Prediction failed",
            extra={"event": "predict_failed", "batch_size": 1},
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    runtime_metrics["total_inference_ms"] += (perf_counter() - start) * 1000
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
    runtime_metrics["requests_total"] += 1
    runtime_metrics["batch_predict_requests"] += 1
    start = perf_counter()
    try:
        pred, anomaly_rate = predict_from_contents(payload.contents, project_root)
    except Exception as exc:
        logger.exception(
            "Batch prediction failed",
            extra={"event": "predict_batch_failed", "batch_size": len(payload.contents)},
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    runtime_metrics["total_inference_ms"] += (perf_counter() - start) * 1000

    predictions = [
        PredictResponse(content=content, anomaly=label)
        for content, label in zip(payload.contents, pred)
    ]
    anomaly_count = sum(pred)
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
