from pathlib import Path
from time import perf_counter
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import predict_from_contents


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

runtime_metrics = {
    "requests_total": 0,
    "predict_requests": 0,
    "batch_predict_requests": 0,
    "total_inference_ms": 0.0,
}


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
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    runtime_metrics["total_inference_ms"] += (perf_counter() - start) * 1000
    return PredictResponse(content=payload.content, anomaly=pred[0])


@app.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest):
    runtime_metrics["requests_total"] += 1
    runtime_metrics["batch_predict_requests"] += 1
    start = perf_counter()
    try:
        pred, anomaly_rate = predict_from_contents(payload.contents, project_root)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    runtime_metrics["total_inference_ms"] += (perf_counter() - start) * 1000

    predictions = [
        PredictResponse(content=content, anomaly=label)
        for content, label in zip(payload.contents, pred)
    ]
    anomaly_count = sum(pred)
    return BatchPredictResponse(
        total=len(payload.contents),
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_rate, 4),
        predictions=predictions,
    )
