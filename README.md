# AI Log Anomaly Detection

An unsupervised machine learning project for detecting anomalies in HDFS logs.

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)](https://scikit-learn.org)

## Project Status

Current version has been upgraded from a single-feature baseline to a multi-feature pipeline.

| Item | Current Status |
|------|----------------|
| Dataset | HDFS_2k structured logs (2,000 rows) |
| Main model | Isolation Forest |
| Feature set | 7 text-derived features |
| Config | YAML-based model configuration |
| Extra analysis | Benchmark script for 3 algorithms |

## What Is Implemented

- Multi-feature extraction module in src/features.py
- Config loader in src/config.py
- Training pipeline updated in src/train.py
- Detection pipeline updated in src/detect.py
- Benchmark comparison in src/benchmark.py
- FastAPI service with structured logs, Prometheus metrics, and live dashboard

### Feature Set (v2)

The model now uses these features from log content:

1. log_length
2. word_count
3. digit_count
4. uppercase_ratio
5. punctuation_count
6. has_block_id
7. keyword_hits

## Project Structure

```text
ai-log-anomaly-detection/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── HDFS_2k.log_structured.csv
│   └── processed/
├── models/
│   └── anomaly_model.pkl
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── config.py
│   ├── data_loader.py
│   ├── detect.py
│   ├── features.py
│   ├── inference.py
│   ├── train.py
│   ├── observability.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Setup

### 1) Clone and enter project

```bash
git clone https://github.com/Ltuan126/ai-log-anomaly-detection.git
cd ai-log-anomaly-detection
```

### 2) Create and activate virtual environment

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Configuration file: configs/config.yaml

```yaml
model:
  contamination: 0.05
  random_state: 42

data:
  raw_path: data/raw/HDFS_2k.log_structured.csv
  model_path: models/anomaly_model.pkl
```

## Run

### 1) Train model

```bash
python src/train.py
```

Expected output includes model path and feature list.

### 2) Detect anomalies

```bash
python src/detect.py
```

Output:
- First 20 rows with anomaly label
- Total anomaly count

### 3) Compare algorithms (benchmark)

```bash
python src/benchmark.py
```

This compares:
- IsolationForest
- LocalOutlierFactor
- OneClassSVM

And prints:
- anomaly_count
- anomaly_rate_%
- runtime_ms

### 4) Start API and monitoring dashboard

```bash
uvicorn app.main:app --reload
```

Open these endpoints:
- `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/dashboard` for the live dashboard
- `http://127.0.0.1:8000/health` for health checks
- `http://127.0.0.1:8000/metrics` for Prometheus scraping
- `http://127.0.0.1:8000/runtime-metrics` for the dashboard JSON payload

### 5) Start full monitoring stack (API + Prometheus + Grafana)

```bash
docker compose -f docker-compose.monitoring.yml up --build
```

Monitoring URLs:
- `http://127.0.0.1:8000/dashboard` for the built-in real-time dashboard
- `http://127.0.0.1:9090` for Prometheus UI
- `http://127.0.0.1:3000` for Grafana UI (default login: `admin` / `admin`)

Grafana automatically loads:
- Prometheus datasource
- `Log Anomaly Monitoring` dashboard

## How It Works

```text
Raw HDFS Logs
      ↓
Feature Extraction (7 features)
      ↓
Model Training / Inference
      ↓
Anomaly Label (0: normal, 1: anomaly)
```

Notes:
- Outlier models return -1 for anomaly and 1 for normal.
- Project maps them to 1 (anomaly) and 0 (normal).

## Tech Stack

| Library | Purpose |
|---------|---------|
| pandas | Data loading and feature processing |
| scikit-learn | IsolationForest, LOF, OneClassSVM |
| joblib | Model serialization |
| pyyaml | YAML config loading |
| numpy | Numeric operations |
| mlflow | Experiment tracking |
| prometheus-client | API monitoring metrics |

## Dataset

- Name: HDFS log dataset (structured sample)
- Size: 2,000 rows
- Format: CSV
- Source: https://github.com/logpai/loghub

## Next Steps

- Add alerting rules (latency/error-rate/anomaly-rate thresholds)
- Add persistent volumes for Prometheus and Grafana data
- Add CI pipeline for lint/test/build and image publishing
- Add request authentication and rate limiting for API endpoints

## Author

Ltuan126✨
