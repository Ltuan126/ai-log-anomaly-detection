# AI Log Anomaly Detection

Block-level anomaly detection for HDFS logs. Detects which HDFS blocks had
an abnormal lifecycle (failed write, missing replica, etc.) from the log
lines they produced.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)](https://scikit-learn.org)

## Project Status

The project went through several iterations before landing on an approach
that's actually validated with real ground-truth labels, not just "runs
without crashing." Full write-up of the evaluation trail is below --
short version: text-derived line features + IsolationForest was close to
random guessing; grouping lines by block and using a supervised classifier
on event-count features gets F1 = 0.999 on held-out data.

| Item | Current Status |
|------|----------------|
| Dataset | Full HDFS_v1 (11,175,629 log lines / 575,061 labeled blocks) |
| Main model | Random Forest on block-level event-count features |
| Feature set | 29 event-template counts per block |
| Held-out F1 | 0.999 (precision 0.999, recall 0.999, ROC-AUC 1.0) |
| Config | YAML-based model configuration |
| Serving | FastAPI, block-level `/upload`, legacy line-level `/predict` |

### Evaluation trail (why the model looks the way it does)

Every step below is reproducible via the script listed; results are saved
as JSON in `data/processed/`.

| # | Experiment | Data | Features | Model | Precision | Recall | F1 | ROC-AUC | Script |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline | HDFS_2k (2,000 lines, ~1 line/block) | 7 text features (length, digit count, ...) | IsolationForest | 0.08 | 0.118 | 0.095 | 0.55 | `src/evaluate.py` |
| 2 | Event-count v1 | HDFS_2k | 14 event-template counts | IsolationForest | 0.12 | 0.044 | 0.065 | 0.51 | `src/evaluate_events.py` |
| 3 | Event-count v2 | HDFS_100k (100k lines, ~13 lines/block) | event-template counts | IsolationForest | 0.376 | 0.470 | 0.418 | 0.73 | `src/evaluate_events.py` |
| 4 | Supervised | HDFS_100k | event-template counts | Logistic Regression | 1.0 | 0.447 | 0.618 | 0.74 | `src/evaluate_supervised.py` |
| 5 | **Full dataset** | **HDFS_v1 (575,061 blocks)** | **29 event-template counts** | **Random Forest** | **0.999** | **0.999** | **0.999** | **1.0** | `src/evaluate_full.py` |

The jump from step 2/3 to step 5 is not a feature-engineering trick -- it's
that HDFS_2k and HDFS_100k are small excerpts of the full log where most
blocks only have 1-2 of their ~13 lifecycle log lines captured, so a
block-level feature (event counts) has almost no signal. Once the full
dataset is used, every block has its complete lifecycle trace and the
model has something real to learn from.

## What Is Implemented

- Line-level baseline pipeline (original): `src/features.py`, `src/train.py`, `src/detect.py`, `src/inference.py`
- Block-level pipeline (current, F1=0.999): `src/blocks.py` (block id extraction), `src/templates.py` (event template matcher), `src/features_events.py` / `src/evaluate_full.py` (event-count features + training), `src/inference_block.py` (production inference)
- Evaluation scripts with real ground-truth labels: `src/evaluate.py`, `src/evaluate_events.py`, `src/evaluate_supervised.py`, `src/evaluate_full.py`
- FastAPI service: `/upload` uses the block-level model; `/predict` and `/predict-batch` still use the original line-level model (see [Known limitations](#known-limitations))
- MLflow experiment tracking, Prometheus metrics, live dashboard

## Project Structure

```text
ai-log-anomaly-detection/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── HDFS_2k.log_structured.csv       # small demo/baseline sample
│   │   ├── anomaly_label.csv                # ground-truth labels, all 575,061 blocks
│   │   └── preprocessed/
│   │       ├── Event_occurrence_matrix.csv  # block x 29-event count matrix (from loghub HDFS_v1.zip)
│   │       └── HDFS.log_templates.csv       # the 29 event templates
│   └── processed/                           # eval results (JSON)
├── models/
│   ├── anomaly_model.pkl                    # legacy line-level model
│   └── anomaly_model_full.pkl               # production block-level model (F1=0.999)
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── blocks.py            # block id extraction
│   ├── templates.py         # event template matcher (production log parsing)
│   ├── features.py          # legacy line-level text features
│   ├── features_events.py   # block-level event-count features
│   ├── config.py
│   ├── data_loader.py
│   ├── train.py             # trains the legacy line-level model
│   ├── detect.py
│   ├── inference.py         # legacy line-level inference
│   ├── inference_block.py   # production block-level inference
│   ├── evaluate.py          # baseline eval (step 1 above)
│   ├── evaluate_events.py   # event-count eval, unsupervised (steps 2-3)
│   ├── evaluate_supervised.py  # supervised eval on 100k sample (step 4)
│   ├── evaluate_full.py     # supervised eval + training on full dataset (step 5, current model)
│   └── utils.py
├── app/
│   └── main.py
├── requirements.txt
├── scripts/
│   ├── start-mlflow-sqlite.ps1
│   └── train-with-mlflow.ps1
└── README.md
```

Note on `data/raw/`: the large raw files (`HDFS.log`, 1.47GB; `HDFS_v1.zip`,
186MB; `Event_traces.csv`, 125MB; `HDFS.npz`, 53MB) are **not** kept in the
repo or committed to git (see `.gitignore`) -- only the derived files
actually needed to retrain/evaluate (`Event_occurrence_matrix.csv`,
`anomaly_label.csv`, `HDFS.log_templates.csv`) are kept. To retrain from
scratch, re-download `HDFS_v1.zip` from
[Zenodo](https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1)
and extract it into `data/raw/`.

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

mlflow:
  tracking_uri: http://127.0.0.1:5000
  registry_uri: ""
  experiment_name: log_anomaly_detection
  run_name: isolation_forest_train
```

Start MLflow tracking server (SQLite backend):

```bash
./scripts/start-mlflow-sqlite.ps1
```

Equivalent manual command:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

## Run

### Reproduce the current production model (F1=0.999)

```bash
python src/evaluate_full.py
```

Requires `data/raw/preprocessed/Event_occurrence_matrix.csv` (see the note
in [Project Structure](#project-structure) about re-downloading `HDFS_v1.zip`
if it's missing). Trains + evaluates Logistic Regression and Random Forest,
saves the best one to `models/anomaly_model_full.pkl`, and writes
`data/processed/full_dataset_eval.json`.

### Legacy line-level pipeline (weaker, kept for `/predict`)

```bash
python src/train.py      # train
python src/detect.py     # detect
python src/benchmark.py  # compare IsolationForest / LOF / OneClassSVM
```

### Start API and monitoring dashboard

```bash
uvicorn app.main:app --reload
```

Open these endpoints:
- `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/dashboard` for the live dashboard
- `http://127.0.0.1:8000/health` for health checks
- `http://127.0.0.1:8000/metrics` for Prometheus scraping
- `http://127.0.0.1:8000/runtime-metrics` for the dashboard JSON payload
- `POST /upload` -- upload a `.log`/`.txt` file, lines are grouped by block and classified with the validated model (F1=0.999)
- `POST /predict`, `POST /predict-batch` -- single/multiple independent lines, legacy line-level model (see limitations below)

### Start full monitoring stack (API + Prometheus + Grafana)

```bash
docker compose -f docker-compose.monitoring.yml up --build
```

Monitoring URLs:
- `http://127.0.0.1:8000/dashboard` for the built-in real-time dashboard
- `http://127.0.0.1:9090` for Prometheus UI
- `http://127.0.0.1:3000` for Grafana UI (default login: `admin` / `admin`)

## How It Works

```text
Raw log lines
      ↓
Group lines by HDFS block id (src/blocks.py)
      ↓
Match each line to 1 of 29 event templates (src/templates.py)
      ↓
Build a 29-dim event-count vector per block (src/features_events.py)
      ↓
Random Forest classifier (models/anomaly_model_full.pkl)
      ↓
Per-block anomaly label (0: normal, 1: anomaly)
```

## Known limitations

- **`/predict` and `/predict-batch` still use the old, weak line-level
  model** (F1=0.095 on real labels). A single log line carries almost no
  signal for this problem -- the validated model needs a block's full set
  of lines to build a meaningful feature vector. These endpoints exist for
  low-latency single-line scoring where waiting for more context isn't an
  option, but their output should not be trusted the way `/upload`'s
  block-level output can be.
- **The live dashboard stream** replays random single lines through
  `/predict`, so it inherits the same limitation -- it's a visual demo,
  not a reliability signal.
- **`/upload` accuracy depends on how much of each block's lifecycle is in
  the uploaded file.** A file with only 1-2 lines per block will behave
  like the HDFS_2k baseline (near-random), not like the F1=0.999 model.
- The event template matcher (`src/templates.py`) is a regex match against
  the 29 known HDFS templates, not a general-purpose log parser (Drain) --
  it works for this dataset's fixed log format but won't generalize to
  other log sources without new templates.

## Tech Stack

| Library | Purpose |
|---------|---------|
| pandas | Data loading and feature processing |
| scikit-learn | IsolationForest, LogisticRegression, RandomForestClassifier |
| joblib | Model serialization |
| pyyaml | YAML config loading |
| numpy | Numeric operations |
| mlflow | Experiment tracking |
| prometheus-client | API monitoring metrics |
| fastapi / uvicorn | API service |

## Dataset

- Name: HDFS_v1 (Hadoop Distributed File System logs)
- Size: 11,175,629 log lines / 575,061 labeled blocks (16,838 anomalous, 2.93%)
- Source: [logpai/loghub](https://github.com/logpai/loghub), via [Zenodo](https://zenodo.org/records/8196385)
- License: CC-BY-4.0 (see loghub repo for citation requirements)

## Next Steps

- Improve recall further (currently misses ~0.1% of anomalies at 99.9% precision) with sequence-based features (event order, not just counts) or more training data via HDFS_v2/v3
- Give `/predict`/`/predict-batch` a documented "low confidence" flag instead of silently using the weak model
- Add alerting rules (latency/error-rate/anomaly-rate thresholds)
- Add persistent volumes for Prometheus and Grafana data
- Add CI pipeline for lint/test/build and image publishing
- Add request authentication and rate limiting for API endpoints

## Author

Ltuan126✨
