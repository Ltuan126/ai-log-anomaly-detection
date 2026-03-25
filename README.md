# 🔍 AI Log Anomaly Detection

An unsupervised machine learning system that detects anomalies in HDFS system logs using Isolation Forest.

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This project applies **Isolation Forest** — an unsupervised anomaly detection algorithm — to identify abnormal entries in HDFS (Hadoop Distributed File System) logs. No labeled data required.

| Metric | Value |
|--------|-------|
| Dataset size | 2,000 log entries |
| Anomalies detected | 91 (~4.55%) |
| Algorithm | Isolation Forest |
| Contamination rate | 5% |
| Feature used | Log message content length |

---

## 🗂️ Project Structure

```
ai-log-anomaly-detection/
├── configs/
│   └── config.yaml          # Model hyperparameters & paths
├── data/
│   ├── raw/
│   │   └── HDFS_2k.log_structured.csv   # Raw HDFS log data
│   └── processed/           # Processed data (auto-generated)
├── models/
│   └── anomaly_model.pkl    # Trained Isolation Forest model
├── notebooks/
│   └── exploration.ipynb    # Data exploration & analysis
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load & preprocess log data
│   ├── detect.py            # Run anomaly detection
│   ├── train.py             # Train and save model
│   └── utils.py             # Helper functions
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ltuan126/ai-log-anomaly-detection.git
cd ai-log-anomaly-detection
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the model

```bash
python src/train.py
```

Model will be saved to `models/anomaly_model.pkl`.

### Run anomaly detection

```bash
python src/detect.py
```

Sample output:

```
   Content                                            anomaly
0  PacketResponder 1 for block blk_38865049064139...       1
1  PacketResponder 0 for block blk_-6952295868487...       0
2  BLOCK* NameSystem.addStoredBlock: blockMap upd...       0
...

Anomaly counts: 91
```

---

## 🧠 How It Works

```
Raw HDFS Logs
      ↓
Feature Extraction (log message length)
      ↓
Isolation Forest (contamination=0.05)
      ↓
Anomaly Score → Label (0: Normal, 1: Anomaly)
```

**Isolation Forest** works by randomly isolating observations. Anomalous points are easier to isolate (shorter path in the tree), resulting in a lower anomaly score.

---

## 📦 Tech Stack

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Isolation Forest model |
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical computation |
| `joblib` | Model serialization (.pkl) |
| `pyyaml` | Config file parsing |

---

## 🔮 Roadmap

- [ ] Add more features (log level, timestamp pattern, block ID)
- [ ] Compare multiple algorithms (LOF, One-Class SVM, Autoencoder)
- [ ] Track experiments with MLflow
- [ ] Build REST API with FastAPI
- [ ] Interactive UI with Streamlit
- [ ] Containerize with Docker

---

## 📄 Dataset

Uses the **HDFS log dataset** — a public benchmark dataset for log-based anomaly detection.

- Format: Structured CSV
- Size: 2,000 log entries (sample)
- Source: [Loghub](https://github.com/logpai/loghub)

---

## 👤 Author

**Ltuan126** — [github.com/Ltuan126](https://github.com/Ltuan126)