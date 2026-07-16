"""
Evaluate the anomaly detection model against ground-truth block labels.

Why this exists
----------------
train.py / detect.py only report anomaly_rate (how many lines got flagged),
which says nothing about whether the flagged lines are *actually* anomalous.
This script joins model predictions with the real HDFS labels (Normal /
Anomaly per block_id) and reports precision, recall, F1 and a confusion
matrix -- the baseline numbers needed before any feature/model change can
be judged as an improvement or not.

Label source
------------
Download `anomaly_label.csv` (BlockId, Label) for the HDFS dataset from:
    https://github.com/logpai/loglizer/blob/master/data/HDFS/anomaly_label.csv
(this is the companion label file for the same HDFS_2k demo sample used in
this project) and place it at: data/raw/anomaly_label.csv

Usage
-----
    python src/evaluate.py
"""

import json
import warnings
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.config import load_config
    from src.inference import load_model_bundle
    from src.features import extract_features
    from src.blocks import extract_block_id
except ModuleNotFoundError:
    from config import load_config
    from inference import load_model_bundle
    from features import extract_features
    from blocks import extract_block_id


def load_labels(label_path: Path) -> pd.DataFrame:
    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file not found at {label_path}.\n"
            "Download it from "
            "https://github.com/logpai/loglizer/blob/master/data/HDFS/anomaly_label.csv "
            "and save it there (columns: BlockId,Label)."
        )
    labels = pd.read_csv(label_path)
    labels["true_anomaly"] = (labels["Label"].str.strip().str.lower() == "anomaly").astype(int)
    return labels[["BlockId", "true_anomaly"]]


def evaluate(project_root: Path) -> dict:
    config = load_config(project_root)

    data_path = project_root / config["data"]["raw_path"]
    label_path = project_root / "data" / "raw" / "anomaly_label.csv"

    df = pd.read_csv(data_path)
    df["block_id"] = df["Content"].apply(extract_block_id)

    model, feature_columns = load_model_bundle(project_root)
    X = extract_features(df)[feature_columns]
    raw_pred = model.predict(X)
    df["line_anomaly"] = [1 if p == -1 else 0 for p in raw_pred]

    # Anomaly score: lower decision_function output = more anomalous.
    # Flip sign so higher score = more anomalous, for ROC-AUC.
    if hasattr(model, "decision_function"):
        df["line_score"] = -model.decision_function(X)
    else:
        df["line_score"] = df["line_anomaly"]

    unmatched = df["block_id"].isna().sum()
    if unmatched:
        warnings.warn(f"{unmatched} rows had no block_id and were dropped from block-level eval.")
    df = df.dropna(subset=["block_id"])

    # Block-level prediction: a block is anomalous if ANY of its lines is flagged.
    block_pred = df.groupby("block_id").agg(
        pred_anomaly=("line_anomaly", "max"),
        pred_score=("line_score", "max"),
    ).reset_index()

    labels = load_labels(label_path)
    merged = block_pred.merge(labels, left_on="block_id", right_on="BlockId", how="inner")

    n_labeled_blocks = len(merged)
    if n_labeled_blocks == 0:
        raise ValueError(
            "No overlap between block_ids in the log sample and the label file. "
            "Check that anomaly_label.csv matches this dataset's block ids."
        )

    y_true = merged["true_anomaly"]
    y_pred = merged["pred_anomaly"]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        auc = roc_auc_score(y_true, merged["pred_score"])
    except ValueError:
        auc = float("nan")  # only one class present

    metrics = {
        "n_blocks_total_in_sample": int(df["block_id"].nunique()),
        "n_blocks_matched_to_labels": n_labeled_blocks,
        "n_true_anomaly_blocks": int(y_true.sum()),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc == auc else None,  # NaN check
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }
    return metrics


def log_to_mlflow(project_root: Path, config: dict, metrics: dict) -> None:
    try:
        from src.train import resolve_store_uri
    except ModuleNotFoundError:
        from train import resolve_store_uri

    tracking_uri = resolve_store_uri(project_root, config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="baseline_evaluation"):
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
        )


def main() -> None:
    project_root = Path(__file__).parent.parent
    config = load_config(project_root)

    metrics = evaluate(project_root)

    print("=== Baseline evaluation (block-level) ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    results_path = project_root / "data" / "processed" / "baseline_eval.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved to {results_path}")

    try:
        log_to_mlflow(project_root, config, metrics)
        print("Logged to MLflow.")
    except Exception as exc:  # MLflow server may not be running
        print(f"(Skipped MLflow logging: {exc})")


if __name__ == "__main__":
    main()
