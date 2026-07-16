"""
Train + evaluate an IsolationForest on block-level event-count features,
and compare directly against the line-level baseline (data/processed/baseline_eval.json).

Same model family (IsolationForest), same contamination hyperparameter, same
labeled data, same eval methodology as evaluate.py -- the only thing that
changes is the feature representation. This isolates whether "better
features" is actually where the improvement comes from.

Usage:
    python src/evaluate_events.py
"""

import json
import warnings
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.config import load_config
    from src.features_events import build_block_event_matrix
    from src.train import resolve_store_uri
except ModuleNotFoundError:
    from config import load_config
    from features_events import build_block_event_matrix
    from train import resolve_store_uri


def load_labels(label_path: Path) -> pd.DataFrame:
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found at {label_path}")
    labels = pd.read_csv(label_path)
    labels["true_anomaly"] = (labels["Label"].str.strip().str.lower() == "anomaly").astype(int)
    return labels.set_index("BlockId")["true_anomaly"]


def evaluate(project_root: Path) -> dict:
    config = load_config(project_root)

    # Use the 100k-line sample (not the 2k baseline) so most blocks have their
    # full event lifecycle captured -- see data/raw/HDFS_100k.log_structured.csv.
    data_path = project_root / "data" / "raw" / "HDFS_100k.log_structured.csv"
    label_path = project_root / "data" / "raw" / "anomaly_label.csv"

    df = pd.read_csv(data_path)
    X = build_block_event_matrix(df)

    labels = load_labels(label_path)
    y_true = labels.reindex(X.index)

    missing = y_true.isna().sum()
    if missing:
        warnings.warn(f"{missing} blocks had no matching label and were dropped.")
    keep = y_true.notna()
    X, y_true = X[keep], y_true[keep].astype(int)

    model = IsolationForest(
        contamination=config["model"]["contamination"],
        random_state=config["model"]["random_state"],
    )
    model.fit(X)
    raw_pred = model.predict(X)
    y_pred = pd.Series([1 if p == -1 else 0 for p in raw_pred], index=X.index)
    y_score = pd.Series(-model.decision_function(X), index=X.index)  # higher = more anomalous

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    metrics = {
        "feature_set": "block_event_counts",
        "n_features": X.shape[1] - 2,  # exclude total_lines/distinct_events from the "event" count
        "n_blocks": len(X),
        "n_true_anomaly_blocks": int(y_true.sum()),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc == auc else None,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }

    model_path = project_root / "models" / "anomaly_model_event_features.pkl"
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, model_path)

    return metrics, config


def print_comparison(project_root: Path, new_metrics: dict) -> None:
    baseline_path = project_root / "data" / "processed" / "baseline_eval.json"
    if not baseline_path.exists():
        print("(No baseline_eval.json found to compare against -- run evaluate.py first.)")
        return

    baseline = json.loads(baseline_path.read_text())
    rows = [
        ("precision", baseline.get("precision"), new_metrics["precision"]),
        ("recall", baseline.get("recall"), new_metrics["recall"]),
        ("f1", baseline.get("f1"), new_metrics["f1"]),
        ("roc_auc", baseline.get("roc_auc"), new_metrics["roc_auc"]),
    ]
    print("\n=== Baseline (7 text features) vs Event-count features ===")
    print(f"{'metric':<10}{'baseline':<12}{'event_count':<12}{'delta':<10}")
    for name, old, new in rows:
        old_v = old if old is not None else float("nan")
        new_v = new if new is not None else float("nan")
        delta = new_v - old_v if old_v == old_v and new_v == new_v else float("nan")
        delta_str = f"{delta:+.4f}" if delta == delta else "n/a"
        print(f"{name:<10}{old_v:<12}{new_v:<12}{delta_str:<10}")


def main() -> None:
    project_root = Path(__file__).parent.parent
    metrics, config = evaluate(project_root)

    print("=== Event-count feature evaluation (block-level) ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    results_path = project_root / "data" / "processed" / "event_features_eval.json"
    results_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved to {results_path}")

    print_comparison(project_root, metrics)

    try:
        tracking_uri = resolve_store_uri(project_root, config["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="event_count_isolation_forest"):
            mlflow.log_params(
                {
                    "model_type": "IsolationForest",
                    "feature_set": "block_event_counts",
                    "contamination": config["model"]["contamination"],
                    "random_state": config["model"]["random_state"],
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
            )
        print("Logged to MLflow.")
    except Exception as exc:
        print(f"(Skipped MLflow logging: {exc})")


if __name__ == "__main__":
    main()
