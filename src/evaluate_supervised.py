"""
IsolationForest (unsupervised) on event-count features didn't beat the
baseline (see evaluate_events.py results) -- ROC-AUC ~0.51, basically random.
That matches published loglizer benchmarks: IsolationForest is a weak
performer on HDFS event-count vectors; PCA / invariant mining / supervised
classifiers are what actually work well on this dataset.

Since we now have real labels (data/raw/anomaly_label.csv), there's no
reason to stay unsupervised. This trains a supervised classifier
(Logistic Regression, class-weighted for the ~3.4% anomaly rate) on the
same block-level event-count features, with a proper held-out test split
so the reported numbers aren't measured on the training data.

Usage:
    python src/evaluate_supervised.py
"""

import json
import warnings
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from src.config import load_config
    from src.features_events import build_block_event_matrix
    from src.train import resolve_store_uri
except ModuleNotFoundError:
    from config import load_config
    from features_events import build_block_event_matrix
    from train import resolve_store_uri


def load_labels(label_path: Path) -> pd.Series:
    labels = pd.read_csv(label_path)
    labels["true_anomaly"] = (labels["Label"].str.strip().str.lower() == "anomaly").astype(int)
    return labels.set_index("BlockId")["true_anomaly"]


def eval_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_test, y_score)
    except ValueError:
        auc = float("nan")

    return model, {
        "model": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_test_anomalies": int(y_test.sum()),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc == auc else None,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def main() -> None:
    project_root = Path(__file__).parent.parent
    config = load_config(project_root)

    data_path = project_root / "data" / "raw" / "HDFS_100k.log_structured.csv"
    label_path = project_root / "data" / "raw" / "anomaly_label.csv"

    df = pd.read_csv(data_path)
    X = build_block_event_matrix(df)
    labels = load_labels(label_path)
    y = labels.reindex(X.index)

    keep = y.notna()
    dropped = (~keep).sum()
    if dropped:
        warnings.warn(f"{dropped} blocks had no label and were dropped.")
    X, y = X[keep], y[keep].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config["model"]["random_state"], stratify=y
    )

    results = []
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=config["model"]["random_state"]
        ),
    }

    best_name, best_model, best_metrics = None, None, None
    for name, model in models.items():
        fitted, metrics = eval_model(name, model, X_train, y_train, X_test, y_test)
        results.append(metrics)
        print(f"\n=== {name} ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_name, best_model, best_metrics = name, fitted, metrics

    results_path = project_root / "data" / "processed" / "supervised_eval.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved all results to {results_path}")
    print(f"Best model: {best_name} (F1={best_metrics['f1']})")

    model_path = project_root / "models" / "anomaly_model_supervised.pkl"
    joblib.dump({"model": best_model, "feature_columns": list(X.columns), "model_name": best_name}, model_path)
    print(f"Saved best model to {model_path}")

    baseline_path = project_root / "data" / "processed" / "baseline_eval.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        print("\n=== Comparison vs unsupervised baseline (IsolationForest, text features) ===")
        print(f"{'metric':<10}{'baseline':<12}{'best_supervised':<16}")
        for k in ("precision", "recall", "f1", "roc_auc"):
            print(f"{k:<10}{baseline.get(k):<12}{best_metrics.get(k):<16}")

    try:
        tracking_uri = resolve_store_uri(project_root, config["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        for metrics in results:
            with mlflow.start_run(run_name=f"supervised_{metrics['model']}"):
                mlflow.log_params({"model_type": metrics["model"], "feature_set": "block_event_counts"})
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
                )
        print("Logged to MLflow.")
    except Exception as exc:
        print(f"(Skipped MLflow logging: {exc})")


if __name__ == "__main__":
    main()
