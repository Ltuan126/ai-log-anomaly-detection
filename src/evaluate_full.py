"""
Train + evaluate on the FULL HDFS_v1 dataset (575,061 blocks), using the
official pre-computed event-count matrix that ships inside loghub's
HDFS_v1.zip (data/raw/preprocessed/Event_occurrence_matrix.csv -- 29 event
types x every block, with the true Success/Fail label already attached).

This is the same feature representation as evaluate_events.py /
evaluate_supervised.py (event counts per block), just at full dataset scale
instead of the 100k-line partial sample. Comparing against those results
isolates how much of the remaining gap was "not enough data".

Usage:
    python src/evaluate_full.py
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
    from src.train import resolve_store_uri
except ModuleNotFoundError:
    from config import load_config
    from train import resolve_store_uri


def load_full_matrix(project_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    path = project_root / "data" / "raw" / "preprocessed" / "Event_occurrence_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- extract HDFS_v1.zip into data/raw/ first.")

    df = pd.read_csv(path)
    event_cols = [c for c in df.columns if c.startswith("E") and c[1:].isdigit()]
    X = df[event_cols].fillna(0)
    y = (df["Label"].str.strip() == "Fail").astype(int)
    return X, y


def eval_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_test, y_score)

    return model, {
        "model": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_test_anomalies": int(y_test.sum()),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def main() -> None:
    project_root = Path(__file__).parent.parent
    config = load_config(project_root)

    X, y = load_full_matrix(project_root)
    print(f"Loaded {len(X)} blocks, {y.sum()} anomalies ({y.mean():.2%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config["model"]["random_state"], stratify=y
    )

    results = []
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=config["model"]["random_state"], n_jobs=-1
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

    results_path = project_root / "data" / "processed" / "full_dataset_eval.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved all results to {results_path}")
    print(f"Best model: {best_name} (F1={best_metrics['f1']})")

    model_path = project_root / "models" / "anomaly_model_full.pkl"
    joblib.dump(
        {"model": best_model, "feature_columns": list(X.columns), "model_name": best_name},
        model_path,
    )
    print(f"Saved best model to {model_path}")

    print("\n=== Comparison across all experiments ===")
    print(f"{'experiment':<30}{'precision':<12}{'recall':<10}{'f1':<10}{'roc_auc':<10}")
    prior = [
        ("baseline (2k, text feat)", 0.08, 0.1176, 0.0952, 0.5519),
        ("event-count (2k)", 0.12, 0.0441, 0.0645, 0.5104),
        ("event-count (100k, IF)", 0.376, 0.4696, 0.4176, 0.7307),
        ("supervised (100k, LR)", 1.0, 0.4468, 0.6176, 0.7368),
        (f"supervised (full 575k, {best_name})", best_metrics["precision"], best_metrics["recall"], best_metrics["f1"], best_metrics["roc_auc"]),
    ]
    for name, p, r, f1, auc in prior:
        print(f"{name:<30}{p:<12}{r:<10}{f1:<10}{auc:<10}")

    try:
        tracking_uri = resolve_store_uri(project_root, config["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        for metrics in results:
            with mlflow.start_run(run_name=f"full_dataset_{metrics['model']}"):
                mlflow.log_params({"model_type": metrics["model"], "feature_set": "event_counts_full_575k"})
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
                )
        print("Logged to MLflow.")
    except Exception as exc:
        print(f"(Skipped MLflow logging: {exc})")


if __name__ == "__main__":
    main()
