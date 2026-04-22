import os
import warnings
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import load_config
from features import FEATURE_COLUMNS, extract_features


def resolve_store_uri(project_root: Path, raw_uri: str) -> str:
    if "://" in raw_uri:
        return raw_uri

    store_path = Path(raw_uri)
    if not store_path.is_absolute():
        store_path = project_root / store_path
    return store_path.resolve().as_uri()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*filesystem tracking backend.*",
        category=FutureWarning,
    )

    project_root = Path(__file__).parent.parent
    config = load_config(project_root)

    data_path = project_root / config["data"]["raw_path"]
    df = pd.read_csv(data_path)
    X = extract_features(df)

    model = IsolationForest(
        contamination=config["model"]["contamination"],
        random_state=config["model"]["random_state"],
    )

    tracking_uri_raw = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    tracking_uri = resolve_store_uri(project_root, tracking_uri_raw)
    mlflow.set_tracking_uri(tracking_uri)

    registry_uri_raw = os.getenv("MLFLOW_REGISTRY_URI", config["mlflow"].get("registry_uri") or tracking_uri_raw)
    registry_uri = resolve_store_uri(project_root, registry_uri_raw)
    mlflow.set_registry_uri(registry_uri)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        model.fit(X)
        raw_pred = model.predict(X)
        anomaly_count = int((raw_pred == -1).sum())
        anomaly_rate = anomaly_count / len(raw_pred) if len(raw_pred) else 0.0

        model_path = project_root / config["data"]["model_path"]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, model_path)

        mlflow.log_params(
            {
                "model_type": "IsolationForest",
                "contamination": config["model"]["contamination"],
                "random_state": config["model"]["random_state"],
                "n_rows": len(df),
                "n_features": X.shape[1],
            }
        )
        mlflow.log_metrics(
            {
                "train_anomaly_count": anomaly_count,
                "train_anomaly_rate": anomaly_rate,
            }
        )
        mlflow.log_artifact(str(model_path), artifact_path="models")

    print(f"Model trained and saved to {model_path}")
    print(f"Features used: {', '.join(FEATURE_COLUMNS)}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow registry URI: {registry_uri}")


if __name__ == "__main__":
    main()
