import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import load_config
from features import FEATURE_COLUMNS, extract_features


def main() -> None:
    project_root = Path(__file__).parent.parent
    config = load_config(project_root)

    data_path = project_root / config["data"]["raw_path"]
    df = pd.read_csv(data_path)
    X = extract_features(df)

    model = IsolationForest(
        contamination=config["model"]["contamination"],
        random_state=config["model"]["random_state"],
    )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", str(project_root / config["mlflow"]["tracking_uri"]))
    mlflow.set_tracking_uri(tracking_uri)
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
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

    print(f"Model trained and saved to {model_path}")
    print(f"Features used: {', '.join(FEATURE_COLUMNS)}")
    print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
