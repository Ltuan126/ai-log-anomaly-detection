from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "model": {
        "contamination": 0.05,
        "random_state": 42,
    },
    "data": {
        "raw_path": "data/raw/HDFS_2k.log_structured.csv",
        "model_path": "models/anomaly_model.pkl",
    },
    "mlflow": {
        "tracking_uri": "http://127.0.0.1:5000",
        "registry_uri": "",
        "experiment_name": "log_anomaly_detection",
        "run_name": "isolation_forest_train",
    },
}


def load_config(project_root: Path) -> dict:
    config_path = project_root / "configs" / "config.yaml"
    if not config_path.exists():
        return DEFAULT_CONFIG

    with config_path.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    model_cfg = config_data.get("model", {})
    data_cfg = config_data.get("data", {})
    mlflow_cfg = config_data.get("mlflow", {})

    merged = {
        "model": {
            "contamination": model_cfg.get("contamination", DEFAULT_CONFIG["model"]["contamination"]),
            "random_state": model_cfg.get("random_state", DEFAULT_CONFIG["model"]["random_state"]),
        },
        "data": {
            "raw_path": data_cfg.get("raw_path", DEFAULT_CONFIG["data"]["raw_path"]),
            "model_path": data_cfg.get("model_path", DEFAULT_CONFIG["data"]["model_path"]),
        },
        "mlflow": {
            "tracking_uri": mlflow_cfg.get("tracking_uri", DEFAULT_CONFIG["mlflow"]["tracking_uri"]),
            "registry_uri": mlflow_cfg.get("registry_uri", DEFAULT_CONFIG["mlflow"]["registry_uri"]),
            "experiment_name": mlflow_cfg.get("experiment_name", DEFAULT_CONFIG["mlflow"]["experiment_name"]),
            "run_name": mlflow_cfg.get("run_name", DEFAULT_CONFIG["mlflow"]["run_name"]),
        },
    }
    return merged
