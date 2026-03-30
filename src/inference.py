from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import pandas as pd

try:
    from src.features import FEATURE_COLUMNS, extract_features
except ModuleNotFoundError:
    from features import FEATURE_COLUMNS, extract_features


def _to_binary_anomaly(raw_pred: Iterable[int]) -> List[int]:
    return [1 if x == -1 else 0 for x in raw_pred]


def load_model_bundle(project_root: Path):
    model_path = project_root / "models" / "anomaly_model.pkl"
    model_payload = joblib.load(model_path)
    if isinstance(model_payload, dict) and "model" in model_payload:
        model = model_payload["model"]
        feature_columns = model_payload.get("feature_columns", FEATURE_COLUMNS)
    else:
        model = model_payload
        feature_columns = FEATURE_COLUMNS

    return model, feature_columns


def predict_from_contents(contents: List[str], project_root: Path) -> Tuple[List[int], float]:
    model, feature_columns = load_model_bundle(project_root)
    df = pd.DataFrame({"Content": contents})
    X = extract_features(df)

    if hasattr(model, "n_features_in_") and model.n_features_in_ != X.shape[1]:
        raise ValueError(
            f"Model expects {model.n_features_in_} features, but got {X.shape[1]}. "
            "Please retrain by running: python src/train.py"
        )

    missing = [col for col in feature_columns if col not in X.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = X[feature_columns]
    raw_pred = model.predict(X)
    pred = _to_binary_anomaly(raw_pred)
    anomaly_rate = (sum(pred) / len(pred)) if pred else 0.0
    return pred, anomaly_rate
