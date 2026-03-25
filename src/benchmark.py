import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from config import load_config
from features import extract_features


def to_binary_anomaly(raw_pred):
    # sklearn outlier models return -1 for anomaly and 1 for normal.
    return pd.Series(raw_pred).map({-1: 1, 1: 0})


def main():
    project_root = Path(__file__).parent.parent
    config = load_config(project_root)

    data_path = project_root / "data" / "raw" / "HDFS_2k.log_structured.csv"
    df = pd.read_csv(data_path)
    X = extract_features(df)

    contamination = config["model"]["contamination"]

    models = {
        "IsolationForest": IsolationForest(
            contamination=contamination,
            random_state=config["model"]["random_state"],
        ),
        "LocalOutlierFactor": LocalOutlierFactor(contamination=contamination),
        "OneClassSVM": OneClassSVM(kernel="rbf", nu=contamination, gamma="scale"),
    }

    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        if name == "LocalOutlierFactor":
            raw_pred = model.fit_predict(X)
        else:
            model.fit(X)
            raw_pred = model.predict(X)

        elapsed_ms = (time.perf_counter() - start) * 1000
        pred = to_binary_anomaly(raw_pred)
        anomaly_count = int(pred.sum())

        rows.append(
            {
                "model": name,
                "anomaly_count": anomaly_count,
                "anomaly_rate_%": round((anomaly_count / len(df)) * 100, 2),
                "runtime_ms": round(elapsed_ms, 2),
            }
        )

    result = pd.DataFrame(rows).sort_values(by="runtime_ms")
    print("Benchmark on HDFS_2k.log_structured.csv")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
