import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
import joblib

project_root = Path(__file__).parent.parent

#load_data
data_path = project_root /"data" / "raw" / "HDFS_2k.log_structured.csv"
df = pd.read_csv(data_path)

# ====== Version 1: Using basic features ======
# use log message length as a feature
df["log_length"] = df["Content"].astype(str).apply(len)

X = df[["log_length"]]

# ===== Train Isolation Forest ======
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# ===== Save the model ======
model_path = project_root / "models" /"anomaly_model.pkl"
joblib.dump(model, model_path)

print(f"Model trained and saved to {model_path}")