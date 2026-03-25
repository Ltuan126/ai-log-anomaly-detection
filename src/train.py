import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
import joblib
from config import load_config
from features import FEATURE_COLUMNS, extract_features

project_root = Path(__file__).parent.parent
config = load_config(project_root)

#load_data
data_path = project_root /"data" / "raw" / "HDFS_2k.log_structured.csv"
df = pd.read_csv(data_path)

# Build feature matrix
X = extract_features(df)

# ===== Train Isolation Forest ======
model = IsolationForest(
	contamination=config["model"]["contamination"],
	random_state=config["model"]["random_state"],
)
model.fit(X)

# ===== Save the model ======
model_path = project_root / "models" /"anomaly_model.pkl"
joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, model_path)

print(f"Model trained and saved to {model_path}")
print(f"Features used: {', '.join(FEATURE_COLUMNS)}")