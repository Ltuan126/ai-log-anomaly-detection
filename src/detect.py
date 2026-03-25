import pandas as pd
from pathlib import Path
import joblib
from features import extract_features

#root project
project_root = Path(__file__).parent.parent

#load_data
data_path = project_root / "data" / "raw" / "HDFS_2k.log_structured.csv"
df = pd.read_csv(data_path)

#Feature similar to training data
X = extract_features(df)

#load model
model_path = project_root/ "models" / "anomaly_model.pkl"
model_payload = joblib.load(model_path)
if isinstance(model_payload, dict) and "model" in model_payload:
	model = model_payload["model"]
else:
	model = model_payload

if hasattr(model, "n_features_in_") and model.n_features_in_ != X.shape[1]:
	raise ValueError(
		f"Model expects {model.n_features_in_} features, but got {X.shape[1]}. "
		"Please retrain by running: python src/train.py"
	)

#predict
df["anomaly"] = model.predict(X)
#convert
# -1 = anomaly, 1 = normal 
df["anomaly"] = df["anomaly"].apply(lambda x : 1 if x == -1 else 0)

#print result
print(df[["Content", "anomaly"]].head(20))
print("\nAnomaly counts:", df["anomaly"].sum())