import pandas as pd
from pathlib import Path
import joblib

#root project
project_root = Path(__file__).parent.parent

#load_data
data_path = project_root / "data" / "raw" / "HDFS_2k.log_structured.csv"
df = pd.read_csv(data_path)

#Feature similar to training data
df["log_length"] = df["Content"].astype(str).apply(len)
X = df[["log_length"]]

#load model
model_path = project_root/ "models" / "anomaly_model.pkl"
model = joblib.load(model_path)

#predict
df["anomaly"] = model.predict(X)
#convert
# -1 = anomaly, 1 = normal 
df["anomaly"] = df["anomaly"].apply(lambda x : 1 if x == -1 else 0)

#print result
print(df[["Content", "anomaly"]].head(20))
print("\nAnomaly counts:", df["anomaly"].sum())