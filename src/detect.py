import pandas as pd
from pathlib import Path
from inference import predict_from_contents

#root project
project_root = Path(__file__).parent.parent

#load_data
data_path = project_root / "data" / "raw" / "HDFS_2k.log_structured.csv"
df = pd.read_csv(data_path)

#predict
pred, _ = predict_from_contents(df["Content"].astype(str).tolist(), project_root)
df["anomaly"] = pred

#print result
print(df[["Content", "anomaly"]].head(20))
print("\nAnomaly counts:", df["anomaly"].sum())