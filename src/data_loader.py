import pandas as pd
from pathlib import Path

def load_logs(path):
    df = pd.read_csv(path)
    return df

# Lấy đường dẫn tuyệt đối của thư mục gốc project
project_root = Path(__file__).parent.parent
csv_path = project_root / "data" / "raw" / "HDFS_2k.log_structured.csv"

if __name__ == "__main__":
    df = load_logs(csv_path)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns)
    print(df.head())