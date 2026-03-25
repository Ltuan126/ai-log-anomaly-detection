import pandas as pd


FEATURE_COLUMNS = [
    "log_length",
    "word_count",
    "digit_count",
    "uppercase_ratio",
    "punctuation_count",
    "has_block_id",
    "keyword_hits",
]


KEYWORDS = ("error", "warn", "fail", "exception", "timeout", "denied", "critical")


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric features from raw HDFS log text."""
    content = df["Content"].fillna("").astype(str)

    features = pd.DataFrame(index=df.index)
    features["log_length"] = content.str.len()
    features["word_count"] = content.str.split().str.len()
    features["digit_count"] = content.str.count(r"\d")
    features["punctuation_count"] = content.str.count(r"[^\w\s]")
    features["has_block_id"] = content.str.contains(r"\bblk_-?\d+\b", regex=True).astype(int)

    # Keep ratio bounded in [0, 1] and avoid division by zero.
    uppercase_count = content.str.count(r"[A-Z]")
    safe_length = features["log_length"].replace(0, 1)
    features["uppercase_ratio"] = uppercase_count / safe_length

    lower_content = content.str.lower()
    features["keyword_hits"] = sum(lower_content.str.count(keyword) for keyword in KEYWORDS)

    return features[FEATURE_COLUMNS]
