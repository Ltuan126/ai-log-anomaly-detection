"""
Block-level event-count features.

The baseline features (log_length, digit_count, ...) look at surface text of
one log LINE at a time. That throws away the one signal HDFS anomaly
detection research actually relies on: which log *event templates* occur,
and how many times, within a block's lifecycle.

HDFS_2k.log_structured.csv already ships with EventId/EventTemplate columns
(produced by a log parser -- Drain -- when this demo file was generated
upstream by logpai/logparser). So "do log parsing" here means: stop
discarding those columns, and aggregate them into a block x event count
matrix -- the classic feature representation used in the loglizer
benchmarks (PCA / IsolationForest / LR papers on this exact dataset).
"""

import pandas as pd

try:
    from src.blocks import extract_block_id
except ModuleNotFoundError:
    from blocks import extract_block_id


def build_block_event_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One row per block_id, one column per EventId, values = event count in that block."""
    df = df.copy()
    df["block_id"] = df["Content"].apply(extract_block_id)
    df = df.dropna(subset=["block_id"])

    matrix = pd.crosstab(df["block_id"], df["EventId"])
    matrix.columns = [f"count_{c}" for c in matrix.columns]

    # Extra block-level signals that come for free once you've grouped by block.
    matrix["total_lines"] = matrix.sum(axis=1)
    matrix["distinct_events"] = (matrix.drop(columns=["total_lines"]) > 0).sum(axis=1)

    return matrix
