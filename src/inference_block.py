"""
Production inference for the block-level model (models/anomaly_model_full.pkl,
F1=0.999 on held-out data -- see src/evaluate_full.py).

Unlike src/inference.py (which scores one log LINE at a time with the
original weak text-feature model), this groups incoming lines by HDFS
block id, maps each line to an event template, builds the same
event-count feature vector used at training time, and classifies each
BLOCK. A single line carries almost no signal on its own (see the
evaluate_events.py findings on the 2k sample) -- this only produces
meaningful results when given enough lines to cover a block's lifecycle
(ideally a real log file, not one line at a time).
"""

from collections import defaultdict
from pathlib import Path
from typing import List

import joblib
import pandas as pd

try:
    from src.blocks import extract_block_id
    from src.templates import TemplateMatcher
except ModuleNotFoundError:
    from blocks import extract_block_id
    from templates import TemplateMatcher

_matcher_cache = {}
_model_cache = {}


def get_matcher(project_root: Path) -> TemplateMatcher:
    key = str(project_root)
    if key not in _matcher_cache:
        templates_path = project_root / "data" / "raw" / "preprocessed" / "HDFS.log_templates.csv"
        _matcher_cache[key] = TemplateMatcher(templates_path)
    return _matcher_cache[key]


def load_block_model(project_root: Path):
    key = str(project_root)
    if key not in _model_cache:
        model_path = project_root / "models" / "anomaly_model_full.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run `python src/evaluate_full.py` to train it "
                "(requires data/raw/preprocessed/Event_occurrence_matrix.csv)."
            )
        payload = joblib.load(model_path)
        _model_cache[key] = (payload["model"], payload["feature_columns"])
    return _model_cache[key]


def predict_blocks_from_lines(lines: List[str], project_root: Path) -> dict:
    """Group lines by block, classify each block, return a summary + per-block results."""
    matcher = get_matcher(project_root)
    model, feature_columns = load_block_model(project_root)

    block_events = defaultdict(lambda: {col: 0 for col in feature_columns})
    block_line_count: dict = defaultdict(int)
    unmatched_lines = 0
    lines_without_block_id = 0

    for line in lines:
        block_id = extract_block_id(line)
        if block_id is None:
            lines_without_block_id += 1
            continue
        block_line_count[block_id] += 1
        event_id = matcher.match(line)
        if event_id is not None and event_id in block_events[block_id]:
            block_events[block_id][event_id] += 1
        else:
            unmatched_lines += 1

    blocks = []
    if block_events:
        block_ids = list(block_events.keys())
        X = pd.DataFrame(
            [[block_events[bid][col] for col in feature_columns] for bid in block_ids],
            columns=feature_columns,
        )
        preds = model.predict(X)
        scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else [None] * len(preds)

        for bid, pred, score in zip(block_ids, preds, scores):
            blocks.append(
                {
                    "block_id": bid,
                    "n_lines": block_line_count[bid],
                    "anomaly": int(pred),
                    "anomaly_score": round(float(score), 4) if score is not None else None,
                }
            )

    anomaly_count = sum(b["anomaly"] for b in blocks)
    total_blocks = len(blocks)

    return {
        "total_lines": len(lines),
        "lines_without_block_id": lines_without_block_id,
        "unmatched_event_lines": unmatched_lines,
        "total_blocks": total_blocks,
        "anomaly_block_count": anomaly_count,
        "anomaly_rate": round(anomaly_count / total_blocks, 4) if total_blocks else 0.0,
        "blocks": blocks,
    }
