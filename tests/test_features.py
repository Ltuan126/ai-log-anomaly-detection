import pandas as pd

from src.features import FEATURE_COLUMNS, extract_features
from src.features_events import build_block_event_matrix


def test_line_features_are_complete_and_bounded():
    result = extract_features(pd.DataFrame({"Content": ["ERROR blk_12 timeout"]}))
    assert list(result.columns) == FEATURE_COLUMNS
    assert result.loc[0, "has_block_id"] == 1
    assert 0 <= result.loc[0, "uppercase_ratio"] <= 1
    assert result.loc[0, "keyword_hits"] == 2


def test_builds_one_event_count_row_per_block():
    source = pd.DataFrame(
        {
            "Content": ["a blk_1", "b blk_1", "c blk_2", "no block"],
            "EventId": ["E1", "E1", "E2", "E3"],
        }
    )
    result = build_block_event_matrix(source)
    assert result.loc["blk_1", "count_E1"] == 2
    assert result.loc["blk_1", "total_lines"] == 2
    assert result.loc["blk_1", "distinct_events"] == 1
    assert result.loc["blk_2", "count_E2"] == 1
