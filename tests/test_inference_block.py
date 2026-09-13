import numpy as np

from src import inference_block


class FakeMatcher:
    def match(self, line):
        return "E1" if line.startswith("known") else None


class FakeModel:
    def predict(self, frame):
        return np.array([int(value > 1) for value in frame["E1"]])

    def predict_proba(self, frame):
        scores = np.array([0.9 if value > 1 else 0.1 for value in frame["E1"]])
        return np.column_stack([1 - scores, scores])


def test_block_prediction_returns_quality_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(inference_block, "get_matcher", lambda _: FakeMatcher())
    monkeypatch.setattr(
        inference_block,
        "load_block_model",
        lambda _: (FakeModel(), ["E1"], "random_forest"),
    )
    result = inference_block.predict_blocks_from_lines(
        ["known blk_1", "known blk_1", "unknown blk_1", "without an id"],
        tmp_path,
    )
    assert result["total_blocks"] == 1
    assert result["anomaly_block_count"] == 1
    assert result["matched_event_rate"] == 0.6667
    assert result["model_name"] == "random_forest"
    assert result["model_version"] == "hdfs-v1-rf-1"
    assert result["insufficient_context"] is False


def test_warns_when_every_block_has_too_little_context(monkeypatch, tmp_path):
    monkeypatch.setattr(inference_block, "get_matcher", lambda _: FakeMatcher())
    monkeypatch.setattr(
        inference_block,
        "load_block_model",
        lambda _: (FakeModel(), ["E1"], "random_forest"),
    )
    result = inference_block.predict_blocks_from_lines(["known blk_1"], tmp_path)
    assert result["insufficient_context"] is True
    assert result["confidence_warning"]
