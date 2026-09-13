from src.blocks import extract_block_id


def test_extracts_positive_and_negative_block_ids():
    assert extract_block_id("received blk_123 from node") == "blk_123"
    assert extract_block_id("received blk_-456 from node") == "blk_-456"


def test_returns_none_without_block_id():
    assert extract_block_id("ordinary application log") is None
