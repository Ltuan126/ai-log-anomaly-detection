"""Shared helper for extracting HDFS block ids from raw log content."""

import re

BLOCK_ID_RE = re.compile(r"blk_-?\d+")


def extract_block_id(content: str) -> str | None:
    match = BLOCK_ID_RE.search(str(content))
    return match.group(0) if match else None
