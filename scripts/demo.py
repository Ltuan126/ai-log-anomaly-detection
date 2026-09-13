"""Run the validated block-level model against a bundled demo log file."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_block import predict_blocks_from_lines  # noqa: E402


DEFAULT_LOG = PROJECT_ROOT / "data" / "demo" / "mixed-blocks.log"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", nargs="?", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    log_file = args.log_file if args.log_file.is_absolute() else PROJECT_ROOT / args.log_file
    lines = [
        line.strip()
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = predict_blocks_from_lines(lines, PROJECT_ROOT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
