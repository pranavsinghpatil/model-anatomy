"""Audit exact pinned DeepSeek checkpoint names against reviewed publisher source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from model_anatomy.src.publisher_name_coverage import audit_name_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        index = json.loads(args.index.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = audit_name_coverage(index, config, args.source_dir, args.converter)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failure = {
            "status": "failed_unverified",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "index_path": str(args.index),
            "config_path": str(args.config),
        }
        path = args.output.with_suffix(".failed.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(path, file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
