"""Verify a pinned Safetensors index against every shard header; read no weights."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from model_anatomy.src.artifacts import MetadataClient, ModelReference
from model_anatomy.src.errors import ArtifactError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="Public Hugging Face namespace/model")
    parser.add_argument("revision", help="Immutable 40-character commit SHA")
    parser.add_argument("--output", type=Path, help="Write verified aggregate JSON after all shards pass")
    args = parser.parse_args()
    if (
        not re.fullmatch(r"[\w.-]+/[\w.-]+", args.model_id)
        or any(part in {".", ".."} for part in args.model_id.split("/"))
        or not re.fullmatch(r"[0-9a-f]{40}", args.revision)
    ):
        parser.error("A Hugging Face model ID and immutable 40-character revision are required")
    reference = ModelReference(args.model_id, args.revision)
    client = MetadataClient(timeout=30)
    try:
        resolved = client.resolve(reference)
        if resolved != args.revision:
            raise ArtifactError("Resolved repository revision does not match the requested commit")
        result = client.verify_indexed_checkpoint_headers(reference, resolved)
    except (ArtifactError, requests.RequestException) as exc:
        failure = {
            "model_id": args.model_id,
            "revision": args.revision,
            "status": "failed_unverified",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_requests": client.audit,
        }
        if args.output:
            failure_path = args.output.with_suffix(".failed.json")
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(failure_path, file=sys.stderr)
        print(f"Header verification failed: {exc}", file=sys.stderr)
        return 1
    document = {
        "model_id": args.model_id,
        "revision": resolved,
        "verification": asdict(result),
    }
    output = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(args.output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
