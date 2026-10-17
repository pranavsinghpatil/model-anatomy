"""Local JSON and Mermaid CLI for static native-Transformers reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from .errors import AnatomyError
from .pipeline import reconstruct
from .render import write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Statically reconstruct native Transformers architecture without weights or forward execution"
    )
    parser.add_argument("model", help="Hugging Face repository ID or model.safetensors browser URL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--preview-dir", type=Path, default=Path("preview"))
    parser.add_argument("--revision", help="Optional immutable 40-character commit SHA")
    parser.add_argument(
        "--inspect", action="store_true", help="Print semantic coverage and unknown component diagnostics"
    )
    args = parser.parse_args(argv)
    try:
        ir = reconstruct(args.model, revision=args.revision)
        slug = ir.model_id.replace("/", "--").lower()
        if ir.model_id == "openai-community/gpt2":
            slug = "gpt2"
        paths = write_outputs(
            ir,
            args.output or Path("output") / slug,
            args.config_dir or Path("config") / slug,
            args.preview_dir,
        )
    except (AnatomyError, requests.RequestException, OSError) as exc:
        parser.exit(1, f"Reconstruction failed: {exc}\n")
    print(f"Model: {ir.model_id} ({ir.architecture})")
    print(f"Revision: {ir.metadata['revision']}")
    print(f"Unique parameters: {ir.metadata['unique_parameter_count']:,}")
    print(
        f"Safetensors header bytes read: {ir.metadata['artifacts']['safetensors_bytes_read']:,}; weight payload bytes: 0"
    )
    for path in paths:
        print(path)
    if args.inspect:
        import json

        print(json.dumps(ir.metadata["analysis_report"], indent=2))
    return 0
