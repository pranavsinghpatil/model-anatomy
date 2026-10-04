"""Local JSON and Mermaid CLI for the metadata-only GPT-2 MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from .errors import AnatomyError
from .pipeline import reconstruct
from .render import write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct hierarchical GPT-2 architecture without downloading weights"
    )
    parser.add_argument("model", help="Hugging Face repository ID or model.safetensors browser URL")
    parser.add_argument("--output", type=Path, default=Path("output/gpt2"))
    parser.add_argument("--config-dir", type=Path, default=Path("config/gpt2"))
    args = parser.parse_args(argv)
    try:
        ir = reconstruct(args.model)
        paths = write_outputs(ir, args.output, args.config_dir, Path(".playground"))
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
    return 0
