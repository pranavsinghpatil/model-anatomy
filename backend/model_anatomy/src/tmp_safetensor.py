from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from math import prod

import requests


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: tuple[int, ...]
    num_parameters: int


@dataclass(frozen=True)
class SafetensorsMetadata:
    metadata: dict[str, str]
    tensors: dict[str, TensorMetadata]


def normalize_hf_safetensors_url(url: str) -> str:
    """
    Convert a Hugging Face browser URL:

        https://huggingface.co/openai-community/gpt2/blob/main/model.safetensors

    into the raw/resolve URL:

        https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
    """
    return url.replace("/blob/", "/resolve/")


def read_safetensors_metadata(
    url: str,
    *,
    timeout: float = 30.0,
) -> SafetensorsMetadata:
    """
    Read only the Safetensors header using HTTP Range requests.

    Weight bytes are NOT downloaded.
    """

    url = normalize_hf_safetensors_url(url)

    # First 8 bytes = uint64 little-endian header length.
    response = requests.get(
        url,
        headers={"Range": "bytes=0-7"},
        timeout=timeout,
    )
    response.raise_for_status()

    if len(response.content) != 8:
        raise ValueError(
            f"Expected 8 bytes for Safetensors header length, "
            f"got {len(response.content)}"
        )

    header_length = struct.unpack("<Q", response.content)[0]

    # Next `header_length` bytes = JSON header.
    header_start = 8
    header_end = 8 + header_length - 1

    response = requests.get(
        url,
        headers={"Range": f"bytes={header_start}-{header_end}"},
        timeout=timeout,
    )
    response.raise_for_status()

    header = json.loads(response.content.decode("utf-8"))

    file_metadata = header.pop("__metadata__", {})

    tensors: dict[str, TensorMetadata] = {}

    for name, info in header.items():
        shape = tuple(info["shape"])

        tensors[name] = TensorMetadata(
            name=name,
            dtype=info["dtype"],
            shape=shape,
            num_parameters=prod(shape),
        )

    return SafetensorsMetadata(
        metadata=file_metadata,
        tensors=tensors,
    )
