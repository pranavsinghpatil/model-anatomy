"""Declared model interface without synthetic inputs or guessed activation shapes."""

from __future__ import annotations

import inspect
from typing import Any

import torch


def describe_inputs(
    model: torch.nn.Module, used_arguments: list[str]
) -> dict[str, Any]:
    try:
        arguments = list(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        arguments = []
    return {
        "forward_arguments": arguments,
        "input_specs": {
            name: {"shape": None, "dtype": None} for name in sorted(set(used_arguments))
        },
        "execution": "disabled",
        "synthetic_inputs_created": False,
        "scope": "Declared forward interface and source-observed module dependencies; no activation shape inference",
    }
