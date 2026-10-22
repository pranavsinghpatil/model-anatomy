"""Pure dtype transfer rules for the bounded static source reader."""

from __future__ import annotations

from collections.abc import Sequence

import torch

_CAST_METHODS = {
    "float": torch.float32,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "double": torch.float64,
}


def cast_dtype(
    method: str,
    current: torch.dtype | None,
    positional: Sequence[object] = (),
    *,
    dtype_keyword: torch.dtype | None = None,
    dtype_keyword_present: bool = False,
    dtype_keyword_none: bool = False,
    type_as_dtype: torch.dtype | None = None,
    positional_is_tensor: bool = False,
) -> torch.dtype | None:
    """Return a known result dtype for standard tensor casts, otherwise None.

    The caller is responsible for establishing that the receiver is a known
    native tensor value and that arguments are statically known.
    """
    if method in _CAST_METHODS:
        return _CAST_METHODS[method]
    if method == "type_as":
        return type_as_dtype
    if method == "to":
        if dtype_keyword_present:
            return current if dtype_keyword_none else dtype_keyword
        if positional:
            if positional_is_tensor:
                return type_as_dtype
            first = positional[0]
            if isinstance(first, torch.dtype):
                return first
            if first is None or type(first) is str or isinstance(first, torch.device):
                if len(positional) > 1:
                    second = positional[1]
                    return current if second is None else second if isinstance(second, torch.dtype) else None
                return current
            return None
    return current


def factory_dtype(
    name: str,
    *,
    prototype: torch.dtype | None = None,
    explicit_dtype: torch.dtype | None = None,
    explicit_dtype_present: bool = False,
    full_value: object = None,
    full_value_known: bool = False,
) -> torch.dtype | None:
    """Return a factory result dtype from native PyTorch argument rules."""
    if explicit_dtype_present:
        return explicit_dtype
    if name.endswith("_like") or name.startswith("new_"):
        return prototype
    if name == "full":
        if not full_value_known:
            return None
        if type(full_value) is bool:
            return torch.bool
        if type(full_value) is int:
            return torch.int64
        if type(full_value) is float:
            return torch.get_default_dtype()
        return None
    if name in {"ones", "zeros", "empty"}:
        return torch.get_default_dtype()
    return None
