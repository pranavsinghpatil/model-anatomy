"""Prevent network and checkpoint loading during native meta construction."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from .native_provider import NativeResolutionError


def _disabled(*args: Any, **kwargs: Any) -> Any:
    raise NativeResolutionError(
        "construction_io_disabled", "meta_construction",
        "Native construction attempted network or pretrained-file loading; it is disabled",
    )


@contextmanager
def construction_guards() -> Iterator[None]:
    """Apply only while constructing a native model in the isolated engine process."""
    import requests
    import safetensors
    import torch
    from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    with ExitStack() as stack:
        for target, name in (
            (requests.Session, "request"), (torch, "load"),
            (torch.hub, "load_state_dict_from_url"), (safetensors, "safe_open"),
            (PretrainedConfig, "from_pretrained"), (PreTrainedModel, "from_pretrained"),
            (AutoConfig, "from_pretrained"), (_BaseAutoModelClass, "from_pretrained"),
        ):
            stack.enter_context(patch.object(target, name, _disabled))
        try:
            import httpx
        except ImportError:
            pass
        else:
            stack.enter_context(patch.object(httpx.Client, "send", _disabled))
            stack.enter_context(patch.object(httpx.AsyncClient, "send", _disabled))
        yield
