"""Source-fingerprinted access to selected native metadata backing fields.

The helpers in this module inspect instance dictionaries only. They never invoke
descriptors, model forwards, or publisher code.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig

CONTRACT_SOURCE_SHA256 = {
    "transformers.configuration_utils": "08a3889c1b8a73d340a37c93f72346ee0e385e339b7a72a03b5c026d4943baf7",
}


def native_config_attribute(config: object, name: str) -> tuple[bool, Any]:
    """Read the reviewed attention implementation property without invoking it."""
    if type(name) is not str or name != "_attn_implementation":
        return False, None
    descriptor = inspect.getattr_static(type(config), name, None)
    trusted = inspect.getattr_static(PretrainedConfig, name, None)
    if (
        not isinstance(config, PretrainedConfig)
        or not isinstance(descriptor, property)
        or not isinstance(trusted, property)
        or descriptor.fget is not trusted.fget
    ):
        return False, None
    try:
        source = inspect.getsourcefile(trusted.fget)
        if (
            not source
            or hashlib.sha256(Path(source).read_bytes()).hexdigest()
            != CONTRACT_SOURCE_SHA256["transformers.configuration_utils"]
        ):
            return False, None
    except (OSError, TypeError):
        return False, None
    backing = vars(config).get("_attn_implementation_internal")
    if type(backing) is str and backing:
        return True, backing
    return False, None
