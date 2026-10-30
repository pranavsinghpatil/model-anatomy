"""Read callable and dispatch metadata without invoking helpers or tensor code."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from typing import Any

from transformers import PretrainedConfig
from transformers.integrations.moe import ExpertsInterface
from transformers.modeling_utils import AttentionInterface
from transformers.utils.generic import GeneralInterface

MISSING = object()
_ATTENTION_LOOKUP = AttentionInterface.get_interface
_EXPERTS_LOOKUP = ExpertsInterface.get_interface
_MAPPING_LOOKUP = GeneralInterface.__getitem__


def static_attribute(owner: Any, name: str) -> Any:
    """Read fields and trivial native property aliases, never execute a getter."""
    fields = vars(owner)
    if name in fields:
        return fields[name]
    if isinstance(owner, PretrainedConfig):
        aliases = inspect.getattr_static(type(owner), "attribute_map", {})
        if isinstance(aliases, dict) and name in aliases:
            return fields.get(aliases[name], MISSING)
        descriptor = inspect.getattr_static(type(owner), name, None)
        if isinstance(descriptor, property) and descriptor.fget is not None:
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(descriptor.fget)))
                statements = [item for item in tree.body[0].body
                              if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))]
                if len(statements) == 1 and isinstance(statements[0], ast.Return):
                    value = statements[0].value
                    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "self":
                        return fields.get(value.attr, MISSING)
            except (OSError, TypeError, SyntaxError):
                pass
    return MISSING


def attention_dispatch(owner: Any, selector: Any, default: Any) -> tuple[Any, dict[str, Any]]:
    """Snapshot the pinned provider's registry contract, including local overrides.

    This is provider metadata routing, not an attention formula or model-name
    rule. Unknown selectors and overridden lookup methods are never guessed.
    """
    lookup = {AttentionInterface: _ATTENTION_LOOKUP, ExpertsInterface: _EXPERTS_LOOKUP}.get(type(owner))
    if (lookup is None
            or inspect.getattr_static(owner, "get_interface", None) is not lookup
            or inspect.getattr_static(owner, "__getitem__", None) is not _MAPPING_LOOKUP):
        return MISSING, {}
    if selector is not None and type(selector) is not str:
        return MISSING, {}
    local = vars(owner).get("_local_mapping")
    global_mapping = inspect.getattr_static(type(owner), "_global_mapping", None)
    if type(local) is not dict or type(global_mapping) is not dict:
        return MISSING, {}
    key = selector
    if type(owner) is AttentionInterface and isinstance(key, str) and key.startswith("paged|") and key != "paged|eager":
        key = key.removeprefix("paged|")
    if key not in local and key not in global_mapping and key not in {None, "eager"}:
        return MISSING, {}
    selected = local.get(key, global_mapping.get(key, default))
    proof = {
        "rule": "native_registry_snapshot", "selector": selector, "normalized_selector": key,
        "local_override": key in local,
        "registry_type": type(owner).__name__,
        "lookup_source_sha256": hashlib.sha256(inspect.getsource(lookup).encode()).hexdigest(),
        "mapping_source_sha256": hashlib.sha256(inspect.getsource(GeneralInterface.__getitem__).encode()).hexdigest(),
        "helper_executed": False,
    }
    return selected, proof
