"""Bounded checkpoint-compatible native architecture selection, without tensor execution."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PretrainedConfig

from . import limits
from .artifacts import CheckpointMetadata, InspectionBudgetError, SafetensorsMetadata
from .errors import ArtifactError, NativeShapeMismatchError, UnsupportedModelError
from .facts import inspect_native_model
from .registry import _auto_model_candidates, instantiate_candidate
from .source_eval import DTYPES


def compatible_dtype(native: str, stored: str | None) -> bool:
    """Allow floating storage precision changes, never assume quantized/integer equivalence."""
    names = {str(dtype): name for dtype, name in DTYPES.items()}
    names.update({"torch.int8": "I8", "torch.int16": "I16", "torch.uint8": "U8"})
    expected = names.get(native)
    floating = {"F16", "BF16", "F32", "F64"}
    return stored in floating if expected in floating else expected is not None and expected == stored


def compatibility(model: torch.nn.Module, raw: dict[str, Any]) -> dict[str, Any]:
    """Require all checkpoint tensors and required native state to have verified owners."""
    missing = [name for name, item in raw["parameters"].items()
               if name == item["canonical_parameter"] and not item["checkpoint_matched"]]
    unowned = [name for name, item in raw["checkpoint_extras"].items() if item["kind"] != "buffer"]
    supplied_buffers = {path for item in raw["checkpoint_extras"].values()
                        if item["kind"] == "buffer" for path in item["buffer_paths"]}
    missing_buffers = []
    buffer_aliases: dict[int, set[str]] = {}
    for name, buffer in model.named_buffers(remove_duplicate=False):
        buffer_aliases.setdefault(id(buffer), set()).add(name)
    supplied_buffers.update(path for group in buffer_aliases.values()
                            if group & supplied_buffers for path in group)
    for name in raw["buffers"]:
        owner, _, local = name.rpartition(".")
        module = model.get_submodule(owner) if owner else model
        if local not in module._non_persistent_buffers_set and name not in supplied_buffers:
            missing_buffers.append(name)
    alias_conflicts = [item["aliases"] for name, item in raw["parameters"].items()
                       if name == item["canonical_parameter"] and len(item["tensor_names"]) > 1]
    dtype_conflicts = [{"parameter": name, "native_dtype": item["module_dtype"], "checkpoint_dtype": item["dtype"]}
                       for name, item in raw["parameters"].items()
                       if name == item["canonical_parameter"] and item["checkpoint_matched"]
                       and not compatible_dtype(item["module_dtype"], item["dtype"])]
    dtype_conflicts.extend({"buffer": path, "checkpoint_tensor": name,
                            "native_dtype": raw["buffers"][path]["dtype"], "checkpoint_dtype": item["checkpoint_dtype"]}
                          for name, item in raw["checkpoint_extras"].items() if item["kind"] == "buffer"
                          for path in item["buffer_paths"]
                          if not compatible_dtype(raw["buffers"][path]["dtype"], item["checkpoint_dtype"]))
    return {
        "status": "exact_metadata_match" if not (missing or unowned or missing_buffers or alias_conflicts or dtype_conflicts or raw["warnings"]) else "incompatible",
        "missing_parameters": missing, "unowned_checkpoint_tensors": unowned,
        "native_ignored_checkpoint_tensors": [name for name, item in raw["checkpoint_extras"].items()
                                               if item["kind"] == "legacy_checkpoint_tensor"],
        "missing_persistent_buffers": missing_buffers, "unverified_tied_values": alias_conflicts,
        "parameter_alias_groups": raw["parameter_alias_groups"],
        "key_conversions": raw["checkpoint_name_conversions"],
        "parameter_shapes_checked": True, "tensor_values_verified": False,
        "dtype_conflicts": dtype_conflicts,
        "dtype_policy": "F16/BF16/F32/F64 storage may feed native floating parameters/buffers; integer and boolean types require an exact metadata dtype match; quantized semantics are not inferred",
        "scope": "Complete native-state ownership and shapes; storage dtypes remain header evidence, not numerical equivalence",
    }


def select_checkpoint_architecture(
    config: PretrainedConfig, checkpoint: CheckpointMetadata | SafetensorsMetadata,
    model: torch.nn.Module, selection: dict[str, Any], raw: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Prefer a declared complete graph; switch only after a unique complete metadata match."""
    declared_class = type(model).__name__
    initial = compatibility(model, raw)
    proof: dict[str, Any] = {
        "rule": "unique_complete_native_checkpoint_match",
        "declared_architectures": list(config.architectures or []),
        "initial_model_class": declared_class, "selected_model_class": declared_class,
        "status": "declared_metadata_match" if config.architectures else "default_metadata_match",
        "search_performed": False, "search_complete": None,
        "candidates": [dict(model_class=declared_class, auto_class=selection["auto_class"], compatibility=initial)],
        "matching_candidates": [], "forward_executed": False, "weight_payload_bytes_read": 0,
        "scope": "Checkpoint-compatible native interpretation, not proof of publisher intent or runtime behavior",
    }
    selection = dict(selection, architecture_selection=proof)
    if initial["status"] == "exact_metadata_match":
        proof["matching_candidates"] = [declared_class]
        return model, selection, raw

    ignored = initial["native_ignored_checkpoint_tensors"]
    active_failures = any(initial[key] for key in (
        "missing_parameters", "missing_persistent_buffers", "unverified_tied_values", "dtype_conflicts"))
    if (config.architectures and ignored and not active_failures and not raw["warnings"]
            and set(initial["unowned_checkpoint_tensors"]) == set(ignored)):
        proof.update(status="declared_native_with_ignored_tensors", native_state_covered=True,
                     checkpoint_ownership_complete=False, native_ignored_checkpoint_tensors=ignored,
                     scope="Declared native implementation with complete active state; ignored checkpoint keys remain ownerless inventory")
        # Ignored keys never make a candidate an exact match or establish a unique alternative.
        return model, selection, raw

    # Multiple Auto task aliases may point at the same concrete implementation.
    candidates: dict[str, tuple[str, type, str]] = {}
    aliases: dict[str, list[str]] = {}
    for candidate in sorted(_auto_model_candidates(config), key=lambda item: (item[2], item[0])):
        name, _, class_name = candidate
        if class_name == declared_class:
            continue
        candidates.setdefault(class_name, candidate)
        aliases.setdefault(class_name, []).append(name)
    proof["candidate_count"] = len(candidates)
    proof["candidate_limit"] = limits.MAX_NATIVE_ARCHITECTURE_CANDIDATES
    proof["search_performed"] = True
    matches = []
    spent = 0
    proof["search_complete"] = len(candidates) <= limits.MAX_NATIVE_ARCHITECTURE_CANDIDATES
    for candidate in list(candidates.values())[:limits.MAX_NATIVE_ARCHITECTURE_CANDIDATES]:
        auto_name, _, class_name = candidate
        record: dict[str, Any] = {"model_class": class_name, "auto_class": auto_name,
                                  "auto_aliases": aliases[class_name]}
        proof["candidates"].append(record)
        try:
            probe, _ = instantiate_candidate(config, candidate)
            facts = inspect_native_model(probe, checkpoint)
            record["native_class"] = f"{type(probe).__module__}.{type(probe).__qualname__}"
            record["forward_source"] = facts["source_fingerprints"].get("")
            record["compatibility"] = compatibility(probe, facts)
            spent += len(facts["modules"]) + len(facts["parameters"]) + len(facts["buffers"])
            if record["compatibility"]["status"] == "exact_metadata_match":
                matches.append(candidate)
            del probe, facts
        except InspectionBudgetError as exc:
            record["compatibility"] = {"status": "budget_exceeded", "reason": str(exc)}
            proof["search_complete"] = False
        except NativeShapeMismatchError as exc:
            record["compatibility"] = {"status": "rejected", "reason": str(exc)}
        except ArtifactError as exc:
            record["compatibility"] = {"status": "unverified", "reason": str(exc)}
            proof["search_complete"] = False
        except (UnsupportedModelError, ImportError, AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            record["compatibility"] = {"status": "unavailable", "reason": str(exc)}
            proof["search_complete"] = False
        if spent >= limits.MAX_ARCHITECTURE_SEARCH_RECORDS:
            proof["search_complete"] = False
            break
    proof["inspected_native_records"] = spent
    proof["matching_candidates"] = [candidate[2] for candidate in matches]
    if len(matches) == 1 and proof["search_complete"]:
        selected, native_selection = instantiate_candidate(config, matches[0])
        selected_raw = inspect_native_model(selected, checkpoint)
        if compatibility(selected, selected_raw)["status"] != "exact_metadata_match":
            raise ArtifactError("Architecture compatibility changed during final native construction")
        proof.update(status="checkpoint_selected", selected_model_class=type(selected).__name__,
                     effective_config_changes={"architectures": [matches[0][2]], "_attn_implementation": "eager"})
        native_selection["architecture_selection"] = proof
        return selected, native_selection, selected_raw

    proof["status"] = "incomplete_search" if not proof["search_complete"] else "ambiguous" if len(matches) > 1 else "no_exact_match"
    selection["warnings"] = [*selection["warnings"],
        f"Native architecture selection {proof['status']}; declared graph retained with checkpoint discrepancies"]
    return model, selection, raw
