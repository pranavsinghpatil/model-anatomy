"""Architecture-scoped completion, independent of expression/activation inference."""

from __future__ import annotations

from typing import Any


def completion_report(raw: dict[str, Any], represented: set[str], covered_by_leaf: dict[str, str],
                      selection: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    missing_modules = sorted(set(raw["modules"]) - represented - covered_by_leaf.keys())
    missing_parameters = [name for name, item in raw["parameters"].items()
                          if name == item["canonical_parameter"] and not item["checkpoint_matched"]]
    unmapped = [name for name, item in raw["checkpoint_extras"].items() if item["kind"] == "unmapped"]
    native_status = selection["architecture_selection"]["status"]
    boundaries = {path: gaps for path, report in topology.items()
                  if (gaps := [item for item in report["boundaries"] if item.get("blocking", True)])}
    notes = {path: items for path, report in topology.items()
             if (items := [item for item in report["boundaries"] if item.get("blocking") is False])}
    checks = {
        "module_inventory": not missing_modules,
        "parameter_mapping": not missing_parameters and not unmapped,
        "native_architecture": native_status in {
            "declared_metadata_match", "default_metadata_match", "checkpoint_selected", "declared_native_with_ignored_tensors",
        },
        "module_topology": not boundaries,
    }
    return {
        "version": 3, "status": "complete" if all(checks.values()) else "partial", "checks": checks,
        "scope": "Native registered module architecture, checkpoint binding and bounded module-call dependencies",
        "excluded_from_scope": ["Tensor expression interpretation", "Intermediate activation shapes",
                                "Runtime branch selection", "Numerical equivalence to publisher custom code"],
        "module_count": len(raw["modules"]), "represented_module_count": len(represented),
        "covered_by_basic_module": covered_by_leaf, "missing_module_paths": missing_modules,
        "missing_parameter_paths": missing_parameters, "unmapped_checkpoint_tensors": unmapped,
        "topology_boundaries": boundaries, "non_blocking_diagnostics": notes,
        "registered_without_observed_invocation": "Registration establishes presence, not execution; optional modules are retained",
    }
