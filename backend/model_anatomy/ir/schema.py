"""Canonical IR types will be defined here.

Keep this representation model-family-neutral. Do not bake Transformer-only
assumptions into the root schema.
"""

from dataclasses import dataclass, field
from typing import Any

@dataclass
class Evidence:
    source: str
    locator: str | None = None
    kind: str = "observed"
    details: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelIR:
    model_id: str
    architecture: dict[str, Any] = field(default_factory=dict)
    components: list[dict[str, Any]] = field(default_factory=list)
    tensors: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    operations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
