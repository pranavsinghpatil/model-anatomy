"""Public reconstruction interface backed by the deterministic inspection engine."""

from typing import Any


class ArchitectureReconstructor:
    """Combine module, graph, source and tensor evidence into IR."""

    def reconstruct(self, model: str | dict[str, Any]) -> dict[str, Any]:
        from model_anatomy.src.pipeline import reconstruct

        reference = model if isinstance(model, str) else model["model_id"]
        revision = model.get("revision") if isinstance(model, dict) else None
        return reconstruct(reference, revision=revision).to_dict()
