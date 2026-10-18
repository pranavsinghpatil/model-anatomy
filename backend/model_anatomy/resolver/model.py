"""Public metadata-only resolution interface."""

from typing import Any

import requests

from model_anatomy.src.artifacts import MetadataClient, parse_reference


class ModelResolver:
    """Resolve a public model to immutable config and checkpoint metadata."""

    def resolve(self, model_id: str) -> dict[str, Any]:
        reference = parse_reference(model_id)
        with requests.Session() as session:
            client = MetadataClient(session)
            revision = client.resolve(reference)
            config = client.json(reference.url("config.json", revision))
            checkpoint = client.checkpoint(reference, revision)
        return {
            "model_id": reference.model_id,
            "revision": revision,
            "config": config,
            "tensors": {name: tensor.public_dict() for name, tensor in checkpoint.tensors.items()},
            "artifacts": client.audit,
        }
