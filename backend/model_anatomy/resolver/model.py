"""Model resolution will live here.

Responsibilities:
- validate/normalize a Hugging Face model reference
- discover repository artifacts
- load configuration safely
- prepare model loading metadata
"""

class ModelResolver:
    """Initial resolver interface; implementation comes next."""

    def resolve(self, model_id: str) -> dict:
        raise NotImplementedError
