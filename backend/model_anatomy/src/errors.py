"""Explicit failures for unsupported models and contradictory evidence."""


class AnatomyError(ValueError):
    """Base error for a failed reconstruction."""


class ArtifactError(AnatomyError):
    """An artifact is unavailable, malformed, or inconsistent."""


class UnsupportedModelError(AnatomyError):
    """The input requires unsupported architecture, code, or dependencies."""
