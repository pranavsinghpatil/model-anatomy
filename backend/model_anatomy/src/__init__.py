"""Static inspection engine. Importing report types never loads model libraries."""

__all__ = ["reconstruct"]


def __getattr__(name: str):
    if name == "reconstruct":
        from .pipeline import reconstruct

        return reconstruct
    raise AttributeError(name)
