"""Architecture reconstruction pipeline."""

class ArchitectureReconstructor:
    """Combine module, graph, source and tensor evidence into IR."""

    def reconstruct(self, model) -> dict:
        raise NotImplementedError
