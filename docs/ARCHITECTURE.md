# Technical Architecture

## High-level pipeline

```
Hugging Face model
        ↓
Model Resolver
        ↓
Artifact inventory
        ↓
Config + nn.Module + FX/Graph + Source + Safetensors
        ↓
Raw Architecture IR
        ↓
Semantic Architecture IR
        ↓
Interactive Model Graph
        ↓
Tensor / Parameter / Evidence inspection
        ↓
Model Autopsy
```

## Resolver
Normalize model IDs/URLs, identify revision, enumerate relevant files, and avoid unnecessary downloads.

## Extraction
Separate extractors for configuration, module tree, graph/tracing, source, and tensor metadata.

## Raw IR
Artifact-derived structure with minimal semantic assumptions.

## Semantic IR
Human-facing concepts such as embedding, attention, MLP, norm, residual, head, repeated block, or custom component.

## Frontend boundary
Frontend consumes IR. It should not rediscover model structure.

## Failure states
Every stage should support:
- success
- partial/degraded
- unsupported
- failed

Never silently invent structure.
