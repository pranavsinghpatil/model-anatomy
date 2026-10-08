# Technical Architecture

```text
model ID / URL
  → resolver (normalized ID, pinned revision, artifact inventory)
  → extractors (config | nn.Module tree | trace | source | Safetensors metadata)
  → Raw Architecture IR
  → Semantic Architecture IR
  → graph and evidence-backed inspector
```

## Responsibilities

The resolver enumerates artifacts before downloading and records the exact revision. Extractors independently collect direct facts. Raw IR stores those facts with provenance and minimal interpretation. Semantic IR groups them into roles such as embedding, attention, MLP, normalization, repeated block, or an explicitly custom component. The frontend only renders IR; it never rediscovers model structure.

## Evidence rules

Every displayed fact is one of:

- **direct** — read from an artifact, such as a tensor shape;
- **derived** — calculated from direct facts, such as parameter count;
- **reconstructed** — supported by multiple recorded facts, such as tensor ownership;
- **unknown** — not established reliably.

Each record carries its artifact source and locator. Semantic labels and graph edges must preserve this provenance.

## Failure model

Each stage reports `success`, `partial`, `unsupported`, or `failed`, with a useful diagnostic. A partial graph is valid; a silently invented edge is not. Unsupported custom models fail honestly and do not block the known primary-model demo.

## Service boundary

The local application is framed by FastAPI. The API exposes typed contracts and
operational health without importing model libraries at startup. Reconstruction
will run behind a bounded worker and persisted job contract, with IR as the
frontend integration boundary. See [service integration](SERVICE_INTEGRATION.md)
for implemented endpoints, proposed technology choices, and production gates.
