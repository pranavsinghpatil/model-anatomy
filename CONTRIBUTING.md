# Contributing

`main` is a protected integration and release branch. Do not develop or push directly to it.

```text
issue / bounded task → topic branch → implementation → deterministic checks → PR → CI + review → merge
```

Use the `codex/` branch prefix unless a collaborator is following another agreed convention. Keep a PR small and include what changed, why, verification evidence, limitations, and the next meaningful task.

When work changes product or architecture direction, add a dated decision to [docs/DECISION_LOG.md](docs/DECISION_LOG.md). Keep the current milestone and release state in [docs/ROADMAP.md](docs/ROADMAP.md). The full quality and release policy is in [docs/QUALITY.md](docs/QUALITY.md).
