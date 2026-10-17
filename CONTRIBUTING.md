# Contributing

`main` is a protected integration and release branch. Do not develop or push directly to it.

```text
issue / bounded task → topic branch → implementation → deterministic checks → PR → CI + review → merge
```

Use `pranav/` branches exclusively and target `b-mvp` for application integration.
Promotion to `main` requires a separate reviewed PR. Keep each increment focused
and include its purpose, verification evidence, limitations and next task.

When work changes product or architecture direction, add a dated decision to [docs/DECISION_LOG.md](docs/DECISION_LOG.md). Keep the current milestone and release state in [docs/ROADMAP.md](docs/ROADMAP.md). The full quality and release policy is in [docs/QUALITY.md](docs/QUALITY.md).
