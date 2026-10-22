# Model Anatomy documentation

The documentation is intentionally small. Each file has one job:

1. [00-GROUND.md](00-GROUND.md) — what we are building and what it is not.
2. [ARCHITECTURE.md](ARCHITECTURE.md) — how evidence becomes an inspectable graph.
3. [ROADMAP.md](ROADMAP.md) — current status, dated milestones, and release plan.
4. [QUALITY.md](QUALITY.md) — working, benchmark, demo, submission, and branch-protection gates.
5. [DECISION_LOG.md](DECISION_LOG.md) — permanent record of binding decisions.

Read [AGENTS.md](../AGENTS.md) first when performing repository work.

For application users, [Reading Reports](READING_REPORTS.md) explains status,
revision, coverage checks, actual warnings, analysis scope and manual verification.

[Backend reliability](BACKEND_RELIABILITY.md) records the current measured gap,
the stepwise acceptance approach and the next reconstruction gate.
[Teammate handoff](TEAMMATE_HANDOFF.md) contains review dependencies, reproduction
commands and the next bounded backend task.
[Dtype evidence](DTYPE_EVIDENCE.md) describes the supported value-specific source
facts and the limits of native input profiles.

[UI redesign](UI_REDESIGN.md) documents the application flow, design system,
responsive workspace and validation requirements. Its annotated
[wireframes](design/workspace-wireframes.svg) describe desktop and mobile layouts.
