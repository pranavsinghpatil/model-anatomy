# Live Backend Acceptance

Verified locally on October 9, 2026. This is not a cloud-deployment claim.

## Real Inspection, Not a Prepared Result

`RUN_HTTP_INTEGRATION=1 pytest -q -s backend/tests/test_live_http.py` launches a
real API and worker against a fresh SQLite database and an absent prepared-report
directory. Neither HTTP traffic nor engine functions are mocked. It submits a
model not included among the prepared examples:

| Fact | Observed Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-360M` |
| Resolved immutable revision | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` |
| Unique parameters | 361,821,120 |
| Checkpoint tensor records | 290 |
| Dimensions | 32 layers, hidden size 960, 15 heads |
| Prepared | `false` |
| First inspection cached | `false` |
| Second inspection cached | `true`, same revision and parameter count |
| Weight payload downloaded | 0 bytes |
| Report status | `partial`, with explicit extraction limitations |

The test compares reported dimensions with the upstream config at the resolved
commit. It retrieves nested graphs, component tensor evidence, JSON and ZIP
exports; rejects another visitor's reads and a third same-visitor submission;
and verifies a nonexistent repository fails without returning a prepared graph.

The running development server on port 8001 was also exercised through Swagger:
POST returned a real persisted job (202); GET returned the inspected 360M report
(200), `prepared=false`, `cached=false`, and the same pinned artifact facts.
Swagger's "Example Value" is schema documentation, not a server response. Use
"Try it out" and "Execute" to issue a request, then inspect "Server response".

## Operational Contract

`GET /health` is the only status endpoint. It checks database, storage and the
worker lease and reports optional AI configuration. It returns 200 when new
analyses can run, or a typed degraded body with 503 when they cannot. The old
`/healthz`, `/readyz` and `/api/v1/capabilities` return 404.

The application opens an own recent job or an empty workspace. Prepared examples
remain explicit opt-in reports, never a fallback for new analyses or failures.

## Remaining Gates

- Supabase PostgreSQL/private storage and live Nebius Nemotron need credentials
  and provider verification. Optional AI is currently reported unavailable.
- Production containers, HTTPS, host restart recovery and Linux memory enforcement
  still require deployment verification; local Windows tests are not sufficient.
- Static extraction is bounded, not forward execution. Unknowns and unsupported
  native/custom models remain honest failures or explicit partial information.
