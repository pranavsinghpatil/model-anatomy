# Frozen GPT-2 inspection for regression tests

This is a historical metadata-only inspection of `openai-community/gpt2` at
revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. It is used only by backend
and browser tests of immutable saved reports; the application does not serve it
as a prepared example or use it as an analysis fallback.

`gpt2-report.json` was previously available only as untracked local
`output/gpt2/model.json`. The October 25 clean-checkout audit relocated that
existing frozen report here so the test suite is reproducible. This is not a
claim that this historical artifact was produced by the current extractor.

- SHA-256: `3083419454422e591e77e66835359d8b9f3b3e177ac6eae7b6d58a3d897dd274`
- Model class: `GPT2LMHeadModel`
- Unique parameter count: 124,439,808
- Weight payload bytes read: 0
- Forward executed: false

The current extractor is checked independently by the opt-in pinned tests in
`test_engine_integration.py` and `test_live_http.py`. Review fixture changes
separately from implementation changes, with the model pin and artifact digest.
