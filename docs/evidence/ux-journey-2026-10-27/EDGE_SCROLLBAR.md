# Edge scrollbar follow-up — 28 October 2026

The catalog and entry pages render dark surfaces even when the workspace theme
preference is light. Edge previously rendered a white viewport scrollbar track
beside those surfaces. The embedded browser did not expose this native scrollbar
appearance during the original visual review.

Viewport scrollbars now use the workspace surface tokens, with a dark track on
the entry and catalog pages. This preserves the workspace theme preference.

Verified in the user's regular Edge window with the populated Supabase catalog:
382 records before and after the CSS change. The white track is replaced by a
dark track. The floating brain-icon controls are injected outside the app's
`#root`; they remain separate from this scrollbar fix.

- [Before](edge-scrollbar-before.png)
- [After](edge-scrollbar-after.png)

This follow-up does not claim to repair the external widget's panel. Its wider
blank panel in the supplied crop was not reproduced in the captured Edge view.

The user subsequently identified `chat-gpt-query-model-wrapper` on the blank
panel. That class is absent from the frontend source. Edge's extension list
includes Sider; the exact wrapper owner has not been confirmed through its DOM.
Changing browser extension settings requires the user's approval.

Validation: frontend production build (including TypeScript), formatting check,
and `git diff --check` passed. Vite retained its existing large-chunk advisory.

## Targeted latch fix

The subsequent site-level fix suppresses both user-identified containers,
`.chat-gpt-query-model-wrapper` and `.chat-gpt-query-model`, on pages containing
the app root. Hiding the outer wrapper removes its panel and latch together.
The stylesheet also applies to containers injected after the app has loaded.
No browser extension settings are changed.

The catalog regression test injects the named wrapper, its fixed white panel
and latch, and a standalone panel. It verifies that the injected UI is hidden
and the catalog remains usable. This is isolated DOM-fixture coverage; it does
not claim another live verification of the user's installed extension.

Validation of the latch fix: production build passed; the focused Playwright
regression passed on desktop and mobile (2 tests) against an isolated production
preview. The navigation assertion uses `Your inspections`, available in both
responsive headers. The preview used the test runner's loopback permissions;
initial isolated-server connection failures were resolved before acceptance.
