# GitHub Main Protection

## Required policy
`main` is PR-only.

Recommended protections:
1. Require a pull request before merging.
2. Require at least 1 approval when a second collaborator can review.
3. Require the `quality` status check.
4. Require conversation resolution before merging.
5. Block force pushes.
6. Block branch deletion.
7. Do not allow bypassing the rules for normal development.

## GitHub UI
Repository → **Settings** → **Branches** → add a branch protection rule for `main`.

GitHub also supports repository rulesets for the same controls.

## Expected workflow
```text
topic branch → commit → PR → CI → review → merge
```

After the first successful PR, select the exact `quality` check under required status checks.