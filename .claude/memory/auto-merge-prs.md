---
name: auto-merge-prs
description: "User directive — automatically merge PRs once CI passes (don't wait for the user)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 491e34f2-de40-467f-89a0-f918efb8a6b3
---

Auto-merge my PRs once CI succeeds (user directive, 2026-07-14). Don't leave them open for the user to merge.

**Why:** the user explicitly asked; keeps the merged base moving so stacked/dependent work isn't blocked.

**How to apply:** the repo has **no required-status-check branch protection**, so `gh pr merge --auto` merges *immediately* (doesn't wait for checks). To honor "once CI SUCCEEDS", instead **watch checks to completion, then merge only if green**:

```
gh pr checks <n> --watch --fail-fast   # blocks until all checks finish; nonzero if any fail
gh pr merge <n> --merge && echo merged  # merge commit (matches repo history)
```

Use `--merge` (merge commit) to match the repo's existing "Merge pull request #N" history. If a check fails, do NOT merge — fix on the branch first. CUDA tests are NOT in CI (no GPU runner), so a green CI does not cover the cuda backend — I validate those locally on the RTX 3060 before pushing. See [[gh-for-pushing]], [[linux-amd64-worker-role]].

**§C16 push-throttle EXCEPTIONS (bypass the ≤1/hr wait — codified in root SPEC.md §C16, merged #131 2026-07-16):**
- EXC1: internal/cichange + .github/workflows/* push immediately (need live CI).
- EXC2: ZERO-CI-minute pushes anytime — diff matches ci.yml paths-ignore `['*.md','docs/**']` (root markdown incl. **SPEC.md**, docs tree) → no runner starts.
- EXC3 [user directive 2026-07-16 "throttle bypass spec only changes that need to be passed to the main agent"]: a SPEC-ONLY change passing a task/finding to the MAIN AGENT (a §T row surfacing a cross-machine lever, e.g. T762 MoE sparse-decode) push+merge IMMEDIATELY — main-agent coordination + doc-only. ALL OTHER pushes (perf/code PRs, e.g. cuda kernels) stay ≤1/hr. So: /spec task-adds for the main agent → ship now; kernel/perf PRs → wait the window. cichange auto-skips the heavy matrix on a SPEC.md-only diff (verified #129/#131: cgo/pure-go/vulkan all "skipping").
