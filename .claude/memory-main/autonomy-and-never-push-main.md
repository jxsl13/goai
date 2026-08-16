---
name: autonomy-and-never-push-main
description: "User directive — act autonomously, ask far less; and NEVER push to main (feature branches only)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 73f72698-b352-4a1d-a7bc-9aca1549b901
  modified: 2026-07-29T14:36:26.909Z
---

Run autonomously. Decide and act instead of surfacing choices to the user. **Never push to `main`** — always a feature branch.

**Why:** given on 2026-07-29 after a session where three separate blockers each became a question: a `core.bare` git-config breakage, a push blocked by a pre-existing red test, and a non-fast-forward after rebase. Each was resolvable from evidence already gathered — the cherry check proved the force-push lost nothing, the red test was diagnosable and fixable, the config was demonstrably wrong because the repo had a checked-out working tree. Asking converted solvable work into user interrupts and stalled an autonomous loop whose whole point is not needing a human in the cycle.

**How to apply:**
- Default to acting. Reserve questions for decisions that are genuinely the user's — product direction, scope changes — not for unblocking mechanics.
- Force-pushing a *feature* branch after a rebase is routine; verify with `git cherry <local> <remote>` that every remote commit has a local equivalent (0 lines marked `+`), then push. Do not ask when that check is clean.
- A pre-existing red test blocking a push is a bug to diagnose and fix, not a permission request. Verify it is pre-existing, fix it, state the cost.
- `main` is never a push target. Branch, push the branch, open a PR.
- Still confirm genuinely destructive or outward-facing acts that evidence cannot make safe — deleting data, force-pushing a branch someone else works on, anything touching `main`.

Related: [[goai-autonomous-loop]] (the loop this governs), [[loop-keep-alive]] (never self-stop or idle), [[pull-rebase-before-new-task]] (the rebase that makes feature-branch force-pushes routine).
