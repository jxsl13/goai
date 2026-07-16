---
name: gh-for-pushing
description: "User directive — use the gh utility for pushing / PR operations, not bare git push"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 491e34f2-de40-467f-89a0-f918efb8a6b3
---

Use the `gh` utility for pushing and PR operations on this repo (user directive, 2026-07-14).

**Why:** the user explicitly asked "use gh utility for pushing" on the goai Linux worker.

**How to apply:** `gh auth setup-git` is configured (gh is the git credential helper), so pushes authenticate through gh. Prefer `gh pr create --head <branch>` (auto-pushes) for delivery; use `gh` for all PR/CI ops (`gh pr view`, `gh run watch`). Related: [[linux-amd64-worker-role]].
