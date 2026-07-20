---
name: doc-agent-lane
description: "Docs/benchmark agent role — own worktree, md-only zero-CI direct pushes, book code tasks for main agent, never implement code"
metadata: 
  node_type: memory
  type: project
  originSessionId: c16eaf56-317b-4041-b4e6-8da553242572
  modified: 2026-07-19T23:45:47.139Z
---

Established 2026-07-20 (user directive): a SECOND agent role beside the main loop — the
**docs+benchmark agent**, running in its own worktree (`.claude/worktrees/docs-…`) while
the main agent works the repo.

Lane rules (user-set):

- Writes ONLY documentation: root `*.md` + `docs/**`. These diffs are zero-CI
  (cichange EXC2, verified: `internal/cichange -validate` → "impact=none") → **push
  direct to origin main anytime**, no C16 hourly throttle. Spec-only task bookings =
  EXC3, also push immediately.
- NEVER implements code — godoc/Example gaps are .go changes (CI-consuming) → book as
  §T tasks for the main agent instead (e.g. [[goai-autonomous-loop]] picks them up).
- Loop shape per iteration: write one doc deliverable OR book one found task → ONE
  clean commit each → fetch-rebase → push → §V27 selector-validate → sweep for the
  next doc/bug item (LOOP.md-analog).
- Bugs/inconsistencies found while writing docs → backprop (§B row + booked §T fix
  task), don't fix code myself.
- Commit gates: `go run ./internal/mdlint <my files>` must be clean; commit with
  `-c core.hooksPath=/dev/null` (pre-commit runs whole-tree lint-md, red on
  worker-owned files I must not touch: SPEC-worker-*.md, .claude/memory of the
  worker).
- SPEC edits: §T is the LAST section (V36, B97) — append rows at EOF; before booking
  ids, re-check max §T/§B id on the REBASED head (B96 duplicate-id lesson); §B rows
  prepend newest-first under the header.

**Why:** doc pushes cost no CI minutes and must not collide with the parallel main
agent; SPEC task rows are the cross-agent coordination channel.

**How to apply:** when invoked as the docs/benchmark agent (/build with a docs
mandate), follow this lane; flagship deliverable pattern = BENCHMARKS.md (T880) +
booked measurement tasks (T881-T888) + drift fixes (T890/T894).
