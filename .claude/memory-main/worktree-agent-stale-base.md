---
name: worktree-agent-stale-base
description: Worktree subagents branch from a STALE main snapshot — sync them to current HEAD before delegating recent-commit-dependent work
metadata:
  node_type: memory
  type: feedback
  originSessionId: 89975edc-f6ec-4912-922c-d5efd862e3d7
---

Worktree-isolierte Subagenten (Agent tool, isolation: "worktree") branchen NICHT
zuverlässig von aktuellem `main` HEAD — sie starten von einem ZUNEHMEND VERALTETEN
Snapshot. Beobachtet 2026-07-14 im Perf-Sprint: drei aufeinanderfolgende Agenten
zweigten von immer älteren Basen ab (vulkan-Agent 1 Commit alt, attention-Agent 6
Commits alt, shape-keyed-cache-Agent zweigte VOR der ganzen MPSGraph-Arbeit ab und
fand `mtl_conv2d_mps_f32`/`mtl_mha_mpsgraph`/T621 gar nicht → totaler Block).

**Why:** je schneller `main` durch gemergte Agent-Wins vorrückt, desto weiter fallen
neue Worktrees zurück. Ein Agent, dessen Task auf kürzlich gemergtem Code aufbaut,
sieht diesen Code nicht → verwirrt sich (attention-Agent behauptete fälschlich, die
Conv-MPSGraph existiere nicht) oder ist blockiert.

**How to apply:**
- Bei Delegation von Arbeit, die auf KÜRZLICH gemergten Commits aufbaut: den Agenten
  im Prompt ZUERST syncen lassen — `git merge main --no-edit` (bzw. `git reset --hard
  main`, da der frische Worktree-Branch keine eigenen Commits hat) — DANN die Task.
  `main` ist im Worktree als Ref sichtbar und zeigt auf den aktuellen HEAD.
- Alternativ: recent-dependent Arbeit DIREKT auf main selbst machen (kein Worktree),
  besonders kleine/fiddly Änderungen.
- Immer den Agenten anweisen, seine Basis zu prüfen (`git log --oneline -5`) und stale
  Base zu MELDEN statt Code zu fabrizieren (der cache-Agent tat das korrekt — kein
  spuriöser Commit).
- Siehe [[goai-autonomous-loop]], [[integration-audit-method]].
