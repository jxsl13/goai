---
name: document-memory-in-repo
description: "User directive: ALWAYS mirror durable memory into the repo (.claude/memory-main/), and NEVER leak secrets when doing so."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 89975edc-f6ec-4912-922c-d5efd862e3d7
---

**User directive (2026-07-20, two paired instructions):** "always document memory into the repository" + "never leak secrets in the repository."

**Why:** the repo already has a version-controlled agent-memory convention — the CUDA worker mirrors its live Claude memory into `<repo>/.claude/memory/` (see its README + `MEMORY.md` index) so knowledge survives host changes and is visible cross-machine. The main (M2) agent must do the same. Session-local `~/.claude/.../memory/` is not shared; the repo copy is.

**How to apply:**
- The main-agent mirror lives at `<repo>/.claude/memory-main/` (sibling of the worker's `.claude/memory/`). Each session, after writing/updating a durable memory in the live `~/.claude/projects/-Users-john-Desktop-goai/memory/` dir, copy the changed file(s) into `.claude/memory-main/` and update that dir's `MEMORY.md` index. Doc-only (`.md`) → pushes under the §C16 zero-CI exception with `--no-verify` (mdlint has a pre-existing tilde residual on agent-memory files, not CI-gated).
- ENGINEERING knowledge (perf findings, anti-pattern catalogs, architecture, bug patterns) ALSO belongs in the proper repo docs: `docs/perf-notes-*.md`, `docs/decisions/ADR-*.md`, SPEC.md §B/§T. The memory mirror is the working index; the repo docs are the polished write-up. This session: `docs/perf-notes-training.md` (nn host-loop sweep) + `docs/perf-notes-lowlevel.md` (gguf decode).
- **NEVER leak secrets.** Before committing any memory/doc, scan for credentials/PII: `grep -rnEi '@googlemail|@gmail|password|api[_-]?key|token[=:]|bearer |ghp_|sk-[A-Za-z0-9]|AKIA|-----BEGIN' <files>`. Opaque `originSessionId` UUIDs are NOT secrets (the worker commits them); real credentials, tokens, private keys, and personal emails ARE. Local home paths (`/Users/…`) are not secrets but prefer repo-relative in prose. When in doubt, exclude it.

Related: [[goai-autonomous-loop]] (§C16 doc-only push exception), [[base-perf-sweep]] (the engineering knowledge that gets mirrored).
