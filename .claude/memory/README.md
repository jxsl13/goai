# `.claude/memory/` — worker persistent memory (version-controlled)

These markdown files are the **Linux/amd64 + RTX 3060 CUDA worker's** persistent
memory, mirrored into the repo so it is version-controlled, survives host/machine
changes, and is visible to the main agent and other hosts (like the in-repo
`.claude/skills/` and `.claude/workflows/`).

`MEMORY.md` is the index (one line per file, loaded every session). Each other file
holds one durable fact with frontmatter (`name`, `description`, `metadata.type` ∈
`user | feedback | project | reference`) and `[[wikilinks]]` between related notes.

**Sync note:** this is a snapshot of the worker's live Claude memory. The live copy
keeps updating each session; re-sync (copy the live memory dir over this one and
commit) periodically so the repo copy does not drift. Doc-only, so it pushes under
the §C16 zero-CI-runtime exception (no runner starts on a `.md`-only diff).

Key entries:
- `never-idle-build.md` — the standing directive: never idle; the throttle limits
  CI/CD runtime, never working time; build (including big cross-cutting levers),
  don't await greenlight.
- `cuda-q4-arc-state.md` — the running CUDA optimization arc log (the bulk).
- `linux-amd64-worker-role.md`, `main-machine-concurrent-campaign.md` — worker role
  and collision-avoidance with the main (M2) machine.
