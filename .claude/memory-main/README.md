# `.claude/memory-main/` — main (M2) agent persistent memory (version-controlled)

These markdown files are the **main macOS/arm64 (Apple M2 Pro) agent's** persistent
memory, mirrored into the repo so it is version-controlled, survives host/machine
changes, and is visible to the CUDA worker and other hosts — the sibling of the
Linux/amd64 CUDA worker's [`.claude/memory/`](../memory/README.md).

`MEMORY.md` is the index (one line per file, loaded every session). Each other file
holds one durable fact with frontmatter (`name`, `description`, `metadata.type` ∈
`user | feedback | project | reference`) and `[[wikilinks]]` between related notes.

**Sync note:** this is a snapshot of the main agent's live Claude memory. The live
copy keeps updating each session; re-sync (copy the live memory dir over this one and
commit) periodically so the repo copy does not drift. Doc-only, so it pushes under the
§C16 zero-CI-runtime exception (no runner starts on a `.md`-only diff).

Key entries:

- `base-perf-sweep.md` — the base-perf anti-pattern catalog (per-element
  `Unravel`/`AtF64`/`SetF64` dispatch, and the worse per-element *allocation* class),
  which layers to sweep, and the measured wins. See also `docs/perf-notes-training.md`
  and `docs/perf-notes-lowlevel.md` for the repo-doc write-ups.
- `self-policing-guard-pattern.md` — how to write a guard that actually guards
  (§B77), and the `WithDefaults()` footgun-vs-correct false-positive trap.
- `goai-autonomous-loop.md` — the autonomous build-loop operating model, push
  protocol, and cross-machine collision-avoidance with the CUDA worker.
- `integration-audit-method.md`, `subagent-output-is-untrusted.md` — auditing and
  delegate-then-independently-reverify discipline.

No secrets are stored here: entries are engineering notes only (no credentials,
tokens, or personal data). The `originSessionId` frontmatter is an opaque session
identifier, not a secret.
