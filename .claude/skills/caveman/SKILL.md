---
name: caveman
description: |
  Caveman encoding for SPEC.md and spec-adjacent writes — telegraphic, precise,
  code/errors kept verbatim. Full grammar/symbols/shapes live in FORMAT.md at repo
  root (read that once per session). Triggers on SPEC.md writes or when the user
  says "caveman", "compress this", "be brief".
---

# caveman — spec encoding

The full encoding (grammar, symbol table, row/invariant shapes, preserve-verbatim
rules) is defined ONCE in `FORMAT.md` at repo root. Read it if not already loaded;
do not restate it here. This skill only adds the guardrails around it.

Applies to: SPEC.md writes, spec-referencing prose, §B/backprop entries.
Does NOT apply to: code, error strings, commit messages, PR descriptions, diffs.

## GUARDRAILS

- Compression, not amputation: if cutting a word loses a fact, keep the word.
- Normal English (skill off) for: prose explanations the user asked for, RFC/pitch
  docs for external review, commit messages, code comments.
- Auto-clarity: never compress a safety warning or an irreversible-action notice —
  spell those out in plain English.

## SHAPE (quick ref; canonical in FORMAT.md)

- Invariant: `V<n> <TAG>: <subject> <relation> <condition>`
- Task row (§T): `id|status|task|cites`  — status `x` done / `~` wip / `.` todo.
- Bug row (§B): `id|date|cause|fix`

Numbering monotonic — never reuse V.n / B.n / T.n.
