# FORMAT — caveman encoding for SPEC.md

Applies to SPEC.md + spec-adjacent writes. ⊥ apply to code, error strings, commits.

## GRAMMAR
Drop articles, filler, aux verbs, hedging. Fragments fine. Short synonyms (fix>implement, big>extensive, run>execute).

## SYMBOLS
```
→ leads to/becomes   ∴ therefore/fix   ∀ every   ∃ some   ! must   ? optional/unknown
⊥ never/forbidden/nil   ≠ not equal   ∈ in   ∉ not in   ≤ at most   ≥ at least   § section   & and   ∨ or
```

## PRESERVE VERBATIM
Code, paths, URLs, identifiers, numbers/versions, error strings, quoted strings.

## SHAPES
Invariant (under §V): `V<n> <TAG>: <subject> <relation> <condition>`

§T, §B, §R are **CLEAN GFM MARKDOWN TABLES** — they MUST render as tables and pass mdlint's table checks (`table-ragged` + `table-separator-mismatch`). Each = a header row, a delimiter row, then data rows; every row has leading & trailing `|` and the SAME column count. Columns:

- §T (tasks): `| id | status | task | cites | state | priority |` — status `x` done, `~` wip, `.` todo; state ∈ done/wip/. (secondary lifecycle); priority ∈ high/med/low. Old rows may leave state+priority empty.
- §B (bugs): `| id | date | cause | fix |`
- §R (research): `| id | claim | source | conf |` — conf ∈ high/med/low/ref.

Each table opens with its header + delimiter row, e.g. §T:

```
| id | status | task | cites |
|----|--------|------|-------|
| T1 | . | scaffold repo | I.L0 |
```

Inside a table CELL, `|` is ALWAYS the column separator — a cell NEVER contains a bare `|`: write logical-"or" as `∨`, and escape any other literal `|` as `\|`. Keep the delimiter row (`|---|---|---|---|`) directly under the header, never omit it.

## RULE
Numbering monotonic — ⊥ reuse V.n/B.n/T.n. If cutting word loses fact, keep it. Compression, not amputation.
Tables clean: ∀ markdown table (here ∨ in docs/README) = valid GFM (header + delimiter row + consistent columns), mdlint-green. ⊥ ship a table missing its delimiter row ∨ with ragged ∨ mismatched columns.
