# FORMAT — caveman encoding for SPEC.md

Applies to SPEC.md + spec-adjacent writes. ⊥ apply to code, error strings, commits.

## GRAMMAR
Drop articles, filler, aux verbs, hedging. Fragments fine. Short synonyms (fix>implement, big>extensive, run>execute).

## SYMBOLS
```
→ leads to/becomes   ∴ therefore/fix   ∀ every   ∃ some   ! must   ? optional/unknown
⊥ never/forbidden/nil   ≠ not equal   ∈ in   ∉ not in   ≤ at most   ≥ at least   § section   & and   | or
```

## PRESERVE VERBATIM
Code, paths, URLs, identifiers, numbers/versions, error strings, quoted strings.

## SHAPES
Invariant: `V<n> <TAG>: <subject> <relation> <condition>`
Task row (under §T): `id|status|task|cites`  — status `x` done, `~` wip, `.` todo. Escape literal `|` as `\|`.
Bug row (under §B): `id|date|cause|fix`
Research row (under §R): `id|claim|source|conf`

## RULE
Numbering monotonic — ⊥ reuse V.n/B.n/T.n. If cutting word loses fact, keep it. Compression, not amputation.
