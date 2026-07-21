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
Invariant (under §V): a table row `| V<n> | <TAG> | <subject> <relation> <condition> |` — the TAG is its own column.

§G, §C, §I, §V, §R, §B, §T, §Bench — EVERY id-bearing section — are **CLEAN GFM MARKDOWN TABLES** — they MUST render as tables and pass mdlint's table checks (`table-ragged` + `table-separator-mismatch`). Each = a header row, a delimiter row, then data rows; every row has leading & trailing `|` and the SAME column count. Columns:

- §G (goals): `| id | goal |`.
- §C (constraints): `| id | constraint |` — any `(tag)` (e.g. C3's `(thr):`) stays INSIDE the constraint cell, ⊥ dropped.
- §I (arch invariants): a layer-model intro paragraph, then `| id | interface |` (the I.L* layer model, then an `INVARIANTS:` label, then the I* invariants — two tables of the same shape).
- §V (verification): `| id | tag | invariant |` — the TAG (`^\| (V\d+) \| <TAG> \|`) is its own column.
- §T (tasks): `| id | status | task | cites | state | priority |` — status `x` done, `~` wip, `.` todo; state ∈ done/wip/. (secondary lifecycle); priority ∈ high/med/low. Old rows may leave state+priority empty.
- §B (bugs): `| id | date | cause | fix |`
- §R (research): `| id | claim | source | conf |` — conf ∈ high/med/low/ref.
- §Bench (benchmark records): `| id | date | benchmark | machine | incumbent | metric | before | after | impact | cites |` — id class `BM<n>` (disjoint from §B: `BM1` ≠ a bug id). benchmark = what's measured; machine = host/config; incumbent = compared system + VERSION (e.g. `torch-mps 2.5`) ∨ `self` for a pure pre/post; metric = unit (ns/op, MB/s, tok/s, ms); before/after = pre/post (∨ incumbent-vs-GoAI) values; impact = COMPUTED speedup `before/after` (e.g. `1.38×`), never hand-typed; cites = the §T task + §V22. Records ALL benchmarks, pre- AND post-optimization; ascending by id; renders AFTER §T (the one section allowed past §T — its `## §Bench` header is invisible to the single-letter §-section regex so §T stays last, §V36).

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

## HIERARCHY (§V41)
`spec/` = source of truth, one file per section; `SPEC.md` + `SPEC-worker-*.md` = GENERATED views — ⊥ hand-edit them (TestRenderSync red).

| file | section |
|------|---------|
| spec/00-preamble.md | title + WORKER SUB-SPECS note |
| spec/10-goals.md | §G |
| spec/20-constraints.md | §C |
| spec/30-arch-invariants.md | §I |
| spec/40-research.md | §R |
| spec/50-verification.md | §V |
| spec/60-backprop.md | §B |
| spec/70-tasks.md | §T (sorts LAST among the single-letter sections → §T last, §V36) |
| spec/80-benchmarks.md | §Bench (benchmark ledger; renders after §T) |
| spec/worker/`<host>`/*.md | §RUN/§MODELS/§H/§Iw/§CPU/§GPU/§GAP/§PERF/§Tw/§GOAL/§NEXT |

render = lexicographic concat, one blank line between sections, byte-deterministic; `docgraph split` = proven inverse.
∀ id-bearing entry mutation via `go run ./internal/docgraph` (`task add/set-status/edit`, `bug add`, `bench add`, `verif|research|goal|constraint|archinv add`, `entry rm`) — ids allocated by the tool (`next-id`), ⊥ hand-numbered. Every mutation re-renders + re-verifies in one transaction. Prose blocks: edit the spec/ fragment + `make spec-render`.
Future lever (⊥ implemented): reserved id blocks per machine (e.g. worker T950–T999) to kill cross-host rebase races (§B96 class).
