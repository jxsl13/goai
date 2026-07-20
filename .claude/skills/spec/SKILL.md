---
name: spec
description: |
  Create, amend, or backprop bugs into SPEC.md at repo root. Sole mutator
  of the project spec. Triggers when the user asks to write a spec, start
  a new spec, distill a spec from existing code, add invariants, amend
  sections (§G, §C, §I, §V, §T, §B), or record a bug via backprop.
  Common phrasings: "write the spec for...", "new spec", "bug: ...",
  "amend §V.3", "distill spec from code", "spec this idea". Reads and
  follows FORMAT.md for the caveman encoding rules and pipe-table shape
  of §T and §B.
---

# spec — spec mutator

Read `FORMAT.md` at repo root if not already loaded. Caveman skill applies to all writes here.

## DISPATCH

Inspect user request and project state:

1. No `SPEC.md` at repo root AND args describe idea → **NEW**
2. No `SPEC.md` AND `from-code` in args → **DISTILL**
3. `SPEC.md` exists AND args start `bug:` → **BACKPROP**
4. `SPEC.md` exists AND args start `amend` → **AMEND**
5. `SPEC.md` exists, no args → ask user which mode

## INPUTS — spec is the sole mutator

The other verbs produce material; spec writes it. Ingest their handoff blocks
into the right section, show a diff, write on OK:

- **grill** → sharpened §G + §C
- **research** → §R rows (add the §R section if absent)
- **review** → drafted §V lines + the risk verdict
- **deepen** → §I/§V/§T amendments

⊥ rewrite a section the handoff did not name. Sectioned ownership (see FORMAT.md).

## NEW — idea → spec

Input: user idea. If it arrived fuzzy, prefer running **grill** first.

Steps:
1. Extract goal (1 line, caveman). → §G.
2. List constraints user stated or implied. → §C.
3. List external surfaces user named. → §I.
4. §R only if **research** ran — else omit the section (right-size).
5. Propose initial invariants. → §V (numbered V1…).
6. Break goal into ordered tasks. → §T pipe table, all status `.`, ids T1…
7. §B section with header row only (`id|date|cause|fix`).

Write the sections as spec/ fragments (spec/10-goals.md … spec/70-tasks.md,
FORMAT.md HIERARCHY layout) and run `make spec-render`; on an existing corpus
use the CLI adds instead. Show user the rendered SPEC.md. Ask: "spec OK?
`/review` if high-blast-radius, else `/build`."

## DISTILL — code → spec

Walk repo. Produce §G (infer from README/package.json/main entry), §C (infer from stack), §I (enumerate public APIs/CLIs/configs), §V (derive from tests and assertions), §T (one task per known TODO or missing test), §B (empty).

Caveman everywhere. Flag uncertain items with `?` in text so user can confirm.

## BACKPROP — bug → §B + §V

Input: `bug: <description>`.

Steps:
1. Parse bug description.
2. Find root cause (read relevant code).
3. Decide: would a new invariant catch recurrence? If yes → draft it.
4. Write the invariant FIRST (id auto-allocated):
   `go run ./internal/specgraph verif add -tag <TAG> "<testable rule>"`
5. Write the §B row citing it (guard chain lands as the fix cell's last sentence):
   `go run ./internal/specgraph bug add -cause "<cause>" -fix "<fix>" -guards V<n>`
6. If fix also changes behavior → `task add` / `task edit`.
7. The CLI re-renders + re-verifies per mutation; show the resulting diff.

Rule: every bug gets a §B entry. Invariant optional but preferred.
⊥ hand-edit SPEC.md ∨ spec/ table rows — the CLI is the writer (§V41).

## AMEND — targeted edit

Input: `amend §V.3` or `amend §T` etc.

Read the section (`specgraph entry get <id>` for one entry). Show current.
Ask user what changes. Apply via the CLI:
- table rows: `task edit -text/-cites/-state/-priority <id>`; removal: `entry rm <id>`
- defs: `goal|constraint|archinv edit <id> "<new text>"`; `verif add` for new §V
- prose sections (§I layer model, worker §RUN/§PERF/§GOAL/§NEXT, preambles):
  edit the `spec/` fragment directly, then `make spec-render`.
Show diff. Never silently rewrite sections user did not name.

## OUTPUT RULES

- Caveman format per `FORMAT.md` (incl. the HIERARCHY section: spec/ = source, SPEC.md = generated view).
- Preserve identifiers, paths, code verbatim.
- Numbering monotonic — ids come from the CLI (`next-id` / auto-allocation on add), never hand-numbered.
- §T `cites` ! list §V/§I deps: `task add -cites V2,I.api "impl auth mw"`.
- Every mutation goes through `go run ./internal/specgraph` — it re-renders the
  views and re-verifies (V36+V39+V41) in one transaction; a violating write is
  rejected with nothing written.

## NON-GOALS

- No sub-agents. Main thread writes.
- No dashboards, no logs, no state files beyond the spec/ tree and its generated views.
- No auto-build after spec. User invokes build explicitly.
