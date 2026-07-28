# GoAI — Autonomous Loop Prompt

> Authoritative per-iteration instruction for the autonomous build loop.
> Referenced by the cron job; changes here take effect from the next fire.
> The truth is the `.spectackle/` spec bundle, reached only through the
> spectackle server — never by editing those files directly. This file is the
> only framing document; the retired planning prompt it used to cite has been
> removed along with the rest of the pre-spectackle spec system.

FULLY AUTONOMOUS, no questions back to the user. Work on the GoAI Go AI
library in this repository. Complete EXACTLY ONE item per fire, end to end.

## Procedure

0. **Orient.** `make spec-state` for the one-call picture (items, rules, graph,
   swarm, drift, health). If siblings may be active, run `swarm` first and read
   their learnings before forming any hypothesis.
1. **Selection:** take the next approved item with no open dependency; `get` on
   its ID returns the full brief. State the ID and its definition of done.
   Before drafting anything new, search the rejection corpus (`find` with
   `scope=rejection`) — a sibling may have already failed at it.
2. **Claim:** `work op=start` leases the scope and returns a worktree root. Do
   ALL code edits, builds and benchmarks under that root; spectackle paths stay
   repo-relative. A scope conflict names the holder — pick different scope, never
   wait idle.
3. **Build:** implement against the contracts the item's context pack names.
   Anything genuinely new goes `draft` → `grill` → explicit approval →
   `rule op=add`; never invent a contract mid-implementation.
4. **Acceptance:** parity goldens against the reference tolerance (generated via
   the `.venv` Python/NumPy when needed), gradient checks, cross-backend
   agreement and property tests — the `NUM-*` contracts.
5. **Optimization work:** first push PURE Go to its ceiling
   (algorithm→layout→simd/avo/NEON→goroutines, every rung green + a benchmark
   delta), then the cgo gate: merge cgo only as an optional build-tag backend
   when a benchmark beats the `BUILD-004` threshold against the fully optimized
   pure-Go version — otherwise discard and record why.
6. **Failure** → backprop: trace the root cause, and where a contract would have
   caught it, add or tighten one before moving on. Never loosen tolerances or
   skip tests. A rejection requires a note, and that note is what a future
   sibling reads instead of repeating the work.
7. **Platform check:** pure Go WITHOUT a C toolchain must be green —
   MANDATORY via `CGO_ENABLED=0 go vet ./...` or `go test ./...` (`BUILD-002`:
   `go build` alone compiles NO _test.go files and misses missing build tags);
   accel stays behind build tags with a fallback. Full sweeps (`go test ./...`)
   ALWAYS with `-timeout 1800s` (llamagpu exceeds the 600s default) and the exit
   code checked UN-PIPED — `| grep | tail` masks a FAIL as exit 0 (`CI-002`).
8. **Completion:** `check` until it reports ok — drift records mean spec and code
   disagree and are resolved by fixing one or the other. Then `work op=submit`
   (gates, merges, propagates state), release any explicit lease, and add a short
   entry to `CHANGELOG.md`.

Cost rule of thumb: the server returns IDs and spans, shell `grep` returns file
contents. Use `find`/`get` for locations and structure; read raw bytes only once
you know the single file you want.

## Performance grind — STANDING MANDATE (user directive 2026-07-21)

Performance leadership is a PERMANENT priority, not just an empty-backlog fallback.
When the active item does not dictate otherwise, advance the perf front:

1. **Beat every incumbent.** The goal is for GoAI to be FASTER than the incumbent
   (torch / numpy / sklearn / llama.cpp / tiktoken / safetensors / …) in EVERY honest
   benchmark. Where a real hardware ceiling exists (BLAS / MPS / silicon / memory
   bandwidth), say so HONESTLY and pivot to the structural-advantage frontiers
   (no-interpreter, no-FFI-binding overhead); keep grinding every gap that is winnable.

2. **Honest, statistically correct benchmarks — MANDATORY.**
   - Every perf claim gets a PRE- and POST-optimization measurement (`PERF-002` same-machine
     A/B, best-of-N INTERLEAVED — A,B,A,B…). Report the median, not a lucky minimum.
   - Compare LIKE-FOR-LIKE ONLY. NEVER put different quantizations, dtypes, or configs
     in the same comparison row — GoAI-Q4_K vs llama.cpp-Q8 is meaningless (4.5-bit vs
     8-bit). Compare GoAI-Q4_K vs incumbent-Q4_K, GoAI-Q8 vs incumbent-Q8, SEPARATELY.
   - Record the incumbent NAME + VERSION and the exact workload / shape / machine
     (`PERF-003`). A number without its incumbent version + config is not a comparison.
   - Ship MEASURED wins only (`BUILD-004`); a claimed win with no committed measurement
     path violates `PERF-003`. Record the numbers in `docs/benchmarking.md`.

3. **Backpropagate generalizable wins into perfscan.** Whenever an optimization removes
   an anti-pattern that could recur (a per-element dispatch/alloc, a batch-API-wrapped
   single item, a re-encode-instead-of-slice, a copy a verbatim-bulk move avoids), ADD
   or extend a detector in `internal/perfscan` so the whole tree is swept for it
   automatically. A one-off fix that leaves the class undetectable is HALF-DONE — the
   tool is how each win compounds across the codebase.

4. **New optimization field?** When you discover a fresh axis/domain to optimize,
   re-read this file and the `base-perf-sweep` notes before diving in.

## Research rule (mandatory — mitigates the StructuredOutput failure)

NEVER use the built-in `/deep-research` workflow for external research: it
forces every sub-agent into a `StructuredOutput` schema call that fails 5×
under rate limits and crashes the whole workflow
(`StructuredOutput retry cap exceeded`).

ALWAYS use the repo's own `research-lite` workflow
(`.claude/workflows/research-lite.js`) instead:
- **small scope**: exactly ONE focused question per run (protects context);
- **schema-free**: no `agent({schema})` → the StructuredOutput failure is
  structurally impossible;
- **compressing sub-agents**: every angle agent returns ≤6 condensed lines,
  one synth agent condenses to ≤8 lines → never blows the context;
- **graceful**: dead agents → `null`, filtered, never a throw.

Invocation: `Workflow({ scriptPath: ".claude/workflows/research-lite.js", args: "<one precise question>" })`.
**Validation ladder (`NUM-008`, mandatory for every algorithm implementation):**
- Tier 1 = bit-/tolerance-exact parity against the official reference lib
  (torch/sklearn/gguf-py/safetensors). Necessary, NOT sufficient.
- Tier 2 (final authority) = the **scientific paper** defining the algorithm
  (arXiv/DOI/canonical textbook) — the implemented formula MUST match the
  paper's equation, cited in the item that introduced it.
- File formats have no paper → the defining source is the format spec /
  reference implementation (record it explicitly as such; never invent a
  paper).
File formats: always round-trip + fuzz (the `FMT-*` contracts).

## Autonomy rule

On ambiguity or a design decision, do NOT stop, do NOT ask — make a
scientifically grounded default choice, document it in
an ADR item via `decide`, plus a contract where one is warranted, keep building.
Only genuine hard blockers (broken toolchain) → a short PushNotification,
otherwise continue. Commit and push authority is currently an OPEN DECISION
(`ADR-0031`): the migrated constraint said working-tree-only with explicit
permission, while the repository's actual history is autonomous branch+PR
pushes. Until it is answered via `decide op=answer`, follow the conservative
reading and do not push without explicit permission. The loop NEVER runs out of
work: when every approved item is done, proceed per the "Empty backlog" rule
below.

## Never idle (mandatory — user directive 2026-07-14)

NEVER be idle. Every fire must do productive work — never sit and wait, and
never spend a fire only re-scheduling the next wakeup. Waiting on an external
gate is NOT a reason to idle: when one thread is blocked, advance another.
Concretely, when blocked on a long-running check (waiting on CI, a benchmark or the hourly
window), a CI run, or a wakeup deadline, keep BUILDING locally — commits
accumulate in the working tree and push when the window opens. Always have
a next productive action from this list, in priority order:

1. Continue the current task's next slice/step.
2. Harden what just landed: more tests (edge cases, fuzz per the `FMT-*`
   contracts, gradient checks per `NUM-002`), documentation, benchmarks.
3. Verify-ahead the next task (research + primary-source extraction so the
   implementation is unblocked) — the verify-first discipline.
4. The Empty-backlog rule below (gap research, then beat-the-incumbents).

The ONLY legitimate idle is a genuine external wait with NO available local
work — which, given rules 1–4 and the empty-backlog rule, does not occur.
Idle-rescheduling a wakeup with productive work available is a process
failure. (Context-length is not a blocker either: if the context is
saturated, do the smallest safe useful increment and commit it, don't idle.)

DEFAULT WHEN IDLING (user directive 2026-07-14): whenever you would otherwise
idle — no current-task step, nothing to harden, nothing to verify-ahead —
spend that time BEATING THE PYTHON/C++ INCUMBENTS' PERFORMANCE. Pick a hot
path GoAI implements (matmul, attention, a decode step, a quantized kernel,
a tokenizer) and make it FASTER than the reference (torch/llama.cpp/ggml/
numpy), proven with a clean A/B benchmark (identical shapes + hardware,
warm-up excluded, variance reported, `PERF-002` discipline; `make bench-compare` /
`make bench-python`). Every measured win — or an honestly documented deficit
with a root-cause analysis and the next lever — is one item and one
docs/benchmarking.md entry. This is the empty-backlog rule 2 promoted to the
standing default: idle time is performance-leadership time.

## Empty backlog rule (mandatory — the loop generates its own work)

When no approved item is left, do NOT idle. In this order:

1. **Topic discovery:** autonomously research new AI topics, methods,
   architectures, formats, and techniques that this library does NOT yet
   implement — online search (current papers, llama.cpp/vLLM/SGLang release
   notes, architecture roundups, framework changelogs) plus a repo
   cross-check so only REAL gaps are booked (verify every
   candidate against the code before calling it a gap). Book each confirmed
   gap as a drafted item citing the research that documents source and scope, and
   work them per the normal procedure, one task per fire.
2. **Beat the incumbents:** once the library has implemented everything
   discoverable as far as possible, the standing task becomes performance
   leadership — make every implementation BETTER and FASTER than the
   industry-standard implementations (usually Python-with-C++-kernels or
   pure C++: torch/llama.cpp/ggml class), and PROVE it with clean,
   industry-standard benchmarks: identical workloads and shapes on identical
   hardware, warm-up excluded, repeated runs with variance reported,
   tokens/s / latency / GFLOPs as the branch-standard metrics, methodology
   and comparison scripts committed (`make bench-compare` /
   `docs/benchmarking.md` discipline, `PERF-002`: measure real workloads, A/B via
   file-toggle). Each measured, documented win (or honestly documented
   remaining deficit with root-cause analysis) is one item.

Rule 2 never completes — faster incumbents, new hardware, and new workloads
keep appearing — so the loop always has a next task.

## Historical record

The pre-spectackle narrative that used to sit here has been removed: its task, bug,
research and verification identifiers no longer resolve to any live entry, and keeping
a second, retired description of how the loop worked invited following it. The
invariants it carried are EARS contracts in the `.spectackle/` bundles; the ledgers it
cited are under `docs/history/`, kept only because live citation contracts reference
their ids. Everything else remains in git history.
