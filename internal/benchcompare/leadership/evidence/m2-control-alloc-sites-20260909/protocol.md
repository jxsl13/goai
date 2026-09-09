# Prospective Softplus allocation-site protocol

Status: corrected design, independent follow-up PASS in `corrected-review.txt`.
No instrumentation or profile run exists yet. This document supersedes conflicting protocol language
in the immutable initial `research.txt`, incorporating every required correction
in `plan-review.txt`. Research: `R-01M23VWNB5FRP`.

## Scope and interpretation

Investigate the repeated parallel Softplus allocation differences in the completed
fixed-count diagnostic. This is a separate, perturbed, process-wide rate-1 profile
experiment, not a latency benchmark, exact attribution of the earlier timed
callback, or qualification of either rejected GELU candidate. The prior raw data,
V1/V2 verdicts, and `ARM64-F64-GELU-CONTROLS-001` remain unchanged.

Only Softplus F64, shape `{1 << 18}`, seed 3, contiguous input, CPU backend,
`backend.Execute(ctx, backend.OpSoftplus, ins, nil)`, discarded outputs. Use the
same input/context setup as `BenchmarkSoftplusF64_256K_cpu`, outside accounting.
Exactly one successful Execute warms dispatch/pool paths. A dedicated
`//go:noinline` helper then directly performs exactly 1,024 Execute calls. No
logging, formatting, setup, snapshots, or assertions inside that helper. An
Execute error is returned and invalidates the invocation after immediate ending
counters are captured; it must never be hidden or counted as a completed region.

New byte-identical, test-only instrumentation and separately named binaries are
required at these detached pins:

- Old: `dd1e779eb085bb621ed5dafff0a4351636b6e656`.
- V2: `361955ac35eaada696045f2cf828817b650174ec`.
- SDK: Go 1.27.1, `/private/tmp/goai-go1271-j1cLvq/go`; darwin/arm64,
  `CGO_ENABLED=0`, `GOEXPERIMENT=simd`, `GOTOOLCHAIN=local`.

Freeze and retain exact source/helper/SDK/binary hashes, build commands and
environment before any matrix run. All earlier binaries and source evidence are
immutable. Missing instrumentation details, schema, buffer capacity and analyzer
tests must be fixed in an implementation task before execution, not chosen after
seeing profiles. The helper remains opt-in and must do no work when disabled.

## Process and accounting boundaries

Each fresh process starts with exactly `GODEBUG=memprofilerate=1`, selected
`GOMAXPROCS=1` or `12`, `GOAI_ALLOC_SITE_DIAGNOSTIC=softplus`, and a unique output
directory. Validate the selector, procs, lifetime startup-rate setting, and
`runtime.MemProfileRate == 1`. Do not pass `-test.memprofilerate` or
`-test.memprofile`, or change the rate in the test. Reject reused/nonempty output
directories and never overwrite evidence.

Preallocate fixed-capacity pre/post `runtime.MemProfileRecord` buffers and all
MemStats destination storage during setup. Open all artifact files before the
pre boundary. Reject capacity overflow; no resize/retry of a primary snapshot.
Keep the buffers alive through serialization. The per-invocation sequence is:

1. Finish setup and one warmup. Call `runtime.GC()`.
2. Read MemStats `preRawBefore`, call `runtime.MemProfile(pre, true)` into its
   preallocated buffer, then read MemStats `S`. Retain both counts and the returned
   length/success flag. Require zero `Mallocs` and `TotalAlloc` movement between
   these reads; otherwise retain the run as invalid. Do not log or assert between
   these accounting calls.
3. Call the noinline region once, then read MemStats `E` immediately, before
   checking/formatting any region error. `E-S` is the immediate process-wide
   region total, not a per-goroutine total. No extra calibration or warmup.
4. Call `runtime.GC()` once as the required publication boundary. Read MemStats
   `postGC`, capture raw post records with `runtime.MemProfile(post, true)`, then
   read MemStats `postRaw`. Retain all boundaries and snapshot status/capacity.
5. Only after raw post capture, validate/aggregate/symbolize and serialize JSON
   and one cumulative post `allocs.pprof`. Then read MemStats `tail` and report
   it separately. The final tail-report write necessarily occurs after this read
   and is explicitly outside the tail count; do not pretend to measure a report's
   own unbounded recursive serialization cost.

The pinned runtime's `runtime.GC` completes sweeping and publishes a stable
profile before returning. This does not make the two windows atomic or equal.
Retain exact nonnegative cumulative counters and checked differences; no floats
in byte/object arithmetic. Report the signed discrepancy between immediate
totals and summed profile allocation deltas as disclosure, never attribution or
an invalidity condition. Tiny suballocations, concurrent process activity,
different boundaries and profiling perturbation prevent an equality claim.

## Public raw records and collision-safe deltas

Retain every raw pre/post row, its original ordinal, all four signed 64-bit
cumulative fields, and the complete public 32-PC array. The public API has no
ObjectSize. Derive allocation **slot** size only when `AllocObjects > 0`:

- All four fields must be nonnegative, with free counts/bytes no greater than
  their allocated counterparts.
- Require `AllocBytes % AllocObjects == 0`, positive derived slot size, and exact
  checked `FreeBytes == FreeObjects * slot_size`.
- If `AllocObjects == 0`, all four numeric fields must be zero. Retain the row
  as `zero-active/size-unknown`, including its PCs and multiplicity, but exclude
  its zero numeric contribution from size-keyed maps. Never divide by zero or
  invent a slot size. A positive post record can have zero numeric pre-value.
- Do not use unexported runtime APIs/linkname to recover internal ObjectSize.

Within an invocation, the visible numeric key is `(slot_size, ordered nonzero
Stack0 prefix)`. **Sum** all four counters for colliding rows at each snapshot;
never overwrite. Retain contributor ordinals and collision cardinality alongside
the unaggregated records. The 32-PC public stack can truncate different deeper
runtime buckets to the same visible prefix.

Compute checked signed post-minus-pre deltas over the union of aggregated keys;
absence means zero. Any decreasing cumulative field for a same visible key or
integer overflow invalidates the invocation. All-zero unknown-size rows remain
a separate multiset, not numeric map keys. Retain every key, including unchanged
ones; no prefilter to an expected output or scheduler site.

## Symbolization, normalization and grouping

After raw post capture, symbolize each retained stack with `runtime.CallersFrames`
and keep raw PCs plus all emitted frame PCs, function names, files and line
numbers. Preserve frame order, repeated frames and inline expansion. An active
nonzero record with an empty/unresolved frame required for its key invalidates
cross-binary site comparison for that invocation. Retain it and the diagnostic
error; do not silently drop or relabel it. Unknown-size zero rows may disclose
unresolved frames without contributing to arithmetic.

The cross-binary numeric key is **only** `(slot_size, ordered fully qualified
function-name stack)`. Do not key on PCs, absolute paths, source lines or function
offsets. Sum all colliding visible raw keys and retain contributor keys and
normalized collision cardinalities. This is deliberately coarser function-stack
aggregation, not proof of a unique source allocation statement. Source locations
remain diagnostic evidence. Current old/V2 line-count neutrality is not a safe
identity contract.

Every numeric stack belongs to exactly one of these groups, in this order:

1. Caller: contains the exact noinline region helper function name.
2. Worker: lacks the caller helper and contains a known
   `github.com/jxsl13/goai/backend/cpu.poolWorker` frame.
3. Other: every remaining process/runtime stack, without exclusions.

Worker stacks are only **temporally associated** with the accounting window;
persistent workers cannot inherit the submitter's stack tag. Caller stacks also
do not define a goroutine-scoped total. Serial inline execution is a structural
contrast, not a causal experiment. Preserve all groups and all sites.

## Frozen matrix, completeness and reporting

First old/old, then old/V2. Each phase has campaigns 1–3, pairs 1–7, process counts
1 and 12, arms A and B: 84 invocations per phase, 168 total. For each campaign,
visit pairs ascending; within each pair use process order 1 then 12 for odd
campaigns and 12 then 1 for even campaigns. For each pair/procs, arm order is AB
when pair+campaign is even, otherwise BA. Old/old uses the exact same old binary
bytes in both arms. Old/V2 uses old A and V2 B. One process per invocation, one
control per process, no overlapping owned build/test/profile/benchmark work.
Retain ordinary background-load disclosure; do not claim an otherwise idle OS.

The future runner must retain every literal command, order metadata, exit
status, stdout/stderr, exact counters, raw rows, aggregates and cumulative
profile. Check pins before and after each phase. Missing, replaced, failing,
wrong-order/pin/N/procs/rate, overflowing, undecodable, unsymbolizable-active,
decreasing-counter or inexact-arithmetic observations make the planned phase
unanalyzable. Preserve invalid observations; no silent retry, substitution,
partial success claim or post-hoc scope extension. Synthetic tests must exercise
zero-active rows, both collision layers, line/PC shifts, worker grouping,
arithmetic beyond 2^53, overflows/decreases, malformed and missing data.

Decode the cumulative pprof as corroboration, never subtract it against an
absent pre-serialization profile. Retain every planned paired B-minus-A exact
immediate total and normalized site delta, ranges, medians and sign counts.
Do not subtract the old/old variation as a noise floor. Any repeatable old/V2
site pattern is a candidate association in this profiled harness only; absence
is not equivalence. Neither outcome establishes GELU causation, unprofiled
behavior, a performance gain, or grounds to revise any existing rejection.
