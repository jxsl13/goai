# Perf notes — memory order, sparsity, and selection

A campaign log for the host-side optimizations that turned out to be about **how data is
visited** rather than how much arithmetic it costs. Every figure is M2 Pro, darwin/arm64,
go1.26.5, measured interleaved over three alternations with min-of-three runs per arm.

The three families below account for almost every win. They are worth recognizing by shape,
because each has a fix that is bit-identical and a fix that is not, and the difference is not
visible from the profile.

## 1. Walking a matrix along the wrong axis

The recurring shape: an inner loop whose index into a row-major buffer **multiplies the inner
loop variable** by a stride, so consecutive iterations jump a whole row.

```go
for c := range dk { // outer: additive
	for r := range dv { // inner: SCALED by dk
		S[r*dk+c] *= at[c] // a different cache line every step
	}
}
```

Correct traversal puts the inner variable in the additive position. `PS6011` finds the
difference from the AST alone by asking only which loop variable is being scaled.

**Three fixes, and which one wins is a measurement, not a rule.**

| Fix | When | Measured |
|---|---|---|
| Interchange the loops | the body is a pure elementwise update | KDA decay (most of **1.75×**) |
| Block four adjacent outer indices | the body accumulates | Sinkhorn Kᵀu **2.65×**, NSA P·V **2.40×**, MoBA **1.85×**, Retention **1.54×/1.70×**, DoRA **2.5×/3.3×** |
| Gather into contiguous scratch | neither is available | WKV VJP **1.43×/1.60×** |

The gather form is the one to remember, because it is the least obvious. WKV's backward is a
recurrence over `t` **within** a channel, so the axes cannot be interchanged at all — but the
column can be copied out once per channel and the O(seq²) inner work then runs sequentially.

**Blocking beats an axpy sweep while the data is cache-resident.** Sinkhorn's transposed half
was also written as a scatter into an accumulator vector, touching `k` once instead of n/4
times. It is bit-identical and beats the original 2.39×, and it **lost** to register blocking
(14.60 ms against 13.34 ms). 512² f64 is 2 MB and sits in L2, so the extra passes cost no DRAM
traffic while a memory accumulator forces a store-to-load round trip per FMA
(`PERF-ACCUM-RESIDENCY-001`).

**A benchmark below the cache threshold measures this family as worthless.** SymEig's
eigenvector relayout is **1.05× at n=64** and **1.89× at n=128** — 64² f64 is 32 KB and
resident, 128² is not. Sized only at n=64 it reads as noise
(`PROC-BENCH-CACHE-THRESHOLD-001`).

**Not every strided walk is fixable.** SymEig's `m` rotation cannot use row reads even though
`m` is symmetric: the (p,q) block emerges from the two rotation loops through different
orders of the same operations, so it can differ by one ulp — and a cyclic sweep visits every
pair, scattering that asymmetry across the matrix. Correcting the current pair is necessary
and nowhere near sufficient (`NUM-SYMMETRY-NOT-EXACT-001`).

## 2. Scanning a full range to use a sparse subset

Masked attention computes a score for every key, marks the excluded ones, and then walks the
whole range again for each output channel — multiplying by an exact zero for most of them.

Collecting the surviving indices **in the pass that already walks every key** turns
O(range · d_k) into O(survivors · d_k):

| | mask density | measured |
|---|---|---|
| DSAAttention | 64 of 1024 keys | **1.35×** |
| NSABranches | only the selection branch is sparse | **1.25×** |
| MoBA | 256 of 512 keys | **1.12×** |

The gain scales with how much of the mask is off, which is why the three differ — that is the
shape, not measurement noise.

Bit-identical for finite values: a masked key has `scores[j] == 0` exactly, and `o + 0*v` is
`o`. It differs in exactly one case, and it is an improvement: if a masked key's `v` were ±Inf
or NaN, the old code propagated NaN through a position the mask excludes.

**This family has no scan rule, deliberately.** The loop has no guard to find — it just
multiplies by a value that happens to be zero, and nothing in the AST says the array is
sparse. The guarded relative (`if !mask[j] { continue }` with a mask the outer loop does not
touch) *is* detectable, but once the soundness test is added — the mask array must not be
rewritten by the outer loop — it matches nothing here, because masks in this codebase are
per-query or per-output by construction. See `PERF-SCANRULE-EMPTY-001`.

## 3. Sorting to answer a membership question

```go
sort(idx)
for r := 0; r < k; r++ {
	drop[idx[r]] = true // membership, not order
}
```

The order past `k` is computed and discarded. A selection answers it in O(n).
**WandaPrune: 282 ms → 55 ms (5.1×)**, and 348 ms → 55 ms combined with the panel transpose.
`PS6013` finds the shape.

**Two preconditions, and both must be checked rather than assumed.** The comparator must be a
**total order** — Wanda's is score ascending with ties broken by input index, and indices are
unique, so no two elements compare equal and the k-smallest set is uniquely determined. And
the consumer must read **membership rather than position**, since a selection leaves the
prefix in arbitrary order.

MoE's routers are the counterexample. They also sort and take a prefix, but the order of
`experts = idx[:k]` decides the order the expert outputs are summed downstream, so a selection
would change the result. The rule stays silent there, which is the right answer.

That same total order is why Lomuto partitioning cannot degrade on a column of identical
scores — index tie-breaking means there are no duplicate keys. Median-of-three pivoting is
still required: score columns are `|w|·‖x‖` products and frequently near-sorted, exactly the
shape that takes a first-element pivot quadratic.

## Allocation is a separate axis, and often the larger one

Several changes moved wall-clock barely and allocations by three orders of magnitude:

| | time | allocations |
|---|---|---|
| WandaPruneNM | 168 ms → 104 ms | 2,097,169 → **16** |
| MoBA | 66.8 ms → 59.5 ms | 20,489 → **12** |
| DSAAttention | 19.4 ms → 18.8 ms | 1,547 → **13** |
| NeuralMemory.Scan | 10.5 ms → 4.0 ms | 24,525 → **3,305** |

Two causes dominate. `sort.Slice` and `sort.SliceStable` reach their swap through
`reflectlite.Swapper`, which allocates on **every call** (`PS6009`) — ruinous when the sort is
inside a per-token or per-column loop. And a `map[int]bool` used as a small integer set,
constructed per query, where a hoisted `[]bool` does the same job.

Always report allocations beside ns/op (`PROC-BENCH-MEMAXIS-001`). One change in this campaign
shipped a **31× memory regression** behind a 2.80× speedup because the commit reported only
wall-clock.

## Traps worth knowing before the next pass

**Naming a subexpression does not pin it against FMA.** A fused path reproducing separately
rounded backend ops must round *every* product explicitly. `inc := g[i] * th` assigned to a
local and used in `s = float64(s*et) - inc` is still inlined and contracted to `fma(-g[i], th,
…)`. This cost three attempts on Titans, because the branch computing `s = -inc` — a negation
with nothing to fuse into — always matched while every later step was off by one ulp, which
reads as a logic bug. `PS6012` finds functions that pin some products and not others.

**A green probe on synthetic data is not evidence.** The Titans divergence was input-dependent:
a diff harness on convenient values passed and proved nothing. Only re-running it on the real
projected and normalized inputs found the defect.

**Mutate your property tests before believing them.** Wanda's selection tests passed against a
generator whose fill loop returned early, so every column was near-constant. They passed
anyway. Inverting the comparator and confirming red is what made them evidence.

**Replay a new scan rule against pre-fix source.** Four rules in `internal/perfscan` failed to
find the case they were built from — variously because the loop sat behind an `if`, behind a
closure, or because `calleeName` collapses a qualified call to its selector. Three were caught
only by replay, not by reading the detector.
