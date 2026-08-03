# nn training-host-loop perf notes (§V22-measured sweep, 2026-07)

Techniques applied to the `nn/` **host-side** training loops — the per-element
Go code that runs each step *around* the compute kernels (optimizer updates,
mixed-precision glue). A/B numbers were observed on darwin/arm64 (M-series,
GOMAXPROCS=12). Every change kept a **bit-identical** trajectory to the code it
replaced, proven by an explicit parity test, not just `go test`.

The compute (matmul/attention/conv) is at the BLAS/Metal/SIMD ceiling and is out
of scope here; this doc is about the host glue, where per-element dispatch and
per-element allocation still hid large wins.

## 1. The typed contiguous fast path (`flatF64`/`flatF32`)

The dominant pattern. An optimizer `Step` loop that updates a parameter
element-by-element via the widening accessors —

```text
for i := range p.Numel() {
    idx := tensor.Unravel(i, p.Shape())   // per-element multi-index
    gv := g.AtF64(idx...)                  // per-element dtype dispatch
    ...
    p.SetF64(newVal, idx...)               // per-element dtype dispatch
}
```

— pays `Unravel` + `AtF64`/`SetF64` dispatch on every element. When the
parameter and gradient are **contiguous** (the common case: a freshly-allocated
weight and its gradient), the flat row-major index *is* the storage index, so a
single dtype switch up front collapses the loop to a typed slice walk:

```text
if pf := flatF64(p); pf != nil {           // nil unless F64 & contiguous
    if gf := flatF64(g); gf != nil {
        for i, gv := range gf { pf[i] = update(pf[i], gv) }
        continue
    }
} else if pf := flatF32(p); pf != nil {     // F32 twin
    ...
}
// generic accessor loop kept verbatim as the strided/mixed-dtype fallback
```

`flatF64`/`flatF32` (in `nn/optim.go`) return the contiguous backing slice or
`nil`. The fast-path arithmetic is a **line-for-line transcription** of the
generic path (all math in `float64`; the F32 branch narrows only on store), so
the two paths are bit-identical. Correctness is guarded by
`TestOptimizerFastPathParity`: it runs each optimizer twice over logically-equal
parameters — once contiguous (fast), once through a transposed `Permute` view
(non-contiguous → generic) — and asserts exact `math.Float64bits` equality after
every step, for F64 and F32.

Measured (StepOnly fixture, best-of-2 interleaved):

| Optimizer      | f64   | note                                             |
|----------------|-------|--------------------------------------------------|
| SGD            | 15.4× | (earlier pass)                                   |
| Adam           | 7.8×  | (earlier pass)                                   |
| Sophia         | 3.5×  |                                                  |
| ScheduleFree   | 6.3×  | generic loop paid two tensor round-trips/element |
| CautiousAdamW  | 4.5×  |                                                  |
| Grokfast       | 5.1×  | gradient EMA filter, per step                    |
| GrokfastMA     | 4.7×  | gradient read once into the ring, then reused    |
| Lookahead      | 7.5×  | slow-weight average is pure read/write           |

**Where it does NOT apply (§C3, measured and left):** the low-rank / matrix
optimizers. GaLore/APOLLO/QGaLore only touch elements in their **1-D-parameter
fallback**; their 2-D weights go through a low-rank SVD projection (`GaLore.Step`
measures ~1.6 s dominated by the pure-Go SVD, refreshed only every `Gap` steps),
so the element loop is <0.1 % of the step. Muon is Newton-Schulz-matmul-bound.
A `flatF64` pass there would be non-winning churn — the lever is SIMD dequant/SVD,
not devirtualization.

**Not only optimizers — the mask builders (T919).** The same fast path applies to
any freshly-allocated contiguous tensor filled per element. `Dropout.Forward` and
`DropPath.Forward` built their Bernoulli / per-sample masks with a per-element
`Unravel`+`SetF64` walk on every training forward; the mask is `tensor.New`'d (so
contiguous and zeroed), and a typed slice walk — writing only the survivors,
`DropPath` sample-major — replaces the dispatch. Bit-identical (the RNG is drawn
once per element in flat order in every branch, proven by a parity test that
reconstructs the old walk on the same PCG stream). Measured best-of-6 on
`[16,128,768]`: Dropout **1.58×** (its 1.57M `rng.Float64` draws are the floor),
DropPath **13.1×** (only 16 draws — the per-element mask fill *was* the whole cost).
Allocs were already flat (14/15): `Unravel` is stack-elided, so the win is pure
dispatch/index-compute, not allocation — worth stating honestly.

## 2. Per-element *allocation* is worse than per-element dispatch

The standout of the pass. `MixedPrecision.Sync` (AMP) rounded each master weight
to half precision by calling a `roundHalf` helper **per element** — and that
helper allocated a one-element tensor and cast it on every call:

```go
func roundHalf(v float64, dt tensor.Dtype) float64 {
	h := tensor.FromFloat64(tensor.Shape{1}, []float64{v}).Cast(dt) // ALLOC per call
	return h.AtF64(0)
}
```

A 512×512 weight therefore did ~2.88 **million** allocations per `Sync` (56 ms).
The fix rounds the whole master in two **bulk** casts and copies the result into
the compute tensor through the contiguous fast path:

```go
src := m.Cast(mp.Dtype).Cast(w.Dtype()) // F32 master → half (round) → w's dtype, in bulk
copy(flatF32(w), flatF32(src))          // (contiguous fast path; generic fallback otherwise)
```

`tensor.Cast` applies the identical element-wise rounding, so the result is
bit-identical (the AMP round-trip and overflow tests pass unchanged). Measured:

| AMP op       | before   | after    | speedup | allocs/op          |
|--------------|----------|----------|---------|--------------------|
| Sync         | 56.06 ms | 1.116 ms | 50.2×   | 2,883,588 → 10     |
| UnscaleGrads | 2.184 ms | 0.221 ms | 9.9×    | (grad unscale)     |

**Lesson / grep to add to any base-perf sweep:** look for
`tensor.New | FromFloat64 | scalarTensor | Cast(` **inside** a `for … Numel()`
loop. An allocation (or a tensor round-trip) per element is far worse than an
`AtF64` dispatch and hides behind an innocent-looking scalar helper. A precise
scan of the whole `nn/` tree after fixing AMP found *no other* hot instance —
the rest were per-parameter/per-row (fine) or one-time construction (cold).

## 3. Every "sweep complete" claim has holdouts

Methodological note, repeatedly re-confirmed. The optimizer-Step sweep was
declared "done" more than once; each re-sweep on a fresh axis found more. The
reliable finder is a structural scan, not memory:

```sh
# functions that index tensors per-element but have no fast path yet:
awk '/^func /{fn=$0;u=a=f=0} /tensor\.Unravel/{u=1} /\.AtF64\(/{a=1} \
     /flatF64|flatF32/{f=1} /^}/{if(u&&a&&!f)print FILENAME": "fn;fn=""}' nn/*.go
```

That single scan surfaced the six optimizers in §1 that the previous "closed"
note had missed. Treat a floor claim as scoped to the path it measured; re-sweep
the adjacent classes (construction, quantization, weight-averaging, the
mixed-precision glue here) explicitly.

**`internal/perfscan` supersedes this awk (T920).** The scan above is now a real
`go/ast` tool: `make perfscan` (or `go run ./internal/perfscan ./...`). It parses
source rather than text, so comments and strings don't false-match and
build-tagged cgo backends are still scanned; it scopes the fast-path check per
function (a `flatF64`/`flatF32` presence silences the finding) and reports the
three patterns above — per-element dispatch, allocation-in-loop, and the
single-row batch wrap of §T917. It is **advisory**: a static check sees the shape
of a hot loop, never its temperature, so every hit still needs an A/B measurement
(§C3) and a bit-identity proof (§V22) before it ships. `-strict` makes it exit
non-zero for optional CI gating.

## See also

- `docs/perf-notes-cpu.md` — the `backend/cpu` compute-kernel sweep
  (bounds-check elimination, fused passes, the shared NEON transcendental leaf).
- `docs/perf-notes-lowlevel.md` — format/parser decode (`rawCopyLE` verbatim
  little-endian bulk copy) and the hostile-header overflow guards.
- SPEC.md §T910–T912 record the individual changes with their citations.

## AWQ's reconstruction error: an axpy through memory (T1183)

`reconErrMat` computes ‖(W−Ŵ)·X‖_F once per candidate scale in AWQ's calibration
search. At out=512, in=512, samples=128 it took **20.46 ms**, and 84% of that sat
on one line:

```go
for s := 0; s < samples; s++ {
    acc[s] += di * xf[base+s]
}
```

Every element pays a load and a store of `acc[s]` for a single multiply-add. The
accumulator is shared across the residual index, so taking eight `i` per pass
loads and stores it once for eight products.

| Benchmark | before | after | delta |
|---|---|---|---|
| `BenchmarkReconErrMat` (512×512×128) | 20.46 ms | 13.88 ms | **−32.1%** |

Width 8 and width 6 are within 0.2% of each other (13.88 vs 13.91 ms); width 4 is
14.09 and width 2 is 15.83.

### The compound assignment is not the same arithmetic

The obvious jam is wrong:

```go
acc[s] += d0*x0 + d1*x1 + d2*x2 + d3*x3   // NOT bit-identical
```

`+=` adds the **sum of the terms** to the accumulator, which associates
differently from the four sequential additions it replaces. The correct form
keeps the original order explicitly:

```go
a := acc[s]
a += d0 * xf[b0+s]
a += d1 * xf[b1+s]
...
acc[s] = a
```

### Width invariance is the gate that catches it

Under the compound form the frozen digests **held at widths 4 and 8 and moved at
2 and 6**. A test of the shipped width alone would have passed a wrong rewrite —
and width 4 was the first width tried.

A correct jam is bit-identical at *every* width. The rewrite above was verified at
2, 3, 4, 5, 6 and 8 before shipping; the digests cover `in ∈ {13, 30, 64, 1}` so
the tail is exercised, `in=1` skips the jammed loop entirely, and both dtype arms
are frozen.

That the value hashed is a `float64` **by bits** rather than compared to a
tolerance matters here for a second reason: Go may contract `x*y + z` into a
fused multiply-add, and the jammed form has to contract the same way term for
term.

### The check already claimed this shape and could not see it

`PS3075` describes "an item loop whose inner loop accumulates into a buffer that
does not vary with the item" — exactly this — but matched only `*ast.RangeStmt`.
This kernel's inner loop is `for s := 0; s < samples; s++`, so the scan walked
past the site its own description named. Reading both loop forms took the
tree-wide count from 55 to **81**.

## The QR backward's rank-1 update (T1184)

`vjp_qr.go` builds M = R·R̄ᵀ − Q̄ᵀ·Q and the second term is a rank-1 update
accumulated over the m rows of Q:

```go
for k := range m {
    qbk, qdk := qb[k], qd[k]
    for i := range n {
        qbki, mmi := qbk[i], mm[i]
        for j := range n {
            mmi[j] -= qbki * qdk[j]   // 50.7% of the benchmark
        }
    }
}
```

M does not vary with k, so the whole n×n matrix is walked once per row of Q — a
load and a store of `M[i][j]` for a single subtraction each. Taking eight k per
pass loads and stores it once for eight subtractions.

| Benchmark | before | after | delta |
|---|---|---|---|
| `BenchmarkQRVJP_256x128` | 10.90 ms | 7.87 ms | **−27.7%** |
| `BenchmarkQRVJP_128x64` | 1.133 ms | 1.020 ms | −10.0% |

The smaller cell moves less because that matrix fits in cache and the round trip
being removed is cheap there.

Eight is measured, not assumed: widths 12 and 16 regress on register pressure
(8.38 and 8.35 ms against 7.87). Every width from 2 to 16 leaves the digests
unchanged, which is the width-invariance gate from T1183 — the accumulator is an
explicit local, so the subtractions keep ascending k rather than being folded
into one compound assignment.

### Three spellings the check could not read

PS3075 describes this shape exactly and missed this site, because it hid behind
all three at once:

1. It accumulates with `-=`, and the check matched only `+=`. Subtraction is no
   more associative than addition and the fix is identical.
2. It addresses the buffer as `mm[i][j]`, and the check expected a bare
   identifier or a base-plus-variable index — so the root resolved to nothing.
3. It reaches the row through `mmi := mm[i]`, rebound on every pass of the item
   loop. A per-item test on the row variable alone hides the shared buffer
   behind it.

Reading all three took the tree-wide count from 81 to **95**. The index-crossing
rule in the root resolution is the one that keeps this sound: `dst[k][j]` with
`k` the item is the item's own row, and reporting it would name a jam with
nothing to hold.

## The MLA backward's key loop (T1187)

MLA's backward accumulates the query gradient into slots fixed by the **query**
— `dqC[i][d]` and `dqRrot[i][e]` — while the loop runs over keys. Each was
loaded and stored once per key for a single multiply-add. Six keys per pass hold
both in registers; `dkC` is per key and keeps its own store.

| Benchmark | before | after | delta |
|---|---|---|---|
| `BenchmarkMLAVJPSeq256` | 11.58 ms | 9.80 ms | **-15.4%** |
| `BenchmarkMLAVJPSeq256F32` | 11.50 ms | 9.62 ms | **-16.4%** |

Width 6 is the measured optimum (9.81 ms) with 8 a hair behind (9.83), 4 at 9.99
and 2 at 10.75. Every width from 2 to 8 leaves the digests unchanged.

### The narrowing arm needs the rounding kept per key

The F32 twin writes `float32` back after **every single accumulation**:

```go
dqcs[i*cols+hc+d] = float32(float64(dqcs[i*cols+hc+d]) + dS*float64(kcs[...]))
```

So the register held across the group has to be a `float32` rounded at each step.
Holding it in `float64` and rounding once would be a better-conditioned
computation — and a different answer. That is the case where the tempting version
of the transform is not merely riskier but strictly more accurate, which is
exactly what bit-identity forbids.

### Gate

Five digests over every returned gradient, not just the query one, since the same
loop writes `dkC` and `dvC` and a bad tail would leave one of those short. Under
a causal mask the key count is `i+1` and runs over every value from 1 to `seq`,
so one causal case exercises every remainder at once — including the lengths
where the jammed loop never runs.

### Still reported

The `dkRrot` second pass. It folds the shared decoupled-key gradient in ascending
`(head, i, j)` order, which is what keeps the whole VJP bit-identical; jamming
its item loop would reorder exactly that.
