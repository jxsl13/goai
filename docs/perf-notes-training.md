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

**External `perfscan` supersedes this awk (T920/T984).** The scan above is now
maintained at `github.com/jxsl13/perfscan/perfscan` and invoked through
`make perfscan`. `make perfscan-install` pins the reviewed release and uses
`GOPROXY=direct`; GoAI-specific identifiers live in the root `perfscan.yaml`.
It analyzes the active Go build through `go/analysis`, so comments and strings
do not false-match; inactive cgo/CUDA/Vulkan build-tag variants need a separate
invocation with the corresponding build environment. It reports a substantially
broader, upstream-maintained family of performance candidates. A static check
sees the shape of a hot loop, never its temperature, so every hit still needs an
A/B measurement (§C3) and a bit-identity proof (§V22) before it ships. GoAI's CI
uses `-exit-zero` to keep findings advisory while configuration, package-loading,
and analyzer failures remain fatal; omit it for a ratcheted findings gate.

## See also

- `docs/perf-notes-cpu.md` — the `backend/cpu` compute-kernel sweep
  (bounds-check elimination, fused passes, the shared NEON transcendental leaf).
- `docs/perf-notes-lowlevel.md` — format/parser decode (`rawCopyLE` verbatim
  little-endian bulk copy) and the hostile-header overflow guards.
- SPEC.md §T910–T912 record the individual changes with their citations.
