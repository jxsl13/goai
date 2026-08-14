# backend/cpu perf notes (last-percentile sweep, 2026-07/08)

Techniques researched and applied during the §V22-measured optimization pass of
`backend/cpu/`, with the A/B numbers observed on the darwin/arm64 (M-series,
GOMAXPROCS=12) host, `CGO_ENABLED=0`. Every change kept the documented parity
class (bit-identical where claimed, ulps otherwise); no tolerance was relaxed.

## 1. Bounds-check elimination via hoisted row subslices + `range`

The single biggest lever of the pass. The typed attention/norm/softmax cores
indexed 2-D data as `x[base+d]` inside `for d := 0; d < dk; d++` loops: every
access pays a bounds check plus full address arithmetic, and the generic
(`[T float32|float64]`) instantiations made it worse. Rewriting each inner loop
to hoist a three-index subslice per row and drive the loop with `range`

```go
qr := q[qBase : qBase+dk : qBase+dk]
kr := k[kBase : kBase+dk : kBase+dk]
for d, qv := range qr {
	s += float64(qv) * float64(kr[d])
}
```

lets the compiler prove the range index in-bounds (range-loop accesses on the
ranged slice are check-free) and shrinks the addressing to a single induction
variable. The three-index form also pins `cap`, which helps the prove pass and
prevents accidental aliasing growth.

Measured (median of interleaved A/B, `-count≥4`):

| kernel | shape | before | after |
|---|---|---|---|
| `mhaBwd` | 512×512, 8h causal f32 | 53.0 ms | 21.6 ms (−59%) |
| `mhaFwd` (incl. §2) | same | 17.4 ms | 14.2 ms (−18%) |
| `flashAttnTyped` (incl. §2) | 512×512 8h blk64 | 27.4 ms | 13.4 ms (−51%) |
| `retentionBwd` | 512×64 | 6.24 ms | 5.39 ms (−14%) |
| `softmaxTyped` | 2048² f32 | 6.3 ms | 5.6 ms (−8%) |

Counter-example: the same rewrite on the norm *forwards* (rmsNormFwd /
layerNormFwd) measured flat — those loops are memory-bandwidth bound, so the
saved checks hide behind loads. Reverted per §C3. BCE only pays where the loop
is issue-limited, not bandwidth-limited.

Sources: [Go 101 — BCE](https://go101.org/optimizations/5-bce.html),
[Bounds checking in Go](https://unskilled.blog/posts/bounds-checking-in-go/),
[Ardan Labs — BCE in Go](https://www.ardanlabs.com/blog/2018/04/bounds-check-elimination-in-go.html).

## 2. Loop-interchange to contiguous streams, division hoisting

`mhaFwd`/`flashAttnTyped`/`retentionFwd` accumulated the output with `d` outer /
`j` (keys) inner, reading `v[j*kvDM+kvOff+d]` — a stride-`kvDM` walk per output
element, and `mhaFwd` recomputed `row[j]/sum` for every `d` (sk·dk divisions).
Swapping to `j` outer / `d` inner with a small f64 accumulator buffer keeps each
output element's ascending-`j` summation order (bit-identical values — division
is deterministic, computing `row[j]/sum` once yields the same float) while
making `v` reads contiguous and collapsing the divisions to one per key.
Measured alone on `mhaFwd`: 21.2 → 16.5 ms (−22%); `retentionFwd` 3.2 → 1.88 ms
(−40%); `flashAttn` 27.4 → 22.8 ms (−17%) before the §1 rewrite stacked on top.

The same reassociation-free trick tiles the norm backwards' dγ/dβ column
reductions: keep each column's ascending-row order but walk the matrix row-major
in per-worker column bands (`layerNormBwd` 818 → 623 µs, `rmsNormBwd` 809 →
540 µs on 512×1024 f32).

## 3. FP contraction (FMA) is context-dependent — pin it at parity edges

The Go spec allows the compiler to fuse `x*y + z` into one rounded FMA "possibly
across statements", and gc's choice is context-dependent: during this pass, the
norm-backward dγ accumulation flipped between fused and unfused when *unrelated
code elsewhere in the package changed*, moving the result a few ulps and
tripping the 1e-15-relative parity test against `backend/ref` (whose own
contraction also shifts as it is edited). Two spec-guaranteed pins exist:

- `acc += float64(u * x)` — an explicit conversion **forces rounding**, i.e.
  plain mul-then-add, never fused;
- `acc = math.FMA(u, x, acc)` — always fused (single rounding; compiles to a
  native FMADD on arm64, no call).

Both are deterministic regardless of what the optimizer does around them.
`norm.go` now fences the tiled dγ/dβ accumulations with the conversion form,
matching ref's devirtualized core (which rounds `g·x̂` through a scratch slice
before accumulating). Cost: not measurable vs the unfenced loop on the norm
benches. Lesson for this package: any reduction that must satisfy an
ulps-tolerance against another compilation unit should pin its contraction
explicitly rather than trust the default.

Sources: [go#17895 — spec: allow FMA](https://github.com/golang/go/issues/17895)
(explicit casts force rounding; parentheses do not),
[go#25819 — math: add guaranteed FMA](https://github.com/golang/go/issues/25819),
[FMA & FP consistency](https://siboehm.com/articles/23/Inlining-FMA-FP-consistency).

## 4. Fork/join barriers dominate medium ops — fuse row-parallel passes

Profiling `conv2d` (581 µs/op) showed ~75% of samples in
`pthread_cond_wait`/`_signal`/`findRunnable`: the kernel ran im2col, GEMM and
the output scatter as three sequential `parallelWork` barriers, and the pool
workers parked/rewoke between each. All three stages operate on the same
disjoint row bands, so they fuse into ONE pass per band (fill rows → GEMM band →
scatter band): identical per-row operations, bit-identical output, two fewer
fork/joins, and the band's im2col rows are still cache-hot for the GEMM.
Measured: conv2d fwd 588 → 480 µs (−18%); conv2d bwd (fill+dXcols fused, and
col2im+dX-store fused per image) 1376 → 1299 µs (−6%).

## 5. Kill transposes: transposed-A GEMM band

conv2d-backward materialized `colsᵀ` (k×rows, all strided writes — 12% of the
kernel's flat profile) just to feed `dW = colsᵀ·dO` to the row-major band GEMM.
`gemmATF64Band` reads A as stored `[K,M]` (`A[p*m+i]`) with the same 4-row
register blocking and the same ascending-p per-element accumulation —
bit-identical to transpose-then-multiply, zero transpose traffic, and the
4-element `A[p*m+i:i+4]` reads are contiguous. Part of the 1557 → 1299 µs
(−17% total) conv2d-backward win.

## 6. Devirtualize hot unary kernels — but only where the call was the cost

`unOp(f)` calls its `func(float64) float64` per element (indirect call).
Direct-call kernels measured: Exp −10%, Tanh −5%, Log −3% (64K f64). A sigmoid
twin was built, measured **flat**, and reverted — sigmoid's cost is the
`math.Exp` + divide, not the dispatch. Devirtualization pays only when the
callee is cheap relative to an indirect call.

## 7. Two-lane F64 SiLU on Apple M2

The ARM64 performance build already used Neon (Arm's 128-bit Advanced SIMD
instruction set) for the float32 activation family, but float64 (F64) SiLU —
the gate in every SwiGLU feed-forward block — still called scalar `math.Exp`.
The existing AMD64 (x86-64) polynomial was ported as a two-lane Neon leaf: Cody-Waite
range reduction, a degree-13 fused-multiply-add (FMA) chain and exponent-bit
reconstruction. Two vectors are interleaved so their dependent FMA chains can
overlap. An operation-specific capability switch enables only SiLU on ARM64;
F64 softplus and soft-cap stay on their unchanged scalar paths.

On an Apple M2 Pro with Go 1.26.5, `CGO_ENABLED=0`, `GOMAXPROCS=12` and the
256×1408 F64 kernel, six alternating runs of prebuilt default and
`GOEXPERIMENT=simd` binaries measured median **901,435 → 395,869 ns/op =
2.28×**. Both paths allocate 9 times per operation; the approximately 2.884 MB
is the output tensor and is unchanged. The 262,145-value accuracy sweep measured
maximum relative error `3.048e-16`, and a separate dense sweep proved every
vector lane bit-identical to the scalar tail. Signed zero, infinities and
not-a-number (NaN) values are pinned explicitly. `go tool objdump` confirms
`FRINTN`, two-lane `VFMLA`,
`FCVTZS`, `VSHL`, `VBSL` and `FDIV`, with no scalar math call in the vector leaf.

The L1 leaf also clears the external incumbent: with input and output already
allocated and one thread, GoAI measured median **555,144 ns/op** versus
**1.06 ms** for PyTorch 2.12.1 `aten.silu.out` on the same F64 shape, a **1.91×
GoAI lead**. The whole allocating operation is not yet the winner: a stable
six-run GoAI median was **225,231 ns/op** at 12 threads, while PyTorch's best
local setting was **196.35 µs** at 8 threads (PyTorch leads 1.15×). That gap was
localized below this arithmetic leaf: deterministic output-buffer reuse and the
Go tensor allocation lifecycle were the next L0 target, rather than another
transcendental rewrite.

### Caller-owned output reuse closes the L0 gap

`backend.ExecuteInto` now accepts caller-owned output tensors while preserving
the exact backend-selection, optimized-CPU fallback and reference-fallback
semantics of `backend.Execute`. The selected backend opts in through the small
`IntoBackend` capability; unsupported operations fail explicitly instead of
silently allocating. The first capability covers CPU SiLU F32 and F64. Its
allocating and caller-owned paths share the same arithmetic body, so output bits
cannot drift between the APIs.

The boundary rejects recorder contexts, released storage, input/output or
output/output aliases, offsets, non-contiguous layout, and mismatched output
count, device, dtype, shape or storage length before arithmetic. This is a
deliberately narrow ownership contract, not a global tensor pool. It follows
the same broad preallocation model as PyTorch's documented
[`out=` contract](https://docs.pytorch.org/docs/stable/notes/out.html), while
keeping backend support explicit.

Allocation profiling exposed two smaller reusable costs after output storage was
removed: a captured per-dispatch closure and an escaping completion barrier in
the worker pool. SiLU dispatch now uses typed pool tasks whose pointers remain
garbage-collector roots, and completed barriers are reused only after all
workers have left the task. The race suite found and closed the narrow
`Wait`-returns-before-final-worker-exit window. Both generalizable patterns were
reported upstream to perfscan as
[#554](https://github.com/jxsl13/perfscan/issues/554) and
[#556](https://github.com/jxsl13/perfscan/issues/556).

On the same Apple M2 Pro, shape and F64 SIMD build, six alternating benchmark
pairs from one prebuilt test binary measured:

| API | Median ns/op | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| `backend.Execute` | 226,358 | approximately 2,884,370 | 7 |
| `backend.ExecuteInto` | 133,497 | 0 | 0 |

Caller-owned output is therefore **1.70× faster** than the allocating GoAI
operation. A fresh PyTorch 2.12.1 sweep measured its best
`torch.ops.aten.silu.out` result at **189,042 ns/op** with 8 threads, so the
GoAI operation is now **1.42× faster** while remaining allocation-free. The
next bottom-up step is profile-driven `IntoBackend` coverage for another
allocation-dominated primitive, not speculative breadth.

## 8. Misc measured wins

- `pool2dMax`: compare natively in `T` instead of converting every tap to f64
  (f32→f64 is exact and monotonic → same winner, same stored bits; NaN
  never wins under strict `>` in both forms). With the `isMax` branch hoisted
  out of the pixel loop and window rows subsliced: MaxPool −3%, AvgPool −8%
  (8×16×64×64, k2 s2 f32).
- `addBiasKernel`: each row is a same-length vector add → route through
  `internal/simd.AddF64/F32` (BCE'd tight loop here, archsimd AVX on
  amd64+GOEXPERIMENT=simd for free). −4–5% on 512×1024 f32, bit-identical.
- conv2d-backward dBias: one row-major pass with `f` accumulators instead of
  `f` stride-`f` column sweeps (ascending-r order per filter preserved →
  bit-identical). ~1% of the backward kernel — marginal but strictly less
  memory traffic on every shape.

## Not touched, deliberately

- `gemm.go`/`gemm_nosimd.go` band kernels: BLIS-style blocking was already
  built, measured and discarded on this host (§T74/§B28 note in gemm.go);
  the remaining headroom is wider FMA SIMD, host-blocked on arm64 (§T11b).
- `binOp`/`broadcastContig`: elementwise binaries are memory-bound through the
  simd primitives already; the Add/Mul benches are dominated by tensor
  allocation, not the loop.
- `parThreshold`/pool dispatch internals: retuning the crossover or chunk
  count reproduced current behavior at the measured break-even; the fused-pass
  work (§4) attacks the same overhead with better returns.
