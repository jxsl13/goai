# Benchmarking & the cgo gate (§V5, §C3)

Every optimization task (§T11, §T12, …) must show a benchmark delta against the
Pure-Go reference before it lands, and cgo is only considered after the Pure-Go
path is optimized to its ceiling and still loses by the §C3 threshold.

## Running

```sh
make bench                       # all benchmarks, CGO_ENABLED=0
go test ./backend/ref -bench .   # a single package
```

Inputs come from `internal/bench` with explicit seeds, so numbers are
reproducible across runs and machines.

## Baselines (Pure-Go reference, §T5–§T9)

Reference kernels favor clarity over speed (index math via `Unravel` allocates
per element) — the high alloc counts are exactly what §T11 removes. Recorded so
the optimized kernels have a concrete target:

| Benchmark | ref ns/op | cpu ns/op | speedup | note |
|-----------|-----------|-----------|---------|------|
| AddF64 4K | ~1.27e5 (4104 allocs) | ~1.2e4 (9 allocs) | ~10× | §T11, bit-identical |
| MatMulF64 128³ | ~9.1e6 (0.46 GFLOP/s) | ~2.97e5 (14.1 GFLOP/s) | ~31× | §T12/§T12b, bit-identical |
| MatMulF64 256³ | — | ~9.22e5 (36.4 GFLOP/s) | +31% vs ikj | §T12b 4-row blocking |
| MatMulF64 512³ | — | ~5.30e6 (50.6 GFLOP/s) | — | §T12b |

(Indicative darwin/arm64 numbers; treat the committed CI run as the source of
truth. Machine/arch is recorded alongside any comparison. §T12b's 4-row register
blocking preserves per-element k-order → still bit-identical to ref (V11 tol 0).
Next GFLOP/s gains: archsimd FMA microkernel on amd64, §T11b.)

## The first cgo gate: Metal/MPS (§T20, PASSED 2026-07-05)

All three §C2 conditions held before merge: (1) Pure-Go GEMM at its documented
ceiling (§T12/§T12b); (2) benchmark over the §C3 threshold (≥1.5×); (3)
`CGO_ENABLED=0` default build untouched (`-tags metal` isolation).

| f32 GEMM | cpu (Pure-Go ceiling) | metal (MPS) | speedup |
|----------|----------------------|-------------|---------|
| 512³  | 4.50ms · 59.7 GFLOP/s | 0.99ms · **272 GFLOP/s** | **4.6×** |
| 1024³ | 30.2ms · 71.1 GFLOP/s | 2.37ms · **906 GFLOP/s** | **12.7×** |

Cross-tolerance (§V11): rtol(K) = 1e-6·√K (MPS accumulates in f32 and reorders).
Run: `make metal-test` / `make metal-bench` (darwin + cgo only).

## Regression policy (§V5)

- An optimized kernel PR includes `benchstat old.txt new.txt`.
- CI fails a benchmark that regresses beyond noise on the reference baseline.
- The cgo gate (§C3): merge a cgo backend only when it beats the **optimized**
  Pure-Go kernel by ≥1.5× or reaches ≥80% of the C++ baseline the Pure-Go path
  cannot — measured on a real workload, not a micro-bench (§B10).
