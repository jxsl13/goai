# M2 tensor pool release-token campaign (2026-08-25)

## Scope

- Machine: Apple M2 Pro, macOS 26.5.1, `darwin/arm64`, `GOMAXPROCS=12`.
- Toolchain: Go 1.27.0.
- Exact control: goai `1823fccf2ab22dd173df3ea19550d3179eb48099`.
- Candidate: pooled F32/F64 `Storage` carries a cold-allocated pointer token
  through `Release`; the same size-class `sync.Pool` continues to own GC
  reclamation and accepts both public slices and internal tokens.
- Public API, dtype, zeroing, capacity classes, and release behavior are
  unchanged. F16/BF16 retain the non-pooled shared-U16 path.

Executable SHA-256:

- exact-main tensor test: `f4a932b13d05f03c5efc3f5c00831f91c738f2a10fe21593c096c76a265b0ebd`
- shipping tensor test: `4d3589e920ca4d2c81a2763aa87b1e9b37d9cb73288f0f3486305d69ddce965c`
- external exact-main harness: `0a1d8e51f4050cc4ec33d2af18ce522d43892f3b6c769cb55c0c4039dd59b6db`
- external candidate harness: `1474056d320bd328700728bc31f27c9014f09a8ad0b9e2b97ba3b9cc13621b93`

The first custom-freelist design was rejected before shipment: even a bounded
freelist would retain large backing buffers across garbage collections, unlike
`sync.Pool`. Creating a new pointer wrapper during every release was also
rejected because it only moves the allocation.

## Method

Control and candidate test binaries were compiled separately. The public-API
F32/F64 matrix used the source in `external_harness_test.go.txt`, replacing the
goai module with the exact control checkout and then the leased candidate
worktree. Each campaign discarded pair 0, then alternated fresh processes as
control/candidate. Serial cells used 500 ms adaptive windows for nine retained
pairs; parallel cells used 500 ms windows for seven retained pairs. A dedicated
two-second, single-cell campaign adjudicated the initially near-neutral large
F64 serial result over nine retained pairs. The older in-package F32 benchmark
was independently run for one-second windows over nine retained pairs. Medians,
never minima, decide the result.

## Results

Serial public `NewOn` plus `Storage.Release`:

| dtype/elements | control ns/op | token ns/op | ratio | paired wins |
|---|---:|---:|---:|---:|
| F32 / 64 | 277.1 | 213.2 | **1.300x** | 7/9 |
| F32 / 65536 | 2985 | 2899 | **1.030x** | 4/9 |
| F64 / 64 | 335.9 | 285.3 | **1.177x** | 6/9 |
| F64 / 65536 | 8169 | 7893 | **1.035x** | 4/9 |

Parallel public `NewOn` plus `Storage.Release`:

| dtype/elements | control ns/op | token ns/op | ratio | paired wins |
|---|---:|---:|---:|---:|
| F32 / 64 | 183.8 | 177.8 | **1.034x** | 5/7 |
| F32 / 65536 | 1411 | 1375 | **1.026x** | 6/7 |
| F64 / 64 | 220.8 | 170.1 | **1.298x** | 5/7 |
| F64 / 65536 | 2402 | 2319 | **1.036x** | 3/7 |

The existing F32 benchmarks independently improve 366.8 to 331.0 ns/op
(**1.108x**, 9/9) at 64 elements and 3496 to 3246 ns/op (**1.077x**, 5/9)
at 65536 elements. Every warm production cell drops from 3 to 2 allocs/op;
small serial cells drop from 248 to 224 B/op. The isolated token cycles measure
0 B/op and 0 allocs/op for both dtypes.

The exported raw `Allocator` path deliberately remains slice-only. Its medians
are neutral: F32/4096 272.4 to 273.2 ns/op (0.997x, 6/9 wins) and F64/64 103.4
to 98.34 ns/op (1.052x, 4/9). It retains 24 B/op and 1 alloc/op because the
public `any` boundary cannot carry ownership metadata without an API break.

## Gates and backpropagation

- Full tensor test binary: pass.
- Race tensor test binary: pass.
- `CGO_ENABLED=0` tensor test binary: pass.
- Zeroing/reuse: F32 and F64, larger same-class reacquire, public-to-owned and
  owned-to-public crossings, idempotent release: pass.
- 24-worker mixed-dtype reuse gate and race detector: pass.
- Generalizable ownership-token guidance: perfscan issue
  [#908](https://github.com/jxsl13/perfscan/issues/908).

Raw retained timings are in `serial-pairs.tsv`, `parallel-pairs.tsv`,
`f64-large-adjudication.tsv`, `established-f32-pairs.tsv`, and
`raw-api-pairs.tsv`.
