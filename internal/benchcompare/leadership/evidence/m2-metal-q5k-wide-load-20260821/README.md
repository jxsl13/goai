# M2 Metal Q5_K aligned-load gate (2026-08-21)

## Claim boundary

- Hardware: Apple M2 Pro, `darwin/arm64`.
- Leaf: resident Metal Q5_K matvec with `M=1`.
- Control: the prior SIMD-group-cooperative kernel in the same runtime-compiled
  Metal source and process.
- Candidate: one aligned `uint4` block-header load plus SIMD broadcasts and
  aligned per-lane `uint2` loads for qh/qs.
- Production route: candidate only when `K*N >= 6,291,456`; smaller shapes,
  `M>1`, unsupported SIMD widths, and a disabled guard retain the historical
  path.
- Both arms use the same resident weight, device buffers, recorder lifecycle,
  synchronization, transfer boundary, work geometry, accumulation precision,
  and allocation boundary.

This is a scoped M2 result, not a universal Q5_K leadership claim.

## Method

The promotion benchmark is `BenchmarkMetalQ5KWideLoad`. It reverses control and
candidate order on every iteration. After one unmeasured pipeline-transition
dispatch, each arm times 32 steady-state dispatches. This avoids both the severe
order bias of blocked sub-benchmarks and the artificial pipeline-switch cost of
alternating every dispatch.

Each formal campaign ran the complete Metal package suite and then seven
same-binary samples:

```text
env GOCACHE=/private/tmp/goai-q5k-wide-load-cache \
  go test ./backend/metal \
  -bench '^BenchmarkMetalQ5KWideLoad$' \
  -benchtime=20x -count=7
```

The promotion statistic is the median `control-ns/op / wide-ns/op` within each
independent count-7 campaign. Every candidate-eligible cell had to reach at
least 1.10x in every campaign.

## Formal results

| M=1 shape | Q5 weight | Campaign 1 | Campaign 2 | Campaign 3 | Minimum |
|---|---:|---:|---:|---:|---:|
| K=2048, N=3072 (mid up) | 4.125 MiB | 1.152x | 1.102x | 1.106x | **1.102x** |
| K=4096, N=2048 (mid down) | 5.500 MiB | 1.223x | 1.216x | 1.216x | **1.216x** |
| K=2048, N=5632 (gate/up) | 7.5625 MiB | 1.255x | 1.243x | 1.249x | **1.243x** |
| K=5632, N=2048 (down) | 7.5625 MiB | 1.404x | 1.394x | 1.398x | **1.394x** |

All 21 full Metal-suite repetitions passed. The route guard also covers the KV
(`K=2048,N=256`) and square (`K=N=2048`) controls; both use the historical
pipeline in both benchmark arms rather than claiming a sub-gate gain.

## Correctness and scope

- Candidate, cooperative control, and scalar Metal outputs agree within the
  existing `2e-5` relative contract across K=256 through K=4096, odd/tail N,
  and a candidate-eligible `K=2048,N=3072` cell.
- Candidate and control preserve NaN class.
- Input activations and quantized weights remain byte-identical.
- A shared native route predicate proves candidate selection at the threshold,
  historical fallback below it, zero candidate selection for `M>1`, and fallback
  when the cooperative guard is disabled.
- The matched pilot retained 32 B/op and one allocation/op in both arms; the
  changed mechanism is the selected Metal compute pipeline.

## Rejected variants and threshold calibration

- Four SIMD groups / 128 threads per threadgroup weakened the large-shape
  advantage and was rejected; the established 64-thread geometry remains.
- Loading qh/qs in half the lanes and sharing four words with
  `simd_shuffle_xor` was slower than coalesced per-lane `uint2` loads in the
  matched pilot (gate/up 1.220x versus 1.242x; down 1.371x versus 1.404x).
- A steady-state boundary pilot measured only 1.055x at `K=N=2048` but 1.117x
  at `K=2048,N=3072`. The latter then cleared all three formal campaigns, so
  `K*N=6,291,456` is the production cutoff.

## Interpretation

Q5_K's byte-granular issue cost was a real residual after cooperative occupancy
had already been fixed. Aligned packed loads help once the quantized weight is
large enough for device work to dominate the fixed recorder/command cost. The
winning M2 strategy is not “one loader plus more SIMD exchange”; it is a small
header broadcast combined with coalesced per-lane vector loads and an explicit
measured shape threshold.

## Perfscan backpropagation

The external CLI was fetched without a Go proxy and pinned for the scan:

```text
GOPROXY=direct go run github.com/jxsl13/perfscan/perfscan@v1.71.0 \
  -config internal/perfscan/perfscan.json -tests ./backend/metal/...
```

It found a new `fmt.Sprintf`-inside-loop issue in the benchmark harness; that
finding and the redundant pre-existing `fmt.Sprintf("%s", name)` in the touched
file were removed. The package scan still exits nonzero on unrelated existing
findings and warns that the legacy internal config contains keys not recognized
by external v1.71.0. The reusable aligned-load candidate pattern, its negative
evidence, and the need for an explicit benchmark gate are reported in
[perfscan issue #783](https://github.com/jxsl13/perfscan/issues/783).
