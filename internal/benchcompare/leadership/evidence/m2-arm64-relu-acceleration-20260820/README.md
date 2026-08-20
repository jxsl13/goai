# M2 arm64 Exact F32 ReLU Acceleration (2026-08-20)

## Outcome

Retained. A semantics-exact arm64 NEON kernel improves the complete parallel CPU F32 ReLU
operation by **2.892x to 6.197x** across the measured 2,048 to 4,194,304-element domain on Apple
M2 Pro. The resulting synchronous Metal selector can route contiguous offset-zero F32 ReLU tensors
through the CPU backend through 4,194,304 elements: every default and `GOEXPERIMENT=simd` route
campaign median clears 1.10x, with observed medians spanning **2.092x to 217.8x** over direct Metal.

An alternating same-binary wide-MLP benchmark measures the production consequence without
cross-process ordering bias. Its three default-build campaign medians are 1.063x, 1.048x, and
1.065x; its three SIMD-build medians are 1.184x, 1.111x, and 1.134x.

Spectackle proposal: `P-01M0G5H18YENC`. Task: `T-01M0G5J741FHK`. Decision:
`ADR-01M0G6N8SDF2X`. Contracts: `ARM64-EXACT-RELU-001`,
`MEASURED-METAL-UNARY-ROUTE-001`, and `MEASURED-METAL-SIMD-UNARY-ROUTE-001`.
General finding: [perfscan #777](https://github.com/jxsl13/perfscan/issues/777).

## Frozen setup and semantics

- Base: `a401b509180e739ff462c7e32b655b672cade6e3`.
- CPU harness baseline: `43f72ef2`; NEON candidate: `9bc7653b`.
- Production-route workload harness: `9a995459`.
- Machine: Apple M2 Pro, macOS 26.5.1 build 25F80, Go 1.26.6, darwin/arm64.
- CPU benchmark boundary: allocation plus the production `parallel` dispatch and complete ReLU.
- Route control: direct synchronous Metal upload, dispatch, wait, and download.
- Route candidate: production selector and exact CPU kernel, including all dispatch overhead.
- CPU gate: bit-exact output and a gain at every retained crossover cell.
- Route gate: every campaign median at least 1.10x; operation parity; default and SIMD builds.
- Workload gate: every count-seven campaign median at least 1.03x in the alternating same binary.

The assembly processes 16 F32 values per iteration. `FCMGT` performs an ordered comparison against
positive zero and `BSL` retains the original bits only for positive values. Negative finite values,
NaNs, positive zero, and negative zero therefore become positive zero; positive finite values and
positive infinity retain their exact bits. This matches the prior Go comparison byte-for-byte.
`FMAX` was deliberately rejected because its NaN and signed-zero semantics do not implement this
contract.

Correctness covers lengths zero through 129, unaligned subslices, random raw bit patterns,
quiet/signaling NaNs, infinities, subnormals, and both zero signs. `go tool objdump` independently
confirmed that the checked-in words decode to `VFCMGT`.

## Complete CPU-operation results

Each invocation used 20 untimed warmups and seven 100-iteration samples. The table reports the
median incumbent and candidate ns/op from three independent campaigns; speedup is incumbent over
candidate. Allocations are unchanged.

| Elements | Campaign | Baseline ns/op | Candidate ns/op | Speedup |
|---:|---:|---:|---:|---:|
| 2,048 | 1 | 3,786 | 1,282 | 2.953x |
| 2,048 | 2 | 3,761 | 1,118 | 3.364x |
| 2,048 | 3 | 3,870 | 1,338 | 2.892x |
| 65,536 | 1 | 153,456 | 33,359 | 4.600x |
| 65,536 | 2 | 149,872 | 32,671 | 4.587x |
| 65,536 | 3 | 153,827 | 32,829 | 4.685x |
| 131,072 | 1 | 224,034 | 64,253 | 3.487x |
| 131,072 | 2 | 204,783 | 69,338 | 2.953x |
| 131,072 | 3 | 211,765 | 66,814 | 3.169x |
| 349,440 | 1 | 400,590 | 79,945 | 5.011x |
| 349,440 | 2 | 384,778 | 87,875 | 4.379x |
| 349,440 | 3 | 390,095 | 90,248 | 4.323x |
| 524,288 | 1 | 492,028 | 99,786 | 4.931x |
| 524,288 | 2 | 494,149 | 137,054 | 3.606x |
| 524,288 | 3 | 494,818 | 103,350 | 4.788x |
| 2,097,152 | 1 | 1,520,207 | 245,301 | 6.197x |
| 2,097,152 | 2 | 1,123,137 | 298,751 | 3.759x |
| 2,097,152 | 3 | 1,113,541 | 274,365 | 4.059x |
| 4,194,304 | 1 | 2,121,798 | 425,480 | 4.987x |
| 4,194,304 | 2 | 3,047,430 | 500,809 | 6.085x |
| 4,194,304 | 3 | 3,015,080 | 511,237 | 5.897x |

Command, run independently at the baseline and candidate revisions:

```text
go test ./backend/cpu -run '^$' -bench '^BenchmarkCPUReLUF32/' \
  -benchtime=100x -count=7
```

## Production Metal-selector results

Each ReLU-only process used 20 untimed warmups per arm and seven 100-iteration samples. Three
independent default campaigns had worst-cell medians of 2.919x, 2.647x, and 2.725x; their best
medians were approximately 189.9x, 217.8x, and 196.4x. Three independent SIMD campaigns had
worst-cell medians of 2.092x, 2.717x, and 2.803x; their best medians were approximately 203.5x,
198.9x, and 196.3x. All measured cells from 2,048 through 4,194,304 elements pass.

SIMD campaign 1 retained transient host interference at 349,440 and 2,097,152 elements. Their
medians remained 2.092x and 3.214x, so the excursion is disclosed rather than discarded.

```text
go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalUnaryRouteCandidates/ReLU/' -benchtime=100x -count=7

GOEXPERIMENT=simd go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalUnaryRouteCandidates/ReLU/' -benchtime=100x -count=7
```

## Alternating wide-MLP workload

The workload performs a 256-by-1 projection to width 1,365, applies ReLU to 349,440 activation
values, then projects to width one. The benchmark forces only the ReLU control arm to direct Metal;
all other operations use the same production backend. Candidate/control order alternates on every
iteration inside the same process. Each campaign used 20 warmups and seven samples of 100 paired
iterations.

```text
go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalReLUWideMLPRouteWorkload$' -benchtime=100x -count=7

GOEXPERIMENT=simd go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalReLUWideMLPRouteWorkload$' -benchtime=100x -count=7
```

Earlier non-alternating baseline/candidate process-order measurements were invalidated after host
drift was observed and are not used as evidence. Individual paired samples can still dip under
machine interference (lowest observed 0.9512x), but every independent count-seven campaign median
clears the 1.03x workload gate.

## Validation

The exact ReLU suite, complete CPU suite in default and SIMD builds, selector threshold/parity
tests, and wide-MLP parity tests pass. CGO-disabled cross-compilation passes for linux/amd64,
linux/arm64, and windows/amd64. The exact short CI lanes pass for both builds:

```text
go test -short ./backend/metal ./llamagpu -count=1
GOEXPERIMENT=simd go test -short ./backend/metal ./llamagpu -count=1
```

Repository preflight and in-tree perfscan pass. A focused changed-file scan reports no finding in
the new ReLU implementation or workload harness; its sole hit is the pre-existing F32 `Abs`
round-trip candidate in `backend/cpu/elementwise.go`. Spectackle reports no drift errors; only
repository-wide pre-existing W001/W002 advisories and its bundled Go 1.25 versus repository Go 1.26
typed-call limitation remain.

The unfiltered default Metal suite reached all functional tests and failed only
`TestMeasurementNoiseFloor`, which measured 28.58% GPU timestamp coefficient of variation. This is
an environmental machine-state sentinel, not a correctness failure; the exact default and SIMD
short CI lanes subsequently passed in full.
