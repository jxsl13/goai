# Apple arm64 F64 WKV NEON evidence

Measured on 2026-08-21 for Spectackle task `T-01M0J17HEEEF7` under proposal `P-01M0J0BWPDE4P`.

## Pinned configuration

- Hardware: physical Apple M2 Pro, 12 logical CPUs
- OS/architecture: `darwin/arm64`
- Toolchain: `go1.26.6`
- Build mode: `GOEXPERIMENT=simd`
- Exact control: `24360555396d1b694cbd5bcfec979c0416332497`
- Exact candidate: `c88bc795d76409c3ea002afef96f302b61fc69cc`
- Benchmark settings: `-test.run='^$' -test.benchtime=500ms -test.count=7 -test.benchmem=true`
- Order: campaign 1 control→candidate, campaign 2 candidate→control, campaign 3 control→candidate

The control and candidate were compiled into separate test binaries before measurement. Raw outputs and aggregate inputs are committed beside this file.

## Change

The Apple-arm64 `goexperiment.simd` WKV path now executes adjacent F64 channels in a fused NEON leaf. It keeps `w/u` and `AA/BB/PP` in D2 registers across the sequence, evaluates the selected stabilized exponential with the existing degree-13 strategy, masks finite arguments below -708 to exact `+0.0`, and stores state only after the pair completes. A no-allocation preflight routes hostile pairs to the scalar implementation before any mutation; odd tails remain scalar.

## Results

| Scope | Campaign | Control median | Candidate median | Delta | p |
|---|---:|---:|---:|---:|---:|
| internal SIMD WKV 512×1024 | 1 | 15.344 ms | 4.717 ms | -69.26% | 0.001 |
| internal SIMD WKV 512×1024 | 2 | 15.489 ms | 5.019 ms | -67.59% | 0.001 |
| internal SIMD WKV 512×1024 | 3 | 15.701 ms | 4.693 ms | -70.11% | 0.001 |
| internal SIMD WKV 512×1024 | aggregate n=21 | 15.489 ms | 4.826 ms | -68.84% | <0.001 |
| CPU OpWKV 512×1024 | 1 | 2.474 ms | 1.035 ms | -58.18% | 0.001 |
| CPU OpWKV 512×1024 | 2 | 2.521 ms | 1.028 ms | -59.22% | 0.001 |
| CPU OpWKV 512×1024 | 3 | 2.491 ms | 1.189 ms | -52.26% | 0.001 |
| CPU OpWKV 512×1024 | aggregate n=21 | 2.496 ms | 1.040 ms | -58.35% | <0.001 |

Every candidate internal sample reported `0 B/op` and `0 allocs/op`. The public backend retained its existing output/worker cost at 4.002 MiB and 29 allocs/op.

## Correctness and portability

- Maximum observed relative error versus scalar: `4.501e-12`, below the `1e-10` rule.
- Fresh `PP=-1e38` sentinel and deep-underflow exact-zero behavior pass.
- NaN, infinity, and `PP-w` overflow cases select scalar before mutation.
- Whole/stateful chunks and whole/2-aligned ranges are bit-identical, including final state.
- Default and SIMD package suites pass for `internal/simd` and `backend/cpu`.
- Native SIMD race suites pass, including the parallel CPU backend.
- Default and SIMD `go vet` pass for affected packages.
- `make preflight-full` passes, including pure-Go, benchmark-smoke, SIMD, cgo, and Metal lanes.
- SIMD product cross-builds pass for Linux arm64, Linux amd64, Windows amd64, and Darwin amd64.
- amd64 SIMD WKV accuracy/state/range tests execute successfully under Rosetta.
- Objdump confirms D2 `FMUL/FMLA/FDIV`, `FRINTN`, cutoff `FCMGE`, `VAND` exact-zero masking, and exponent `VSHL $52`.

