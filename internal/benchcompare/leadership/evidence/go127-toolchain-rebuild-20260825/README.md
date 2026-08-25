# Go 1.27 compatibility rebuild evidence

This artifact validates the repository-wide Go 1.27 compatibility change. The
benchmark binaries use identical GoAI source from merge commit
`8e81197759faa8b8660ffab9f989d1edd2a8e6ea`; only `GOTOOLCHAIN` differs.

- Host: Apple M2 Pro, macOS 26.5.1, darwin/arm64.
- Toolchains: Go 1.26.6 and Go 1.27.0.
- Process order: nine alternating pairs, 1.26→1.27 on odd pairs and reversed on
  even pairs.
- Timing: 500 ms per serial cell; 750 ms per parallel autograd cell. The SVD
  cells used 500 ms inside the same alternating frozen-binary campaign.
- Warmups: one untimed pass per binary, excluded.

`pairs.tsv` is the raw retained evidence. Medians and pair directions are
reported in `docs/benchmarking.md`. Absolute timings varied with concurrent
host load, so the weak parallel Eigh direction is not promoted to a compiler
speedup claim. The parallel MoE regression remains explicit.

Representative command shape:

```sh
GOTOOLCHAIN=go1.26.6 go test -c ./autograd -o /tmp/autograd-go126.test
GOTOOLCHAIN=go1.27.0 go test -c ./autograd -o /tmp/autograd-go127.test
GOMAXPROCS=12 /tmp/autograd-go126.test -test.run '^$' \
  -test.bench '^(BenchmarkEighVJP_128|BenchmarkMoECombineBackward)$' \
  -test.benchtime=750ms -test.count=1 -test.benchmem
```

The exact rebuilt head is accepted only after full pure-Go, cgo/race, Metal,
Vulkan, CUDA compile, tidy, vet, and external perfscan gates pass with Go 1.27.
The reusable toolchain-attribution finding is perfscan issue #912.

## Local race attribution

The complete Darwin/ARM64 race sweep has two pre-existing invalid cells. The
MLA bit oracle emits the same four non-golden digests from frozen Go 1.26.6 and
Go 1.27.0 race binaries; ordinary binaries pass. The classic wall-time guard is
also incompatible with race overhead: isolated GradientBoosting takes 2.942 s
under Go 1.26.6 and 1.573 s under Go 1.27.0 against a non-race 800 ms ceiling.
Every other package in the local race sweep passed. The required Linux cgo/race
authority is therefore the PR CI lane, which does not share the Darwin MLA
floating-point boundary.

## Go 1.27 AMD64 SIMD migration

The first Go 1.27 PR run exposed a source-compatibility break in the
experimental `simd/archsimd` API on Ubuntu/AMD64. The tree-wide port maps the
removed `Load*Slice` and `StoreSlice` forms to their Go 1.27 slice APIs, maps
the vector `RoundToEven` spelling to `Round`, and adapts the existing
pointer-based unaligned loads through fixed-length `unsafe.Slice` views. It
does not change arithmetic order, reduction topology, dispatch, or fallbacks.

Validation on the exact replacement source includes:

- a complete Linux/AMD64 `CGO_ENABLED=0 GOEXPERIMENT=simd` build;
- Linux/AMD64 SIMD test-binary compiles for `internal/simd`, `backend/cpu`,
  and `format/gguf`;
- x86_64 execution under Rosetta on the Apple M2 Pro, where the vector F64
  exponential underflow and accuracy gates pass (maximum relative error
  `3.409e-16`, maximum ULP error `2.00`); and
- the normal Apple ARM64 preflight, which continues to exercise the portable
  and native M2 paths.

This API-only repair has no independent speedup claim. Its performance
evidence is the same-source Go 1.26.6 versus Go 1.27.0 campaign above; an
AMD64 runtime comparison would conflate the compiler upgrade with the API
rename and is therefore deliberately not promoted as kernel leverage. PR CI
is the native Linux/AMD64 execution authority.
