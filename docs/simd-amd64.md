# AMD64 archsimd — the first real SIMD kernels (§T11b, ADR-0005)

> **In plain terms:** Go 1.26 ships an experimental `simd/archsimd` package
> that gives Go code direct access to the CPU's vector instructions (AVX on
> x86). It only exists when you build with `GOEXPERIMENT=simd`. This is the
> first time GoAI has had an amd64 host to build and verify it on, so this
> page records the first archsimd kernels that landed and what they measured.

## What landed

The `internal/simd` elementwise primitives (`Add/Sub/Mul/Div` × `F32/F64`)
gained an amd64 archsimd override (`internal/simd/simd_avx.go`, build tag
`amd64 && goexperiment.simd`). The scalar `simd.go` carries the complementary
`!(amd64 && goexperiment.simd)` tag, so exactly one definition compiles per
build and the **default pure-Go build is byte-for-byte unchanged** (§V7/§V23).

- f32 → `Float32x8` (256-bit, 8 lanes); f64 → `Float64x4` (4 lanes).
- Whole-vector body + scalar tail for lengths not divisible by the lane count.
- `archsimd.X86.AVX()` gates the intrinsics at runtime (§I4): a binary built
  with the experiment but run on a pre-AVX CPU takes the scalar path instead
  of faulting on an illegal instruction.

## Correctness

Each lane performs the identical IEEE-754 op as the scalar reference, so the
override is **bit-exact** (§V3/§V11, tol 0) — verified by
`TestElementwise{F32,F64}Parity` across sizes `{0,1,3,4,7,8,9,15,16,17,31,33,
64,1000}` (sub-lane, exact-multiple, and multiple+tail for both the 4-wide and
8-wide strides). The parity test is untagged, so it runs the archsimd kernels
under `GOEXPERIMENT=simd` and is a cheap scalar self-check otherwise.

## A/B measurement (§V22)

Same benchmark, two builds. Host: AMD Ryzen 7 5700G (Zen 3, AVX2+FMA), Go
1.26.5, `-benchtime 300ms -count 3`, medians.

```sh
# A (scalar):   CGO_ENABLED=0 go test -bench 'BenchmarkAdd...' ./internal/simd/
# B (archsimd): GOEXPERIMENT=simd go test -tags=simd -bench 'BenchmarkAdd...' ./internal/simd/
```

| benchmark | scalar | archsimd | speedup |
|-----------|--------|----------|---------|
| AddF32 4K (L1-resident) | 26.5 GB/s | 60.0 GB/s | **2.26×** |
| AddF32 256K (L2/SLC)    | 25.9 GB/s | 57.6 GB/s | **2.17×** |
| AddF64 4K               | 52.4 GB/s | 60.0 GB/s | 1.15× |
| AddF64 256K             | 50.8 GB/s | 58.9 GB/s | 1.16× |

## Reading the result

The scalar **f32** path ran at ~26 GB/s — *half* of the scalar f64 path's
~52 GB/s. Both moved the same number of *elements* per cycle, so the f32 path
was wasting f32's 2× density (this is the same "F32 ≈ F64" gap measured for
GEMM in `benchmarking-amd64.md`). The 8-wide archsimd kernel recovers it: f32
now reaches the *same* ~58–60 GB/s as f64 → the elementwise op is now
**memory-bandwidth-bound**, the physical ceiling. Hence:

- **f32 elementwise: a real ~2.2× win**, merged (pure-Go behind a build tag —
  the §C3 cgo gate does not apply).
- **f64 elementwise: ~1.15×** — the scalar loop was already near bandwidth
  (consistent with §B27's arm64 finding); the small gain is tighter
  load/store, not more lanes of useful work.

The important consequence is for **GEMM** (§T11b/§T74): elementwise is
bandwidth-bound and so caps out at ~2×, but GEMM does O(N³) compute over
O(N²) data — it is *compute*-bound, where the 8-wide f32 FMA (`MulAdd`) has
far more headroom than this ~2×. The elementwise landing proves the
archsimd-in-tree pattern (build tags, runtime feature gate, tail handling,
CI simd soft-gate) that the GEMM microkernel will reuse.
