# Apple ARM64 F32 MHA band-alignment evidence

## Claim boundary

This evidence establishes a measured GoAI winner matrix against GoAI's exact
pre-change implementation on one physical Apple M2 Pro. It does not claim
universal cross-library leadership. Both arms use identical public operation
boundaries, tensors, F32 semantics, output allocation, worker count, warm/cold
handling, and synchronization.

- Control: `305dd29b65cccf0521bded9f773546b3e587c166`
- Candidate implementation: `e8750ed96030f56a9fb3eec91d24f63278e0b67c`
- Toolchain: Go 1.26.6, `GOEXPERIMENT=simd`
- Hardware: Apple M2 Pro, darwin/arm64
- Runtime: `GOMAXPROCS=10`
- Statistics: three alternating campaigns, seven samples per arm and campaign,
  500 ms per sample, `benchstat`

The only production change is the ARM64-SIMD row-band constant: 30 becomes 32.
Every non-ARM64-SIMD build retains 30. Thirty aligns the amd64 six-row AVX2
kernel, but leaves a two-row scalar remainder in every full four-row Apple NEON
band. The selected 32-row value came from a physical-M2 pilot sweep over aligned
candidates 24, 28, 32, 36, and 40.

## Aggregate latency matrix

| Cell | Control median | Candidate median | Delta | p | n/arm |
|---|---:|---:|---:|---:|---:|
| `MHA512/fwd/cpu` | 1102.9 us | 697.6 us | -36.75% | <0.001 | 21 |
| GQA F32 seq128 | 437.2 us | 327.9 us | -24.99% | <0.001 | 21 |
| full MHA F32 seq128 | 474.8 us | 361.3 us | -23.90% | <0.001 | 21 |
| GQA F32 seq512 | 3.987 ms | 2.357 ms | -40.88% | <0.001 | 21 |
| full MHA F32 seq512 | 4.304 ms | 2.657 ms | -38.26% | <0.001 | 21 |
| single-token GQA decode, KV512 | 158.3 us | 158.7 us | neutral | 0.881 | 21 |

The six-cell geomean improves by 28.63%. Every latency gain repeats with
`p=0.001` in each independent seven-sample campaign. Decode is the explicit
small-shape no-regression control because its one-row work cannot consume a
full scheduler band.

## Numerical and allocation gates

The candidate's deterministic causal/full/GQA output digest is exactly
`73550b82110bb18f`. A deliberately corrupted expected digest was observed to
fail before restoration, establishing non-vacuity. The full default and ARM64
SIMD backend/cpu test binaries pass.

Ordinary benchmark allocation counts vary by one when Go's GC evicts
`sync.Pool` scratch; aggregate GQA seq512 therefore displays a misleading
11-to-12 median even though the change adds no allocation site. The dedicated
steady-state control disables GC only after constructing exact binaries and
shows both arms at 9 allocs/op. After the first control warm-up sample, both
arms report 4,194,988–4,194,997 B/op. The raw result is in
`alloc-steady-control.txt` and `alloc-steady-candidate.txt`.

## Profile attribution

Matched five-second CPU profiles on `BenchmarkMHA512/fwd/cpu` expose the causal
mechanism. The control samples `gemmF32RowsScalar` for 6.02 s and
`gemmF32Tile4x16Neon` for 6.13 s. In the candidate, the scalar remainder is
absent from the reported profile while the NEON tile remains present at 1.29 s.
The matched benchmark falls from 1,127,773 to 702,028 ns/op in these profile
runs. See `profile-control.txt` and `profile-candidate.txt`.

## Reproduction

Compile a backend/cpu test binary at each exact commit with
`GOEXPERIMENT=simd`, then run the benchmark binary directly (never through a
filtered `go test` invocation):

```text
env GOMAXPROCS=10 <binary> -test.run=^$ -test.bench=<cell-regex> \
  -test.benchtime=500ms -test.count=7 -test.benchmem
```

The campaign order was control/candidate, candidate/control, then
control/candidate. `campaign{1,2,3}-{control,candidate}.txt` are the unmodified
Go benchmark streams, and the corresponding `benchstat-*.txt` files preserve
the analyses. `benchstat-aggregate.txt` contains all 21 samples per arm.

The steady-state allocation audit uses:

```text
env GOMAXPROCS=10 GOGC=off <binary> -test.run=^$ \
  -test.bench=^BenchmarkAttention/MHAFwdGQA/f32/seq512$ \
  -test.benchtime=500ms -test.count=7 -test.benchmem
```

Generalizable mixed-ISA scheduler detection is tracked in
`jxsl13/perfscan#797`.
