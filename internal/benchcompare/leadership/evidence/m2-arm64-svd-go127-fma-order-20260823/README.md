# M2 ARM64 SVD FMA-order compatibility

This evidence record compares merged main `7360079a` with source commit
`86add846` on an Apple M2 Pro using Go 1.26.6 and Go 1.27.0.

Go 1.27 changes which product is fused in the ARM64 lowering of
`sn*a + c*b`. Go 1.26 rounds `sn*a` and fuses `c*b`; Go 1.27 rounds `c*b`
and fuses `sn*a`. The deterministic fixture inputs are bit-identical between
the compilers, but four Jacobi SVD output digests differ. The 40x1 control,
whose V rotation never executes, remains identical.

| Shape | Input digest, both releases | Go 1.26 output | Unpinned Go 1.27 output |
| ---: | ---: | ---: | ---: |
| 8x8 | 4787771493518508411 | 3416335863526090039 | 1075440299122508612 |
| 32x12 | 16286964955943267212 | 16807871276312544421 | 15546730886483352861 |
| 12x32 | 6180745371401977785 | 12459786375205486409 | 6969607345239151379 |
| 64x64 | 7447501432688432545 | 5080245599646072399 | 14950887596508767193 |
| 40x1 | 2125998721918384801 | 18129668683049372422 | 18129668683049372422 |

The ARM64 helper now spells the incumbent contraction as
`math.FMA(c, b, sn*a)`. It inlines with zero helper calls and emits one
multiply plus one FMADD under both Go releases. The generic helper retains the
original expression; Rosetta AMD64 preserves its existing goldens under both
releases.

## Go 1.27 performance gate

Seven interleaved baseline/candidate campaigns used 300 ms per benchmark cell.
Speedup is baseline ns/op divided by candidate ns/op. Exact two-sided p-values
use the Mann-Whitney permutation over seven observations per side.

| Benchmark | Baseline median ns/op | Candidate median ns/op | Speedup | p |
| --- | ---: | ---: | ---: | ---: |
| SVDPCA | 35,582,903 | 35,648,255 | 0.9982x | 0.5350 |
| SVD 128x128 | 24,558,488 | 24,658,851 | 0.9959x | 0.1282 |
| SVD 192x192 | 86,205,594 | 86,445,979 | 0.9972x | 0.3176 |
| SVD 256x64 | 8,171,220 | 8,188,519 | 0.9979x | 0.9015 |

No result is statistically significant and all medians remain within 0.41%
of baseline.

## Go 1.26 compatibility control

| Benchmark | Baseline median ns/op | Candidate median ns/op | Speedup | p |
| --- | ---: | ---: | ---: | ---: |
| SVDPCA | 35,385,787 | 35,087,722 | 1.0085x | 0.3176 |
| SVD 128x128 | 24,441,720 | 24,423,455 | 1.0007x | 1.0000 |
| SVD 192x192 | 85,955,636 | 86,115,823 | 0.9981x | 0.9015 |
| SVD 256x64 | 8,127,354 | 8,092,632 | 1.0043x | 0.0379 |

Median allocation counts remain unchanged. The complete linalg test binary
passes on ARM64 and Rosetta AMD64 under both Go versions.

External `github.com/jxsl13/perfscan/perfscan@v1.81.0` ran with
`GOPROXY=direct` and reported no finding in the changed path. The reusable
compiler-dependent FMA-order hazard is tracked in
[perfscan #837](https://github.com/jxsl13/perfscan/issues/837).
