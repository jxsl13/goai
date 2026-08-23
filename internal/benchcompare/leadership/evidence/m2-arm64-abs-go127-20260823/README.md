# M2 ARM64 exact F32 Abs, Go 1.27

This evidence record compares merged main `8c655287` with candidate source
`bd76b0a1` on an Apple M2 Pro (12 logical CPUs, 32 GiB) using Go 1.27.0.

Go 1.27 lowers `float32(math.Abs(float64(x)))` to sign-bit clearing. The old
NEON kernel also detected NaNs and set the quiet bit, so signaling-NaN payloads
did not match the scalar oracle. The Go 1.27 candidate removes four unsigned
compares, four quiet-bit masks, and four ORs per 16 values. Its objdump span
falls from 32 to 16 source instruction lines.

Go 1.26.6 still quiets the same signaling NaN. The final implementation uses
Go release build tags: Go 1.26 retains the original 32-line exact kernel, while
Go 1.27 and newer select the 16-line sign-clear kernel.

`TestAbsF32Arm64ExactAllLengths` now passes for lengths 0 through 257, including
unaligned slices, every F32 edge class, and random bit patterns. A new in-place
test covers 257 random bit patterns. Both tests pass under Go 1.26.6 and Go
1.27.0. The complete `backend/cpu` test binary and Linux AMD64 cross-compile
also pass.

## Preallocated leaf gate

Seven interleaved Go 1.27 count-one campaigns used 300 ms per cell. Speedup is
baseline ns/op divided by candidate ns/op. Exact two-sided p-values use a
Mann-Whitney permutation over the seven observations per side.

| Elements | Baseline median ns/op | Candidate median ns/op | Speedup | p | Allocs/op |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 161.1 | 106.1 | 1.518x | 0.0006 | 0 |
| 65,536 | 5,219 | 5,161 | 1.011x | 0.0973 | 0 |
| 349,440 | 28,151 | 27,637 | 1.019x | 0.0530 | 0 |
| 8,388,608 | 872,222 | 878,023 | 0.993x | 0.7104 | 0 |

The instruction-bound small leaf clears the 1.25x gate. Larger memory-bound
controls remain above 0.97x and show no significant regression.

## Complete-operation control

The production benchmark includes output allocation, dispatch, parallel fan-out,
and garbage collection. The same seven-pair schedule produced the following
medians; none of the differences is statistically significant.

| Elements | Baseline median ns/op | Candidate median ns/op | Speedup | p |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1,449 | 1,490 | 0.973x | 1.0000 |
| 65,536 | 31,246 | 26,636 | 1.173x | 0.5350 |
| 349,440 | 112,947 | 101,347 | 1.114x | 0.5350 |
| 8,388,608 | 868,947 | 903,804 | 0.961x | 0.1282 |

External `github.com/jxsl13/perfscan/perfscan@v1.81.0` ran with
`GOPROXY=direct` and reported no finding in the changed Go files. The reusable
redundant-NaN-quieting opportunity is tracked in
[perfscan #836](https://github.com/jxsl13/perfscan/issues/836).
