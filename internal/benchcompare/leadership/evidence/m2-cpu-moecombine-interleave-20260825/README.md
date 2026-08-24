# M2 CPU MoECombine exact output interleaving

## Scope

Spectackle proposal `P-01M0TYHVNWF91` and task `T-01M0TYMJGMFMQ`
optimize the production CPU `OpMoECombine` reduction. The former loop completed
one output column at a time, so Go 1.27 emitted one loop-carried `FMADD`
accumulator chain. The candidate processes four adjacent output columns with
four independent accumulators while every column still visits experts in the
original ascending order.

The frozen baseline is commit
`288923de13d889bed1ec3396fbc92b5f97bd827c`; the four-column candidate is
`d0f10f24198c2574f1b379a5c3413e1c050577cc`. Their test executable SHA-256
values are:

- baseline: `2d9d1d07128b3d8d3879303902c7559048ed448130dc3f25d496ea9351c1aac5`;
- candidate: `4f5e008b3cc7ea7fa2baabedebf5da4b5f9761dba821eb5a2328e92bcac309d6`.

## Apple M2 Pro result

The final campaign used Go 1.27.0, macOS 26.5.1, darwin/arm64, and
`GOMAXPROCS=12`. One complete A/B warm-up pair was discarded. Nine recorded
fresh-process pairs then ran exactly 100 operations per benchmark in A/B order.
All weights were explicitly positive so every timed token executes the mixture
reduction rather than the `denom <= 0` zero-fill control.

| benchmark | baseline median | candidate median | result | paired wins |
|---|---:|---:|---:|---:|
| decode E8 F64, T=1 D=4096 | 27,095 ns | 15,895 ns | **1.7046x** | 6/9 |
| decode E8 F32, T=1 D=4096 | 24,028 ns | 12,965 ns | **1.8533x** | 7/9 |
| prefill E8 F64, T=128 D=2048 | 963,309 ns | 589,262 ns | **1.6348x** | 8/9 |
| prefill E8 F32, T=128 D=2048 | 955,999 ns | 664,469 ns | **1.4387x** | 9/9 |
| high-expert E64 F64, T=32 D=2048 | 3,117,331 ns | 1,932,820 ns | **1.6128x** | 9/9 |
| high-expert E64 F32, T=32 D=2048 | 3,029,996 ns | 1,488,315 ns | **2.0359x** | 9/9 |

The untouched zero-denominator control moved from 80,842 to 75,321 ns at the
medians (1.0733x, 6/9), well below every target delta. Allocation counts are
identical in every cell. Decode bytes/op are identical; parallel cells differ
by at most 55 median bytes/op from runtime worker accounting, with no new
production allocation site.

Raw samples are in `frozen-prepost.tsv`.

## Compiler and numerical gates

Apple `otool` disassembly of the Go 1.27 arm64 test binaries shows one
loop-carried `FMADD` in the baseline output loop and four independent `FMADD`
destinations in both candidate dtype loops. The scalar tail retains the former
single chain. There are no indirect calls in either reduction loop.

`TestMoECombineCPUByteIdenticalToRef` covers F64 and F32, expert counts
1/2/3/4/8, odd output widths, and multiple token counts. The exceptional gate
adds signed zero, zero and non-finite denominators, NaNs, infinities, and an
order-sensitive cancellation fixture. Reversing the production expert loop
temporarily makes the cancellation fixture fail for both dtypes. The restored
candidate passes the complete `backend/cpu` package.

The F32 path deliberately retains F64 widening and rounds once at the final
store, matching the reference contract exactly.

## Rejected rung

An eight-column variant remained bit-exact but did not dominate four columns.
Across three short alternating pairs it repeatedly regressed high-expert F64
and showed no consistent prefill advantage. Extra independent chains and lower
loop overhead therefore do not monotonically improve this M2 kernel; the
four-column width is a measured choice, not an architecture-independent
constant. Raw staged values are in `staged-rungs.tsv`.

The generalizable exact-preserving PS4008 refinement is tracked in
[perfscan issue #906](https://github.com/jxsl13/perfscan/issues/906). The
random-input branch-bypass benchmark defect is tracked separately in
[perfscan issue #907](https://github.com/jxsl13/perfscan/issues/907).

## Reproduction

```sh
GOPROXY=direct go test -c ./backend/cpu -o /private/tmp/goai-moe.test
GOMAXPROCS=12 /private/tmp/goai-moe.test \
  -test.run '^$' \
  -test.bench '^BenchmarkMoECombineInterleave' \
  -test.benchtime=100x -test.count=1
```
