# Go 1.27 ARM64 F32 Abs crossover evidence

This artifact reprices the production serial/parallel dispatch threshold after
the Go 1.27 ARM64 leaf became a four-register NEON sign-clear stream.

- Host: Apple M2 Pro, macOS 26.5.1, darwin/arm64.
- Toolchain: Go 1.27.0; `GOMAXPROCS=12`.
- Baseline: `absF32ParallelThreshold = 1 << 18`.
- Selected candidate: `1 << 21`; values below 2,097,152 run serially and the
  boundary itself remains parallel.
- Pairing: nine fresh-process pairs per focused cell, with process order
  reversed on even pairs. Warm-up/calibration iterations are excluded.
- Semantics: raw F32 bits match Go 1.27 `float32(math.Abs(float64(x)))` below
  and at the selected boundary; the in-place leaf and all lengths 0..257 are
  also pinned.

`pairs.tsv` retains the focused policy and complete-operation measurements.
The policy benchmark preallocates source and destination, while the production
benchmark includes tensor allocation and backend dispatch. The latter allocates
1-8 MiB per iteration and is correspondingly noisier; decisions use pair
direction as well as latency.

The focused policy cells place the crossover between 1,048,576 and 2,097,152
values: serial wins 7/9 pairs at 1,048,576 with a 1.108x median paired
advantage, while parallel wins 7/9 at 2,097,152. At the complete operation
boundary, the selected serial route wins 6/9 at 1,048,576; keeping the 2M cell
parallel beats the rejected `1<<22` candidate 8/9 with a 1.092x median paired
advantage. At 349,440, the old parallel route's 354,955 ns median becomes
344,511 ns serial with 8/9 pair wins, while allocations fall from six to four.

`threshold-sweep.tsv` is the exploratory one-sample sweep across every frozen
power-of-two candidate and the requested 131K, 262K, 349K, 524K, 2M, 4M, and
8M shapes. It is retained for audit, not used as a standalone speedup claim:
the ordered, allocation-heavy sweep visibly captures host/GC drift. The focused
fresh-process pairs are the decision authority.

Representative commands:

```sh
go test -c -o /tmp/abs-t21.test ./backend/cpu
GOMAXPROCS=12 /tmp/abs-t21.test -test.run '^$' \
  -test.bench '^BenchmarkAbsF32CPU/n1048576$' \
  -test.benchtime=500ms -test.count=1
GOMAXPROCS=12 /tmp/abs-t21.test -test.run '^$' \
  -test.bench '^BenchmarkAbsF32DispatchPolicy/n2097152/(serial|parallel)$' \
  -test.benchtime=500ms -test.count=1
```

Frozen executable SHA-256 fingerprints used by the campaigns:

```text
915add5a149abdd401041eb02c186aaf473c6fe024ec6c6b8ec54c4c7a3232bb  abs-t18 (focused 349K production)
1fd5aae8684ce41ff5123061eae8898c5ea146bdd152a48b236f0455e6bbd0a9  abs-t19 (focused 349K/524K production)
19397e66abe8b8c0dd2ee74748f8f537a2abe5c777f722a2ba8cb351dcd95e9d  abs-t20 (focused 1M production)
b3d9e5c4314efa27bee84ef97c0a87a4ad4f0513bf7f0c6f99ed6ce534eefc5c  abs-t21 (focused 1M production)
ed63f653e90506c4f20f69bece517ae49f6348b72bf1ebb46c24e7fad33a2e22  dispatch-policy binary
```

The generalizable stale-crossover detector requirement is tracked in
[perfscan issue #914](https://github.com/jxsl13/perfscan/issues/914).
