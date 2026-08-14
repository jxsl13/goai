# Rejected M2 K-quant threadgroup-width sweep — 2026-08-14

This bundle records the bounded screen from Spectackle task
`T-01M010BHVXFDT`. The current cooperative Q4_K/Q6_K kernel maps two output
rows per SIMD group and dispatches two SIMD groups (64 threads) per threadgroup.
The experiment derived row mapping from the actual dispatched size and exposed
1, 2, 4, and 8 SIMD groups (32/64/128/256 threads) in the same binary. The
64-thread path was the control; scalar, M>1, unsupported-device, and odd-N
fallbacks were preserved.

All four geometries matched the scalar Q4_K/Q6_K odd-N reference. The 200x
screen below covers every existing Q4_K/Q6_K benchmark shape. No alternate
geometry approached the standing 1.10x leaf threshold. The best observations
were Q4_K K2048,N16384 with one SIMD group (1.055x), Q4_K K2048,N32000 with
four groups (1.051x), and Q6_K K2048,N256 with four groups (1.036x). Several
other rows regressed. Every mode remained 8 B/op and 1 alloc/op.

Because this was a stop-gate screen with one sample per geometry and shape, the
small differences are not publishable wins. They are sufficient to reject this
dimension: even the most favorable observation is less than half the required
margin, and the previous T995 experiment demonstrated why sub-threshold screen
noise must not be promoted. No full-model or cold-start run was performed.

## Decision

The selector, dynamic row mapping, benchmark modes, and public experimental hook
are removed; production remains the exact two-SIMD-group kernel. Threadgroup
width alone is not the remaining M2 lever. A future kernel search must alter the
work decomposition itself—rows per SIMD group, lane/K partitioning, or fused
projection reuse—or move upward to a measured graph-fusion seam.

The general perfscan boundary remains the same as T995: execution-geometry
suggestions need device/shape evidence and should not be emitted merely because
a kernel uses a fixed threadgroup width.

## Reproduce

```sh
go test -c -o /tmp/goai-metal-kquant-geometry.test ./backend/metal

/tmp/goai-metal-kquant-geometry.test -test.run='^$' \
  -test.bench='BenchmarkMetalQ4KDecodeLeaf/.*/(simdgroups1|cooperative|simdgroups4|simdgroups8)$' \
  -test.benchtime=200x -test.count=1 -test.benchmem

/tmp/goai-metal-kquant-geometry.test -test.run='^$' \
  -test.bench='BenchmarkMetalQ6KDecodeLeaf/.*/(simdgroups1|cooperative|simdgroups4|simdgroups8)$' \
  -test.benchtime=200x -test.count=1 -test.benchmem
```

`screen.txt` is the raw benchmark output. `metadata.json` pins the measured
experimental binary and source hashes; the rejected code is not retained.
