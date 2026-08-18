# M2 Metal prefill gate/up grouping — rejected

Date: 2026-08-18
Hardware: Apple M2 Pro
Baseline: `41b109de513de2605149e1839d9564ea54700b6e`
Prototype checkpoint: `2fb33c46`
Shape: Q4_K, M=64, K=2048, gate N=5632, up N=5632

## Hypothesis

TinyLlama prefill executes the SwiGLU gate and up projections as two cached-f16 MPS matrix
multiplications. The prototype retained the separate raw-quant weights for decode and combined
the gate and up rows only for 24 through 64 prefill rows. One cached-f16 MPS projection produced a
row-major `gate|up` result, followed by one Metal `SwiGLUHalves` kernel.

The frozen prerequisite for trained-model testing was a leaf median improvement of at least
1.08x. Trained pp64 required at least 1.03x in each of three independent invocations, while pp512
and tg64 each had to remain at least 0.99x.

## Correctness and structural proof

The new Metal halves kernel was bit-exact with the established `BinaryN` SwiGLU expression at 24
and 64 rows. A one-layer Q4_K decoder produced bit-exact logits between control and candidate at
both row counts. At the full TinyLlama gate/up shape, the grouped output was bit-exact with two
separate projections followed by SwiGLU.

Metal profiling recorded exactly two omitted MPS projections for the control and one for the
candidate, proving that the intended execution boundary changed.

## Leaf measurements

Command:

```text
go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalQ4KPrefillGateUpGrouping$' \
  -benchtime=300ms -count=10
```

Warm latency samples in ns/op:

| Arm | Samples |
|---|---|
| Separate | 915053, 907239, 871809, 869935, 868491, 874872, 868089, 877520, 896607, 908907 |
| Grouped | 852094, 858072, 859105, 858074, 853256, 858059, 863040, 881923, 861843, 856727 |

| Arm | Median | Allocations |
|---|---:|---:|
| Separate | 876.196 us | 32 B/op, 1 alloc/op |
| Grouped | 858.073 us | 32 B/op, 1 alloc/op |

The candidate improved the leaf by only **1.021x**, below the frozen **1.08x** prerequisite.
Control spread was 1.054x and candidate spread was 1.035x, so noise does not explain the failed
gate.

## Decision and learning

The trained-model campaign was intentionally skipped because its prerequisite leaf gate failed.
All executable prototype changes were reverted. Halving the MPS projection count did not halve
the arithmetic or weight traffic; it saved only a small amount of library and conversion overhead.
The candidate also duplicated about 12.4 MiB of packed gate/up weights while retaining the
separate decode weights, making the 2.1% leaf gain particularly poor leverage.

Structural reductions and latency gains must remain separate promotion claims. A profiler can
prove that a fusion removed calls without proving that the fusion is worth shipping.

Generalized follow-up: <https://github.com/jxsl13/perfscan/issues/770>
