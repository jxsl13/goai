# M2 Metal SIMD Activation Route Revalidation (2026-08-20)

## Outcome

Retained as a build- and size-bounded route. On Apple M2 Pro, the production
`darwin/arm64` `GOEXPERIMENT=simd` selector is 1.743x to 85.043x faster by campaign median than
the incumbent synchronous direct-Metal route across the frozen matrix. Valid contiguous,
offset-zero F32 GELU and SiLU forward and backward tensors containing at most 4,194,304 elements
route through the typed NEON CPU kernels. Direct Metal remains active everywhere else.

Spectackle proposal: `P-01M0FYJQFBF59`. Task: `T-01M0FYK7ZYFNT`. Decision:
`ADR-01M0FYKCJMFRE`. Contract: `MEASURED-METAL-SIMD-ACTIVATION-ROUTE-001`.
General findings: [route invalidation](https://github.com/jxsl13/perfscan/issues/773#issuecomment-5358702768)
and [warmup/interference diagnostics](https://github.com/jxsl13/perfscan/issues/774#issuecomment-5358703663).

## Why the old result changed

T535 correctly rejected CPU routing for Metal unary activations: the then-current CPU alternatives
were scalar closure kernels and reduced full Metal training throughput by 13 percent. T663 through
T666 later introduced typed arm64 NEON GELU and SiLU forward and backward kernels under
`GOEXPERIMENT=simd`. That implementation-class change invalidated the old comparison without
invalidating its original conclusion.

The default build deliberately remains on direct Metal. Its CPU forward kernels still use scalar
math, and its CPU backend does not register the F32 activation VJPs. Intel Darwin, non-contiguous
or offset views, invalid or empty inputs, and tensors above the measured ceiling also retain their
incumbent behavior.

## Frozen setup

- Base: `0ce9594f8eb2916eee7460bc5849b52ce7261519`.
- Machine: Apple M2 Pro, macOS 26.5.1, Go 1.26.6, darwin/arm64.
- Build: `GOEXPERIMENT=simd`.
- Control: isolated incumbent upload to Metal, one activation dispatch, wait, and download.
- Candidate: the production selector, including build, layout, dtype, validity, and size gates.
- Shapes: 2,048; 65,536; 131,072; 349,440; 524,288; 2,097,152; and 4,194,304 elements.
- Gate: every campaign median at least 1.10x faster; CPU-002 parity (relative 2e-3, absolute 1e-4);
  both selector arms; full GPT training throughput at least 0.99x; complete validation.

The short pilot exposed scheduler interference, so the frozen method increased from 10 to 20
untimed warmups per arm and isolated each operation in its own process. Each final campaign used
100 measured iterations and seven samples:

```text
GOEXPERIMENT=simd go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalSIMDActivationRoute/<operation>/' \
  -benchtime=100x -count=7
```

## Production-selector results

Each cell is direct-Metal median / production-selector median. The last column is the weakest of
the three independent campaign medians.

| Operation | Elements | Campaign 1 | Campaign 2 | Campaign 3 | Worst |
|---|---:|---:|---:|---:|---:|
| GELU | 2,048 | 60.566x | 65.764x | 65.578x | 60.566x |
| GELU | 65,536 | 3.373x | 3.131x | 3.338x | 3.131x |
| GELU | 131,072 | 2.315x | 2.062x | 2.352x | 2.062x |
| GELU | 349,440 | 2.136x | 1.940x | 1.922x | 1.922x |
| GELU | 524,288 | 2.263x | 1.997x | 1.946x | 1.946x |
| GELU | 2,097,152 | 1.754x | 1.777x | 1.743x | 1.743x |
| GELU | 4,194,304 | 2.128x | 2.155x | 2.125x | 2.125x |
| GELU backward | 2,048 | 69.031x | 68.987x | 58.410x | 58.410x |
| GELU backward | 65,536 | 3.047x | 3.017x | 2.944x | 2.944x |
| GELU backward | 131,072 | 2.423x | 2.268x | 1.870x | 1.870x |
| GELU backward | 349,440 | 2.081x | 2.164x | 2.190x | 2.081x |
| GELU backward | 524,288 | 2.390x | 2.288x | 2.008x | 2.008x |
| GELU backward | 2,097,152 | 2.121x | 2.210x | 2.165x | 2.121x |
| GELU backward | 4,194,304 | 2.262x | 2.295x | 2.289x | 2.262x |
| SiLU | 2,048 | 67.585x | 75.497x | 78.287x | 67.585x |
| SiLU | 65,536 | 3.516x | 3.815x | 3.987x | 3.516x |
| SiLU | 131,072 | 2.492x | 2.367x | 2.295x | 2.295x |
| SiLU | 349,440 | 2.089x | 2.129x | 2.096x | 2.089x |
| SiLU | 524,288 | 2.140x | 2.107x | 2.068x | 2.068x |
| SiLU | 2,097,152 | 2.109x | 1.928x | 1.879x | 1.879x |
| SiLU | 4,194,304 | 2.460x | 2.437x | 2.988x | 2.437x |
| SiLU backward | 2,048 | 85.043x | 84.863x | 81.438x | 81.438x |
| SiLU backward | 65,536 | 3.487x | 3.450x | 3.564x | 3.450x |
| SiLU backward | 131,072 | 2.773x | 2.753x | 2.821x | 2.753x |
| SiLU backward | 349,440 | 2.346x | 2.306x | 2.328x | 2.306x |
| SiLU backward | 524,288 | 2.448x | 2.448x | 2.450x | 2.448x |
| SiLU backward | 2,097,152 | 2.624x | 2.514x | 2.671x | 2.514x |
| SiLU backward | 4,194,304 | 2.565x | 2.569x | 2.506x | 2.506x |

All 84 medians clear the 1.10x gate. Campaigns 1 and 2 had worst candidate max/min spreads of
1.276x and 1.158x. Campaign 3 contained one non-recurring GELU-backward scheduler excursion with
a 7.339x spread; its median still cleared the gate. An immediate unchanged fourth diagnostic
campaign kept every GELU-backward spread at or below 1.193x and every median between 2.085x and
70.637x. The excursion remains disclosed rather than discarded or treated as a winner reversal.

## End-to-end and correctness

`BenchmarkGPTTrainingStep/metal` ran with `GOEXPERIMENT=simd`, `-benchtime=1x`, and `-count=7`
in an isolated `origin/main` worktree and on the candidate. The robust median improved from
74,552,583 ns (3,434 tokens/s) to 71,823,292 ns (3,564 tokens/s), a 1.038x speedup.

`TestMetalSIMDActivationMeasuredThresholdRoutesBothArms` pins every operation to the CPU arm
inside the measured zone and to direct Metal at `maxHostSIMDActivationElements+1`, byte for byte.
`TestMetalDefaultActivationRouteRemainsDirect` proves the default build remains byte-identical to
direct Metal. Existing forward and backward cross-reference tests cover the SIMD CPU-002 budget.

The complete default Metal suite passed in 44.523 seconds, and the complete SIMD Metal suite
passed in 44.602 seconds. Repository preflight passed. Focused in-tree perfscan found no candidate
in the new route files; its pre-existing `Recorder.Profile` scratch-buffer finding is tracked as
[perfscan #776](https://github.com/jxsl13/perfscan/issues/776). Spectackle reported no drift or
errors; only pre-existing repository-wide W001/W002 advisories and its Go 1.25 versus repository
Go 1.26 typed-call limitation remain.
