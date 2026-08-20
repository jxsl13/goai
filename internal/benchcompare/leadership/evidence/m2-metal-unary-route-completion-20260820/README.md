# M2 Metal Synchronous Unary Route Completion (2026-08-20)

## Outcome

Retained as operation-, build-, architecture-, layout-, and size-bounded routes. On Apple M2 Pro,
all 81 default-build and all 153 `GOEXPERIMENT=simd` routed campaign medians clear the 1.10x gate.
The default build improves by 1.105x to 114.171x; the SIMD build improves by 1.111x to 117.094x.

Default `darwin/arm64` builds route valid contiguous offset-zero F32 Neg, Sqrt, and Abs tensors
through the optimized CPU backend through 4,194,304 elements, ReLU through 65,536, and scalar
Exp, Log, Tanh, and Sigmoid through 2,048. SIMD builds extend Exp, Log, Tanh, and Sigmoid through
4,194,304 elements. Direct Metal remains active for Intel Darwin, strided/offset layouts,
empty/invalid inputs, larger tensors, and every operation outside its measured zone.

Spectackle proposal: `P-01M0G1P2CTFSR`. Task: `T-01M0G1PRH9E13`. Decision:
`ADR-01M0G1PTQPF57`. Contracts: `MEASURED-METAL-UNARY-ROUTE-001`,
`MEASURED-METAL-SIMD-UNARY-ROUTE-001`, and `MEASURED-METAL-UNARY-FALLBACK-001`.
General finding: [perfscan #773](https://github.com/jxsl13/perfscan/issues/773#issuecomment-5359321002).

## Why one threshold is wrong

Historical T535 correctly rejected broad CPU routing while the alternatives were scalar closure
kernels. The CPU backend later gained devirtualized parallel Neg/ReLU/Sqrt/Abs implementations and
optional typed arm64 NEON Exp/Log/Tanh/Sigmoid implementations. The resulting crossovers differ by
operation and build: default scalar transcendentals lose beyond 2K, ReLU crosses near 64K in both
builds, while every other SIMD unary remains faster through 4M. A family-wide or timeless CPU/GPU
decision would therefore preserve large losses or introduce regressions.

## Frozen setup

- Base: `09f9c3362c5114dbd46c34fc66f726a0254bb8c8`.
- Machine: Apple M2 Pro, macOS 26.5.1, Go 1.26.6, darwin/arm64.
- Builds: default and `GOEXPERIMENT=simd`, measured independently.
- Control: isolated incumbent synchronous upload to Metal, unary dispatch, wait, and download.
- Candidate: production selector, including architecture, build, operation, layout, dtype, validity,
  and size gates.
- Shapes: 2,048; 65,536; 131,072; 349,440; 524,288; 2,097,152; and 4,194,304 elements.
- Gate: every routed campaign median at least 1.10x; operation-specific parity; both selector arms;
  exactly one autograd record; affected model workload at least 0.99x; complete validation.

Each operation ran in a separate process with 20 untimed warmups per arm. Each final campaign used
100 measured iterations and seven samples:

```text
go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalUnaryRouteCandidates/<operation>/' \
  -benchtime=100x -count=7

GOEXPERIMENT=simd go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalUnaryRouteCandidates/<operation>/' \
  -benchtime=100x -count=7
```

## Default-build production-selector results

Each cell is direct-Metal median / production-selector median. Only retained routed cells are shown.

| Operation | Elements | Campaign 1 | Campaign 2 | Campaign 3 | Worst |
|---|---:|---:|---:|---:|---:|
| Neg | 2,048 | 99.969x | 112.676x | 114.171x | 99.969x |
| Neg | 65,536 | 4.461x | 4.351x | 4.461x | 4.351x |
| Neg | 131,072 | 2.376x | 2.412x | 2.316x | 2.316x |
| Neg | 349,440 | 2.061x | 2.085x | 2.191x | 2.061x |
| Neg | 524,288 | 2.154x | 2.141x | 2.175x | 2.141x |
| Neg | 2,097,152 | 2.056x | 2.309x | 2.315x | 2.056x |
| Neg | 4,194,304 | 2.839x | 2.857x | 2.839x | 2.839x |
| Exp | 2,048 | 10.095x | 14.080x | 16.746x | 10.095x |
| Log | 2,048 | 11.011x | 16.737x | 13.947x | 11.011x |
| Tanh | 2,048 | 14.769x | 16.072x | 14.211x | 14.211x |
| ReLU | 2,048 | 97.516x | 70.628x | 95.480x | 70.628x |
| ReLU | 65,536 | 1.123x | 1.105x | 1.231x | 1.105x |
| Sigmoid | 2,048 | 12.031x | 14.003x | 11.727x | 11.727x |
| Sqrt | 2,048 | 93.886x | 110.292x | 95.654x | 93.886x |
| Sqrt | 65,536 | 3.752x | 3.916x | 3.832x | 3.752x |
| Sqrt | 131,072 | 2.152x | 2.270x | 2.347x | 2.152x |
| Sqrt | 349,440 | 1.888x | 1.956x | 1.865x | 1.865x |
| Sqrt | 524,288 | 1.920x | 1.845x | 1.853x | 1.845x |
| Sqrt | 2,097,152 | 1.979x | 1.975x | 1.925x | 1.925x |
| Sqrt | 4,194,304 | 2.476x | 2.492x | 2.451x | 2.451x |
| Abs | 2,048 | 98.058x | 91.310x | 88.293x | 88.293x |
| Abs | 65,536 | 5.324x | 4.129x | 4.270x | 4.129x |
| Abs | 131,072 | 3.455x | 2.123x | 2.276x | 2.123x |
| Abs | 349,440 | 2.824x | 2.277x | 3.637x | 2.277x |
| Abs | 524,288 | 2.277x | 2.409x | 2.497x | 2.277x |
| Abs | 2,097,152 | 2.487x | 2.278x | 2.590x | 2.278x |
| Abs | 4,194,304 | 2.853x | 2.840x | 2.857x | 2.840x |

The widest routed default candidate max/min spread was 2.628x. All 81 medians pass.

## SIMD production-selector results

| Operation | Elements | Campaign 1 | Campaign 2 | Campaign 3 | Worst |
|---|---:|---:|---:|---:|---:|
| Neg | 2,048 | 104.702x | 109.080x | 105.623x | 104.702x |
| Neg | 65,536 | 4.274x | 4.105x | 4.300x | 4.105x |
| Neg | 131,072 | 2.416x | 2.602x | 2.238x | 2.238x |
| Neg | 349,440 | 2.219x | 2.090x | 2.134x | 2.090x |
| Neg | 524,288 | 2.431x | 2.038x | 2.117x | 2.038x |
| Neg | 2,097,152 | 2.287x | 2.410x | 2.365x | 2.287x |
| Neg | 4,194,304 | 2.875x | 2.821x | 2.833x | 2.821x |
| Exp | 2,048 | 63.554x | 101.627x | 91.726x | 63.554x |
| Exp | 65,536 | 3.300x | 3.571x | 3.898x | 3.300x |
| Exp | 131,072 | 2.166x | 2.460x | 2.376x | 2.166x |
| Exp | 349,440 | 2.106x | 2.007x | 1.944x | 1.944x |
| Exp | 524,288 | 2.339x | 2.015x | 1.890x | 1.890x |
| Exp | 2,097,152 | 2.039x | 2.029x | 1.972x | 1.972x |
| Exp | 4,194,304 | 2.419x | 2.466x | 2.450x | 2.419x |
| Log | 2,048 | 62.066x | 80.086x | 86.382x | 62.066x |
| Log | 65,536 | 3.068x | 2.926x | 3.093x | 2.926x |
| Log | 131,072 | 2.096x | 2.270x | 2.351x | 2.096x |
| Log | 349,440 | 1.921x | 1.782x | 1.723x | 1.723x |
| Log | 524,288 | 1.811x | 1.721x | 1.837x | 1.721x |
| Log | 2,097,152 | 1.659x | 1.573x | 1.626x | 1.573x |
| Log | 4,194,304 | 1.988x | 1.975x | 1.975x | 1.975x |
| Tanh | 2,048 | 108.430x | 102.942x | 71.193x | 71.193x |
| Tanh | 65,536 | 3.785x | 3.675x | 3.612x | 3.612x |
| Tanh | 131,072 | 2.461x | 2.245x | 2.168x | 2.168x |
| Tanh | 349,440 | 2.101x | 1.925x | 2.263x | 1.925x |
| Tanh | 524,288 | 2.093x | 2.077x | 2.335x | 2.077x |
| Tanh | 2,097,152 | 1.746x | 1.935x | 2.119x | 1.746x |
| Tanh | 4,194,304 | 2.449x | 2.529x | 2.533x | 2.449x |
| ReLU | 2,048 | 76.032x | 99.183x | 102.869x | 76.032x |
| ReLU | 65,536 | 1.127x | 1.137x | 1.111x | 1.111x |
| Sigmoid | 2,048 | 102.481x | 97.246x | 117.094x | 97.246x |
| Sigmoid | 65,536 | 3.730x | 4.185x | 3.918x | 3.730x |
| Sigmoid | 131,072 | 2.221x | 2.470x | 2.163x | 2.163x |
| Sigmoid | 349,440 | 2.009x | 2.092x | 2.128x | 2.009x |
| Sigmoid | 524,288 | 2.287x | 2.172x | 2.112x | 2.112x |
| Sigmoid | 2,097,152 | 2.057x | 2.098x | 2.080x | 2.057x |
| Sigmoid | 4,194,304 | 2.618x | 2.666x | 2.692x | 2.618x |
| Sqrt | 2,048 | 87.609x | 102.654x | 87.720x | 87.609x |
| Sqrt | 65,536 | 3.262x | 3.804x | 3.784x | 3.262x |
| Sqrt | 131,072 | 2.096x | 2.358x | 2.136x | 2.096x |
| Sqrt | 349,440 | 1.873x | 1.845x | 1.846x | 1.845x |
| Sqrt | 524,288 | 1.951x | 1.830x | 1.904x | 1.830x |
| Sqrt | 2,097,152 | 2.093x | 2.030x | 2.161x | 2.030x |
| Sqrt | 4,194,304 | 2.487x | 2.481x | 2.458x | 2.458x |
| Abs | 2,048 | 89.249x | 104.776x | 105.822x | 89.249x |
| Abs | 65,536 | 4.373x | 4.779x | 4.268x | 4.268x |
| Abs | 131,072 | 2.495x | 2.255x | 2.516x | 2.255x |
| Abs | 349,440 | 2.259x | 1.983x | 1.929x | 1.929x |
| Abs | 524,288 | 2.150x | 2.185x | 1.837x | 1.837x |
| Abs | 2,097,152 | 2.166x | 2.183x | 1.237x | 1.237x |
| Abs | 4,194,304 | 2.853x | 2.841x | 3.056x | 2.841x |

All 153 medians pass. SIMD campaign 1 contained a ReLU candidate max/min spread of 3.026x;
campaign 3 contained an Abs spread of 3.530x. Immediate unchanged diagnostics retained every gain
and bounded routed ReLU spreads at 2.770x and Abs spreads at 1.268x. The original excursions remain
disclosed rather than discarded.

## End-to-end and correctness

`BenchmarkMetalRGLRUUnaryRouteWorkload` exercises Griffin's F32 real-gated recurrent unit, whose
forward dispatches Sigmoid, Neg, Exp, and Sqrt. With `-benchtime=1x -count=7`, the default-build
median improved from 1,897,416 ns on pinned `origin/main` to 1,008,917 ns, or 1.881x. The SIMD
median improved from 2,195,792 ns to 939,625 ns, or 2.337x.

Threshold tests pin the exact per-build operation map. Mutation-resistant selector tests compare
the production output bytewise with CPU at every measured ceiling, direct Metal at ceiling+1, and
direct Metal for strided views. The recorder test proves each public unary produces exactly one
autograd node. Existing CPU and Metal cross-reference suites cover operation-specific leaf parity;
the affected RGLRU output also matches the reference backend within relative/absolute 2e-3.

Repository preflight passed. The complete default and SIMD Metal functional suites passed in
64.861 and 63.143 seconds with only `TestMeasurementNoiseFloor` excluded. The exact short Metal CI
lanes then passed for both builds, including llamagpu. The unfiltered default suite reached every
functional test but its machine-state diagnostic measured 28.1% GPU timestamp CV and deliberately
failed; two isolated retries measured 26.8% and 25.9%. This is disclosed rather than relabeled as a
code failure or used for further A/B evidence. The frozen route campaigns completed before that
degradation and satisfy their per-cell spread gates, including the unchanged diagnostics above.

In-tree perfscan passed. A focused `backend/metal` scan reports no finding in the new route files;
its two findings are the pre-existing `Recorder.Profile` fixed scratch allocation tracked by
[perfscan #776](https://github.com/jxsl13/perfscan/issues/776). Spectackle reports no drift errors;
only repository-wide pre-existing W001/W002 advisories and its Go 1.25 versus repository Go 1.26
typed-call limitation remain.
