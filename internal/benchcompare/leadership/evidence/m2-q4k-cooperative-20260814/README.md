# M2 Q4_K cooperative decode evidence — 2026-08-14

This bundle records the T988 bottleneck selection and the first T989 M2-native
Metal implementation. The old GoAI Q4_K kernel assigned one GPU thread to each
output and accumulated all of K serially. The selected kernel splits K across
SIMD-group lanes, computes two output rows per SIMD group, and finishes with
`simd_sum`. Two SIMD groups share each 64-thread threadgroup.

## Forced-off result

`q4k-scalar.txt` and `q4k-cooperative.txt` contain ten paired samples per exact
decode shape. Mode order alternated by pair, the Go test binary was prebuilt,
weights and I/O buffers were device-resident, and every process performed 20
untimed warmups. Each sample then timed 500 executions. The measured boundary
includes recorder creation, parameter encoding, command submission, and command
completion; weight upload and pipeline compilation are excluded.

| M=1 shape | scalar median | cooperative median | improvement |
|---|---:|---:|---:|
| K2048,N2048 attention output | 546.2 us | 250.8 us | 2.177x |
| K2048,N2560 fused QKV width | 546.1 us | 256.0 us | 2.134x |
| K2048,N5632 FFN up/gate | 553.4 us | 288.4 us | 1.919x |
| K5632,N2048 FFN down | 1189.2 us | 284.2 us | 4.185x |
| K2048,N32000 LM head | 970.9 us | 574.6 us | 1.690x |

Every leaf row has `p=0.000`, `n=10`; the geomean time falls 56.34%. The final
production dispatch is deliberately limited to resident M=1 decode. M>1 keeps
the previous kernel until a separately measured matrix/prefill implementation
wins. Unsupported devices also retain the scalar fallback. `quality.txt`
records the scalar and f64 cross-references, including odd-N tail coverage.

After the primary grid was captured, the final source added a cached capability
guard requiring a 32-lane pipeline and at least 64 threads per threadgroup. The
kernel and M2-selected path did not change. Ten new alternating K2048,N5632
samples on that final source retain a 1.853x median win; a final full-model pair
is 10.1 versus 7.0 tok/s. `final-guard-confirmation.txt` keeps those samples.

The process-level model A/B uses the identical TinyLlama-1.1B Q4_K_M GGUF and
ten alternating mode pairs. Median decode rises from 7.0 to 9.9 tok/s (1.414x)
over 32 sequential tokens through all 22 layers. A single 13.9 tok/s sample is
retained, not selected. Prefill is intentionally unaffected by the M=1 gate.

## Warm/cold and Metal trace

`cold-start.txt` keeps ten fresh-process pairs. First-call medians are 5.174 ms
cooperative and 4.251 ms scalar, but variance overlaps heavily and both calls
compile the same Metal source plus both Q4_K pipeline objects. No cold leadership
claim is made. Second-call medians are 0.326 ms and 1.041 ms respectively.

Xcode Instruments 16.0 Metal System Trace captured 5,000 K2048,N5632 calls for
both paths. Under tracing overhead the benchmark reported 401.236 us/op for the
cooperative path and 633.798 us/op for scalar, preserving the directional win.
The `.trace` bundles are intentionally not checked in; the exact commands below
recreate them. Shader Timeline and GPU counter collection were disabled by the
stock template, so occupancy or stall claims are not inferred from this trace.

## Ranked T988 bottlenecks

| rank | seam | evidence | decision |
|---:|---|---|---|
| 1 | Q4_K M=1 fused dequant+matvec | serial-K source audit; 1.690–4.185x leaf and 1.414x end-to-end forced-off wins | implemented as the default M=1 cooperative path |
| 2 | Q6_K M=1 projections | same one-thread-per-output/serial-K structure; Q4_K_M uses Q6_K for selected higher-precision tensors | next native-Metal leaf to measure and parallelize |
| 3 | remaining K-quant matvecs | Q2_K/Q3_K/Q5_K retain the serial reduction pattern but occur less in this model | measure after Q6_K, share the cooperative dispatch design where valid |
| 4 | command and resource lifetime | decoder already records a full step in one command buffer; weights, KV, pipelines, and buffers are resident/cached | retain; not the first bottleneck |
| 5 | attention | cooperative decode attention already exists; short-context TinyLlama run is projection dominated | re-profile at long context before further fusion |
| 6 | prefill and ViT batching | M>1 and vision are outside this decode leaf | separate shape cells; never extrapolate this win |

The first naive cooperative prototype broadcast every scalar across lanes and
was rejected: it was flat at small shapes and about 2x slower at N=32000. The
retained implementation follows the pinned llama.cpp lane decomposition and
vector mask/scaling layout instead. This rejection is important: subgroup use
alone is not evidence of a faster kernel.

Pinned study sources:

- llama.cpp commit `48d22e295e2b86b47366c16390794f3e05ba970a`
  (llama-bench build 10360), local checkout
  `/tmp/goai-t988-llamacpp-20260814`
- MLX v0.32.0 commit `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`,
  local checkout `/tmp/goai-t988-mlx-20260814`
- Go 1.26.6; macOS 26.5.1 (25F80); Apple M2 Pro, 19-core GPU, 32 GiB

Current external directional baselines on the same host are llama.cpp 118.64
tok/s average (`pp1,tg32,r10`) on the byte-identical GGUF and MLX 0.32.0 138.1
tok/s best-of-3 (`pp56,tg64`) on its different affine 4-bit conversion. The
llama.cpp comparison shows that Q4_K was a real lever but not the final gap;
the MLX result is not a same-weight leadership claim.

## Reproduce

```sh
go test -c -o /tmp/goai-metal-q4k.test ./backend/metal

# Run ten pairs, reversing modes every pair. Repeat for each shape listed above.
/tmp/goai-metal-q4k.test -test.run='^$' \
  -test.bench='^BenchmarkMetalQ4KDecodeLeaf/K2048N5632/cooperative$' \
  -test.benchtime=500x -test.count=1 -test.benchmem
/tmp/goai-metal-q4k.test -test.run='^$' \
  -test.bench='^BenchmarkMetalQ4KDecodeLeaf/K2048N5632/scalar$' \
  -test.benchtime=500x -test.count=1 -test.benchmem

$(go env GOPATH)/bin/benchstat \
  internal/benchcompare/leadership/evidence/m2-q4k-cooperative-20260814/q4k-scalar.txt \
  internal/benchcompare/leadership/evidence/m2-q4k-cooperative-20260814/q4k-cooperative.txt

GOEXPERIMENT=simd go test -tags vulkan -c -o /tmp/goai-prod-q4k.test ./internal/benchcompare
GOAI_Q4K_COOPERATIVE=true GOAI_PROD_DECODE_TOKENS=32 \
  GOAI_PROD_PREFILL_TOKENS=1 GOAI_PROD_REPS=1 \
  TINYLLAMA_GGUF=$PWD/models/tinyllama-1.1b-q4km.gguf \
  /tmp/goai-prod-q4k.test -test.run='^TestProdDecodeGGUF$' -test.count=1 -test.v

xcrun xctrace record --template 'Metal System Trace' \
  --output /tmp/goai-q4k-cooperative.trace --time-limit 8s --no-prompt \
  --target-stdout - --launch -- /tmp/goai-metal-q4k.test \
  -test.run '^$' \
  -test.bench '^BenchmarkMetalQ4KDecodeLeaf/K2048N5632/cooperative$' \
  -test.benchtime=5000x -test.count=1 -test.benchmem
```

The generalizable serial-K detector finding is tracked in
<https://github.com/jxsl13/perfscan/issues/565>.
