# M2 post-Q6_K decode profile — 2026-08-14

This bundle records the T994 bottleneck measurement immediately after the
cooperative Q4_K and Q6_K M=1 kernels. It is a selection artifact, not a
leadership claim. The byte-identical TinyLlama Q4_K_M model and the T993
production binary were used with both cooperative kernels enabled.

## Decision

The next M2 decode work remains in the Q4_K/Q6_K compute kernels:

- quantized model bytes are 77.06% Q4_K and 22.89% Q6_K; the model contains no
  Q2_K, Q3_K, or Q5_K tensors, so adding those specializations has no leverage
  for this workload;
- a representative warm decode command buffer spends about 57.4 ms on the GPU,
  while CPU encoder work is normally 0.24–0.65 ms; command encoding, submission,
  and the existing wait boundary therefore are not the primary bottleneck;
- a typical decode command buffer contains 45 encoders and its GPU timeline is
  23 compute intervals plus 22 short KV-cache blits. Most blits are about
  0.002 ms, with one observed at 0.174 ms, so eliminating blits cannot explain
  the remaining roughly 7.3x same-GGUF gap to pinned llama.cpp;
- Metal System Trace did not expose per-shader timing in this capture. Adjacent
  operation encoders appear as grouped compute intervals, so this evidence does
  not assign individual interval cost to attention, norm, or a projection.

The next controlled experiment is therefore an M2-specific Q4_K/Q6_K kernel
variant: explicitly unroll the fixed-trip decode loops and remove the row-tail
branch only for aligned `N % 4 == 0` shapes, while retaining the current
cooperative kernel as a same-binary control and preserving the odd-N fallback.
If that does not produce a material end-to-end gain, the next search dimension
is autotuning rows per SIMD group and SIMD groups per threadgroup, keyed by
device, quant type, K, and N. The fixed `2 rows × 2 SIMD groups` geometry is the
same as the pinned llama.cpp source, whose own comment marks these parameters as
device/work-size-specific tuning opportunities.

This follows `ARCHITECTURE-RESEARCH.md` invariants A4, A6, A7, A8, and A10:
capability precedes specialization; the current path remains the fallback;
there is no new hot-loop allocation, compilation, copy, or synchronization;
selection evidence spans kernel, operation, and full-model boundaries; and a
successful variant belongs in the device/shape/dtype kernel registry rather
than another public process-global toggle.

## Tensor incidence

`tensor-incidence.txt` records the raw tensor counts, elements, and encoded
bytes from the exact GGUF. The 21 Q6_K tensors are the output projection plus
selected `ffn_down` and `attn_v` matrices; the remaining quantized matrices are
Q4_K. F32 is limited to small tensors and represents 0.06% of encoded bytes.

## Metal trace boundary

The trace was captured with Xcode `xctrace` 16.0 (17F113), macOS 26.5.1
(25F80), and an Apple M2 Pro 19-core GPU. The production harness decoded 32
tokens after one prefill token with both cooperative kernels forced on. The
observed process produced 35 command buffers and 1,679 exported GPU interval
rows. Across command buffers, the median of summed GPU interval durations was
57.437583 ms. A typical warm command buffer had 45 GPU commands: 23 compute
intervals and 22 blits.

CPU encoder durations were read from the Application command-buffer table;
most warm decode buffers reported 0.24–0.65 ms, with an isolated 1.49 ms value.
The warm command-buffer lifetime was approximately 55–63 ms. This makes the
current decode GPU-kernel-bound at this short context. It does not prove which
individual shader is dominant because shader timeline/counter collection was
unavailable in this capture.

The 104 MiB `.trace` package and 34 MiB exported XML are intentionally not
checked into the repository. `metadata.json` records hashes and the exact
reproduction boundary. `metal-trace-summary.txt` is the compact retained
output.

## Reproduce

```sh
GOEXPERIMENT=simd go test -tags vulkan -c \
  -o /tmp/goai-prod-post-q6.test ./internal/benchcompare

GOAI_Q4K_COOPERATIVE=true GOAI_Q6K_COOPERATIVE=true \
  GOAI_PROD_DECODE_TOKENS=32 GOAI_PROD_PREFILL_TOKENS=1 GOAI_PROD_REPS=1 \
  TINYLLAMA_GGUF=$PWD/models/tinyllama-1.1b-q4km.gguf \
  xcrun xctrace record --template 'Metal System Trace' \
    --output /tmp/goai-post-q6.trace --time-limit 12s --no-prompt \
    --target-stdout - --launch -- \
    /tmp/goai-prod-post-q6.test \
      -test.run='^TestProdDecodeGGUF$' -test.count=1 -test.v

xcrun xctrace export --input /tmp/goai-post-q6.trace --toc
# Export the Application/Metal GPU interval tables selected from the TOC, then
# aggregate duration by command-buffer identifier. Keep process rows belonging
# to the launched test binary; do not mix Xcode/system processes into totals.
```

The trace is diagnostic evidence. Any successor kernel must still pass the
same-binary forced-control, correctness, cold/warm, interleaved, allocation,
and full-model gates before becoming the default.
