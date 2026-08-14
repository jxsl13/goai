# M2 Q6_K cooperative decode evidence — 2026-08-14

This bundle records the T993 M2-native Q6_K milestone selected by the preceding
Q4_K bottleneck study. The historical kernel assigned one GPU thread to an
output and accumulated the entire K dimension serially. The retained kernel
uses the pinned llama.cpp lane decomposition: two SIMD groups per 64-thread
threadgroup, two output rows per SIMD group, split-K lane accumulation, Q6_K
reconstruction and signed scale application in registers, then `simd_sum`.
It does not materialize a dequantized matrix.

## Architecture gate and selection boundary

This implementation follows `ARCHITECTURE-RESEARCH.md` invariants A4, A6, A7,
A10 and the M2 production sequence in section 6.4:

- the hot path preserves resident weights, input, and output and adds no new
  allocation, compilation, or host transfer;
- capability precedes specialization: the compiled pipeline must report SIMD
  width 32 and at least 64 threads per threadgroup;
- the cooperative path is limited to resident M=1 decode; M>1 and unsupported
  devices use the scalar fallback;
- `SetQ6KCooperative` provides a same-binary forced-off control;
- kernel-resident, operation, cold-start, and model-scale boundaries are kept
  separate rather than collapsed into one claim.

The direct Q4_K/Q6_K selector is a transitional leaf mechanism. A third format
will first be profiled for model incidence and end-to-end leverage; repeated
format-specific policy is a signal to move selection into the shape/dtype/device
kernel registry described by the architecture, not to accumulate conditionals.

## Forced-off warm result

`q6k-scalar.txt` and `q6k-cooperative.txt` contain ten paired samples per exact
decode shape. Mode order alternated by pair, the test binary was prebuilt,
weights and I/O buffers were resident, and every process performed 20 untimed
warmups. Each sample timed 500 executions. The boundary includes recorder
creation, parameter encoding, command submission, and command completion;
weight upload and pipeline compilation are excluded.

| M=1 shape | scalar median | cooperative median | improvement |
|---|---:|---:|---:|
| K2048,N256 small projection | 577.0 us | 214.4 us | 2.692x |
| K2048,N2048 attention output | 1459.7 us | 253.5 us | 5.759x |
| K5632,N2048 FFN down | 3546.3 us | 300.9 us | 11.788x |
| K2048,N32000 LM head | 3187.4 us | 727.7 us | 4.380x |

Every row has `p=0.000`, `n=10`; geomean time falls 81.20%. Both paths retain
8 B/op and 1 alloc/op, so the speedup is kernel execution rather than a changed
Go allocation boundary. `quality.txt` records the forced-off and f64
cross-references, odd-N tail, validation, capability fallback, and corrected
Q4_K non-vacuity check.

## Model-scale and cold-start results

The process-level isolated A/B keeps cooperative Q4_K enabled in both modes and
changes only Q6_K. On the byte-identical TinyLlama-1.1B Q4_K_M GGUF, median
32-token decode rises from 9.75 to 16.35 tok/s over ten alternating pairs, an
incremental 1.677x. Relative to the earlier both-scalar 7.0 tok/s baseline, the
two cooperative kernels yield a directional cumulative 2.336x improvement.

The current result remains far from external leadership: the pinned llama.cpp
Metal baseline is 119.9 tok/s median on the same GGUF, approximately 7.33x
faster. The pinned MLX baseline is 138.1 tok/s best-of-three, approximately
8.45x faster, but uses a different affine 4-bit conversion and is directional
only. Neither external row is interleaved with GoAI, and the bare workspace
cannot provide an immutable GoAI revision, so no publishable leadership claim
is made.

`cold-start.txt` keeps ten fresh-process pairs at K5632,N2048. First-call
medians, including Metal library and both Q6_K pipeline compilations, are
4.253 ms cooperative and 8.229 ms scalar (1.935x). Second-call medians are
1.136 ms and 4.718 ms (4.151x). These are local startup data for the exact
binary, not an external cold-start claim.

## Pinned executable textbooks

- llama.cpp commit `48d22e295e2b86b47366c16390794f3e05ba970a`
  (llama-bench build 10360), checkout
  `/tmp/goai-t988-llamacpp-20260814`
- MLX v0.32.0 commit `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`,
  checkout `/tmp/goai-t988-mlx-20260814`
- Go 1.26.6; macOS 26.5.1 (25F80); Apple M2 Pro, 19-core GPU, 32 GiB

The implementation reuses the pinned llama.cpp Q6_K bit reconstruction and
lane mapping as an executable textbook, while retaining GoAI's ABI, resident
recorder, fallback, validation, and measurement boundaries.

## Reproduce

```sh
go test -c -o /tmp/goai-metal-q6k.test ./backend/metal

# Run ten pairs, reversing modes every pair. Repeat for every shape above.
/tmp/goai-metal-q6k.test -test.run='^$' \
  -test.bench='^BenchmarkMetalQ6KDecodeLeaf/K5632N2048/cooperative$' \
  -test.benchtime=500x -test.count=1 -test.benchmem
/tmp/goai-metal-q6k.test -test.run='^$' \
  -test.bench='^BenchmarkMetalQ6KDecodeLeaf/K5632N2048/scalar$' \
  -test.benchtime=500x -test.count=1 -test.benchmem

$(go env GOPATH)/bin/benchstat \
  internal/benchcompare/leadership/evidence/m2-q6k-cooperative-20260814/q6k-scalar.txt \
  internal/benchcompare/leadership/evidence/m2-q6k-cooperative-20260814/q6k-cooperative.txt

GOEXPERIMENT=simd go test -tags vulkan -c -o /tmp/goai-prod-q6k.test ./internal/benchcompare
GOAI_Q4K_COOPERATIVE=true GOAI_Q6K_COOPERATIVE=true \
  GOAI_PROD_DECODE_TOKENS=32 GOAI_PROD_PREFILL_TOKENS=1 GOAI_PROD_REPS=1 \
  TINYLLAMA_GGUF=$PWD/models/tinyllama-1.1b-q4km.gguf \
  /tmp/goai-prod-q6k.test -test.run='^TestProdDecodeGGUF$' -test.count=1 -test.v

GOAI_Q6K_COLD_PROFILE=1 GOAI_Q6K_COOPERATIVE=true \
  /tmp/goai-metal-q6k.test -test.run='^TestMetalQ6KColdProfile$' -test.count=1 -test.v
```

The generalizable serial-K detector finding is tracked in
<https://github.com/jxsl13/perfscan/issues/565>; the Q6_K measurements are added
there as a second quantization-format validation of the signal.
