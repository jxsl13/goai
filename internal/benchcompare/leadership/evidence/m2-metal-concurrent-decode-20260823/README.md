# M2 dependency-tracked concurrent Metal decode

This bundle promotes a decode-only `MTLDispatchTypeConcurrent` recorder for the dense, pre-norm,
quantized f16-KV Llama graph. Q/K/V and gate/up projections remain barrier-free inside one encoder;
every real producer-consumer edge receives a buffer-scope barrier. Blit, MPS, commit, finish, and
free boundaries physically end the shared encoder. Profiling, prefill, f32-KV, other architectures,
and other backends retain their established recorders.

The first correct prototype issued each graph barrier as its own CGo call. It improved median GPU
time only 1.0184x and regressed wall throughput to 0.9302x under host contention. The retained path
defers the barrier in Go and piggybacks one bit on the next existing native operation; Objective-C
executes the identical `memoryBarrierWithScope:MTLBarrierScopeBuffers` before that kernel. This
removes about 264 added Go-to-C crossings per token. The generalizable result is reported in
[perfscan #869](https://github.com/jxsl13/perfscan/issues/869).

## Promotion gate

The trained-model gate alternates arm order in one binary after a discarded warmup pair. Each arm
runs 64 single-token steps at positions 0 through 63. Command-buffer GPU duration isolates the
unrelated perfscan processes that were active on the host; wall duration remains an end-to-end
requirement. All fourteen retained arms produced final-logit digest `d706f8fb616cef3b`.

| Metric | Gate | Result |
|---|---:|---:|
| Median GPU speedup | >=1.03x | 1.0350x |
| Median wall speedup | >=1.02x | 1.0300x |
| GPU-ratio max/min spread | <=1.05x | 1.0053x |
| Raw-bit output parity | exact | exact, 7/7 pairs |

Three additional fresh-process pairs use the production tg64/pp64 harness with five measured
repetitions per process. Median speedups are 1.0369x for tg64 GPU time, 1.0322x for tg64 wall time,
and 1.0086x for pp64. Prefill does not use the new recorder; pp64 is a non-regression gate, not a
prefill acceleration claim.

## Correctness and lifecycle coverage

- a native test proves the toggle opens distinct ordinary/concurrent recorders;
- compute -> barrier -> compute -> blit -> compute produces exact values;
- freeing an uncommitted recorder with an active encoder is safe;
- 24 decoder steps match every logit bit-for-bit against the established recorder;
- the established RoPE/f16-KV fallback toggle remains exact with the concurrent recorder;
- profiling replaces the decode factory with the historical one-event-per-encoder recorder.

## Reproduction

```sh
go test -tags vulkan -c -o /tmp/goai-prod-concurrent.test ./internal/benchcompare
TINYLLAMA_GGUF=/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
GOAI_CONCURRENT_PAIRS=7 GOAI_CONCURRENT_GATE=true \
/tmp/goai-prod-concurrent.test -test.run '^TestProdMetalConcurrentDecodeCampaign$' -test.v -test.count=1

TINYLLAMA_GGUF=/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
GOAI_PROD_KV=f16 GOAI_PROD_REPS=5 GOAI_METAL_CONCURRENT=true \
/tmp/goai-prod-concurrent.test -test.run '^TestProdDecodeGGUF$' -test.count=1
```

The binary invocation uses the Go test binary's runtime selector; compilation never filters tests
with `go test -run`.
