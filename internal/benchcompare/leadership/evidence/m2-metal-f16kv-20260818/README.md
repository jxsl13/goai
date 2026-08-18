# M2 Metal opt-in f16 KV cache

This bundle is the promotion evidence for Spectackle proposal
`P-01M093C3MFF0E8E1THCWYA1757`. It evaluates the opt-in
`llamagpu.NewQuantF16KV` path at commit
`5ade743b3457370a7b3e40c70ae064696f568d0d`; `NewQuant` and every non-Metal
backend retain their established f32-cache behavior.

## Verdict

The path passes its promotion gates on the Apple M2 Pro:

- retained K/V storage is exactly half the f32-cache size;
- production scalar and paired converters match deterministic IEEE binary16
  round-to-nearest-even golden bits, including an odd tail;
- Metal timestamp labels prove `kv.f32_to_f16_pair` and
  `mha.f16kv.decode.splitk.*` execute in the quantized decoder;
- the trained TinyLlama run preserves 76/76 greedy tokens across repeated f16
  executions, keeps every checked logit finite, and measures logit NRMSE
  `0.000170697` with maximum absolute error `0.00239384`;
- all three post-commit interleaved end-to-end campaigns pass: ctx512 is
  1.0310-1.0358x, ctx1024 is 1.0503-1.0634x, and ctx1536 is 1.0624-1.0691x;
  the ctx8 gate is 1.0270-1.0318x and unchanged-f32 control spread never
  exceeds 1.056x;
- the f16-reading attention leaf is 1.192-1.290x faster at ctx512 and grows to
  1.525-1.751x at ctx2048 across five campaigns.

The required five-pair shipping comparison uses the same pinned
TinyLlama-1.1B Q4_K_M file and llama.cpp b10450 f16-KV/FlashAttention-auto
configuration as the preceding attribution bundle. GoAI's median is 178.4
tok/s at tg64 and 1,516.2 tok/s at pp64; llama.cpp's is 195.466625 and
1,772.392904 tok/s. The remaining incumbent factors are therefore 1.0957x at
decode and 1.1690x at prefill. Relative to the preceding GoAI f32-cache
shipping medians, decode moved from 172.2 to 178.4 tok/s (1.0360x), while
prefill is neutral at 0.9989x. This historical-baseline delta is corroborative;
the same-process trained-model A/B campaigns are the primary gain proof.

The result closes the f16-cache capability gap but does not claim overall
llama.cpp leadership. Decode remains 9.57% behind by incumbent/GoAI factor and
prefill remains 16.90% behind. The next shipping work must target persistent
graph/command execution and the remaining quantized projection/fusion cost,
not relabel this feature win as full-stack parity.

## End-to-end protocol

`trained-e2e.tsv` contains three independent invocations of the promotion
test. Each context records three f32-before/f16/f32-after rounds. The table
retains the median of six f32 controls, median of three f16 candidates, their
ratio, and the worst unchanged-f32 within-round spread. The command was:

```sh
GOAI_TINYLLAMA_GGUF=/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
  go test ./llamagpu \
  -run '^TestMetalF16KVRealModelQualityAndSpeed$' -count=3 -v
```

The test performs real Unigram tokenization of "Explain in one sentence why
the sky is blue.", generates 64 greedy tokens, repeats the f16 run for
determinism, compares one decode-step logit vector, profiles the ctx512 paths,
and then runs the interleaved throughput campaigns. Model parsing, allocation,
weight upload, and context prefill are outside each measured 32-step window.

At ctx512, representative production profiles attributed 394-405 us to f32
split-K attention plus 67-84 us to cache blits. The f16 profiles attributed
290-333 us to split-K attention plus 99-108 us to the paired conversion. All
profiles had zero omitted MPS events. Whole-command samples remain noisier than
the guarded repeated throughput campaign and are used for path attribution,
not as the headline speedup.

## Leaf protocol

`leaf-ranges.tsv` records the range across five interleaved campaigns. Every
campaign used sq=1, 32 query heads, four KV heads, dk=64, eight warmups, and 31
samples per arm. `gpu_speedup` compares timestamped attention encoders;
`host_speedup` includes command submission and completion.

```sh
go test ./backend/metal \
  -run '^TestF16KVAttentionInterleavedAB$' -count=5 -v

go test ./backend/metal -run '^$' \
  -bench '^BenchmarkF16KVAttention$' -benchtime=750ms -count=5
```

Leaf output is compared bit-for-bit against f32 attention supplied the same
pre-rounded K/V values. This distinguishes attention correctness from the
separately gated f32-to-f16 conversion semantics.

## Pinned incumbent campaign

`shipping.tsv` contains five fresh-process pairs. Each round launches GoAI
first and llama.cpp second, with one measured repetition per process. Both use
the identical model, batch one, tg64/pp64 boundaries, f16 K/V storage, and the
same Metal GPU. Model load, upload, warmup, and process startup are excluded.

```sh
TINYLLAMA_GGUF=/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
GOAI_PROD_KV=f16 GOAI_PROD_REPS=1 \
  go test -tags vulkan ./internal/benchcompare \
  -run '^TestProdDecodeGGUF$' -count=1 -v

/opt/homebrew/Cellar/llama.cpp/10450/bin/llama-bench \
  -m /Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
  -p 64 -n 64 -r 1 -ctk f16 -ctv f16 -fa auto -o json
```

The incumbent is pinned in `manifest.json`; its JSON reported build 10450,
upstream commit `ece963f41`, ggml 0.20.1, model payload 667,078,656 bytes, and
f16 K/V types. The model file hash is identical to the earlier attribution
campaign.

## Generalizable findings

The feature initially missed the ctx512 gate when K and V used separate
conversion dispatches. Pairing the streams in one alignment-aware `half2`
kernel reduced the conversion family from about 143.6 us to 95-108 us and
turned a 1.019-1.023x result into a repeatable result above 1.03x. This is
reported as [perfscan #764](https://github.com/jxsl13/perfscan/issues/764).
The measurement-floor interaction is also tracked in
[perfscan #763](https://github.com/jxsl13/perfscan/issues/763).
