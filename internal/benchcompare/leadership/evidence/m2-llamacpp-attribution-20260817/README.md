# M2 TinyLlama attribution against llama.cpp b10450

This bundle replaces the stale 2026-07-20 production headline with a
current-main, same-host comparison and records the remaining GPU-kernel
attribution. It is an attribution result, not a production speedup claim.

## Verdict

On the median of five alternating fresh-process pairs, llama.cpp is 1.0434x
faster at decode and 1.1193x faster at pp64 when both engines use the same
Q4_K_M file, f32 KV, batch one, context positions 0..63, a generation warmup,
and last-row-only prefill output. Against llama.cpp's shipping f16-KV,
FlashAttention-auto configuration, the incumbent is 1.1428x faster at decode
and 1.1678x faster at pp64.

The old GoAI 9.9 tok/s and 82 tok/s row was therefore obsolete, not a current
20x deficit. Current medians are 172.0/1517.9 tok/s in the strict campaign and
172.2/1517.9 tok/s in the shipping campaign. The harness now uses
`StepNLast`, matching llama-bench's last-logits-only prompt boundary; the API
already existed, so this correction is not claimed as a library performance
gain.

## Aggregate protocol

- Host: Apple M2 Pro, 32 GiB, macOS 26.5.1 (25F80), darwin/arm64.
- GoAI base: `35de401722eca2ed2e08fc8e4349dae4025bef57`.
- Model: `tinyllama-1.1b-q4km.gguf`, 668,788,096 file bytes, SHA-256
  `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
- Five rounds, each interweaving GoAI/llama.cpp strict then GoAI/llama.cpp
  shipping; each arm is a fresh process and one measured repetition.
- GoAI runs one untimed generation step, then times positions 0..63. Prefill
  warms and times `StepNLast` for 64 tokens. KV storage is f32.
- llama-bench performs its own warmup and reports tg64/pp64. Strict uses f32 KV
  and `-fa off`; shipping uses explicit f16 KV and `-fa auto`.
- Both engines fully offload to the same Metal GPU. Model load, weight upload,
  warmup, and process startup are outside the throughput measurements.

The exact compact observations are in `aggregate.tsv`. The campaigns compare
identical shapes and storage semantics, not identical token values: both
decoders use deterministic valid token IDs, and token values do not alter the
dense projection geometry under test.

## Pinned incumbent and f32-KV parser repair

The Homebrew executable is llama.cpp build 10450, upstream commit
`ece963f41b0b02d7a0d61436ae365762c073a4c8`, linked to ggml 0.20.1. Its help
advertises `-ctk/-ctv`, but this exact build's local llama-bench parser omitted
the `f32` spelling. `llama-b10450-f32kv.patch` is the complete source repair.
Only the rebuilt `libllama-bench-impl.dylib` was injected with
`DYLD_LIBRARY_PATH`; the original Homebrew executable and its llama, ggml,
Metal, BLAS, and CPU libraries remained in use. The JSON output was checked for
`type_k=f32`, `type_v=f32`, build 10450, and the pinned commit. Hashes for the
executable, patch, injected library, and model are in `manifest.json`.

Strict aggregate commands, repeated once per alternating round:

```sh
TINYLLAMA_GGUF=/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
GOAI_PROD_REPS=1 go test -tags vulkan ./internal/benchcompare \
  -run '^TestProdDecodeGGUF$' -count=1 -v

DYLD_LIBRARY_PATH=/private/tmp/llama-b10450-f32kv-lib \
/opt/homebrew/Cellar/llama.cpp/10450/bin/llama-bench \
  -m /Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
  -p 64 -n 64 -r 1 -ctk f32 -ctv f32 -fa off -o json
```

The shipping arm removes `DYLD_LIBRARY_PATH` and uses
`-ctk f16 -ctv f16 -fa auto`.

## Per-kernel attribution

GoAI's production timestamp profiler observes the actual recorder encoders.
The retained physical sample has 340 events over a 9.655 ms command buffer,
7.927 ms attributed duration (82.11% coverage), no omissions, and bit-identical
profiled/ordinary logits. Q4_K and Q6_K together account for 75.90% of
attributed encoder duration and 62.32% of total command time. `goai-profile.txt`
contains every ranked family.

llama.cpp does not expose GoAI's encoder sidecar, so the external analyzer
launches the pinned executable under Xcode Instruments and joins command-buffer,
GPU-interval, counter, and shader-profiler tables by exact process and GPU. The
capture emitted 13 command buffers: the final buffer is a 2.96 us blit, so the
analyzer explicitly skips it and selects the preceding compute buffer. The
selected shader span is 4.909 ms; 47 sampled shader intervals cover 2.790 ms
(56.84%). Q4_K plus Q6_K account for 99.982% of sampled shader duration.
`llama-shader-summary.json` preserves the sample distributions and the three
headline counters.

The strict exclusive-GPU replay correctly rejected this capture: six
WindowServer intervals overlap 417,667 ns, or 8.51% of the target shader span.
The retained non-exclusive replay records that contamination instead of
hiding it. Consequently, kernel rankings are valid attribution evidence, while
absolute cross-engine per-kernel durations are not claimed from this trace.
The f32-KV parser-injected process could be benchmarked directly but Instruments
rejected/stalled the injected launch, so this shader capture uses the unmodified
shipping f16-KV/FA-auto binary. K-quant weight kernels are shared; non-K timing
is not compared to the strict f32 campaign.

Capture and replay:

```sh
go run ./internal/benchcompare/metalcounters \
  -launch-json '["/opt/homebrew/Cellar/llama.cpp/10450/bin/llama-bench","-m","/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf","-p","1","-n","1","-r","1","-ctk","f16","-ctv","f16","-fa","auto","-o","json"]' \
  -launch-label llama.cpp-b10450-tg1 -launch-require '"n_gen": 1' \
  -iterations 1 -buffers-per-iteration 1 -launch-skip-final-buffers 1 \
  -keep-temp -output /tmp/llama-shader.json

go run ./internal/benchcompare/metalcounters \
  -analyze-dir /absolute/path/to/retained-export \
  -launch-json '["/opt/homebrew/Cellar/llama.cpp/10450/bin/llama-bench"]' \
  -launch-label llama.cpp-b10450-tg1 -launch-require llama \
  -iterations 1 -buffers-per-iteration 1 -launch-skip-final-buffers 1
```

## Decision

The earlier research claim that GoAI sustained only about 16 GB/s and trailed
llama.cpp by 7.1x (`R-01M01Q0AG0EHN`) is consumed as historical evidence. It
does not describe current main after the intervening cooperative, vectorized,
f16-cache, attention, and recorder work. K-quant matvec remains the largest
GoAI family, but the strict end-to-end decode gap is only 4.34%; this evidence
does not justify another speculative leaf rewrite. The larger shipping gap also
includes an explicit f16-KV/FlashAttention capability difference and should be
attacked as a feature/graph frontier, not mislabeled as a Q4_K kernel deficit.

Generalizable harness risk was reported as perfscan issue
https://github.com/jxsl13/perfscan/issues/756.
