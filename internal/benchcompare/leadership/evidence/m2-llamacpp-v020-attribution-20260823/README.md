# M2 TinyLlama attribution against llama.cpp v0.2.0

This bundle refreshes the M2 Metal leadership cell against Homebrew
`llama.cpp` v0.2.0 at upstream revision
`bb4caa7540188872173c44d161602d9271386413` and ggml v0.21.0. It is a
scoped attribution result, not a universal performance claim.

## Verdict

Five fresh-process AB/BA pairs used the identical TinyLlama Q4_K_M file,
f16 K/V storage, batch one, decode positions 0..63, a generation warmup,
last-row-only pp64 output, and full Metal offload. The median throughputs are:

| Boundary | GoAI | llama.cpp | llama.cpp / GoAI |
| --- | ---: | ---: | ---: |
| tg64 | 127.7 tok/s | 134.429447 tok/s | 1.0527x |
| pp64 | 1556.1 tok/s | 1744.558477 tok/s | 1.1211x |

The machine was not quiet enough for a fine-grained headline: the median of
the five paired ratios is 1.1225x for tg64 and 1.1031x for pp64, and one pair
shows simultaneous system-wide slowdown. Every observation is retained in
`aggregate.tsv`; no outlier was removed. This cell establishes direction and
the next attribution target, not percent-level precision.

## Aggregate protocol

- Host: Apple M2 Pro, 32 GiB, macOS 26.5.1 (25F80), darwin/arm64.
- GoAI base: `d838664bab1570d22b3faae92d3582657b7da302`, Go 1.27.0.
- Model: 668,788,096 bytes, SHA-256
  `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
- Five rounds alternate GoAI-first and llama.cpp-first. Each arm is a fresh
  process and retains one measured repetition after its own warmup.
- GoAI uses `NewQuantF16KV`, warms position zero, times positions 0..63, then
  warms and times `StepNLast` at pp64.
- llama-bench uses `-p 64 -n 64 -r 1 -ctk f16 -ctv f16 -fa auto -o json`.
- Model load, resident upload, pipeline compilation, and process startup are
  outside both timed throughput boundaries.

The GoAI harness was compiled once, then launched independently per sample:

```sh
go test -tags vulkan -c -o /private/tmp/goai-prod-current.test ./internal/benchcompare
GOAI_PROD_KV=f16 GOAI_PROD_REPS=1 \
TINYLLAMA_GGUF=/absolute/path/to/tinyllama-1.1b-q4km.gguf \
/private/tmp/goai-prod-current.test \
  -test.run '^TestProdDecodeGGUF$' -test.count=1 -test.v
```

## Per-kernel attribution

GoAI's ordinary f16-KV recorder profile at context 512 covers 6.315 ms of a
7.752 ms command buffer. Q4_K is 4.001 ms and Q6_K is 1.250 ms: together they
consume 83.15% of attributed encoder time and 67.73% of the complete command
buffer. Fused f16-KV split-K attention is 0.353 ms. `goai-profile.txt` retains
the complete label aggregate.

The current llama.cpp trace selects a 5.498 ms compute command buffer. Its
shader profiler samples cover 1.963 ms, or 35.71% of that span. Q4_K and Q6_K
account for 99.985% of sampled shader duration. No foreign GPU interval
overlaps the selected buffer. The sampled distribution is directional rather
than an absolute cross-engine duration comparison because coverage is partial.
`llama-shader-summary.json` retains the command, counter, and shader summary.

The throughput run uses llama.cpp's shipping residency-set configuration. The
shader trace disables residency sets because Xcode 26.6 otherwise blocks the
instrumented process before model execution; shader rankings are retained, but
no residency-overhead claim is made. The model is exposed through a temporary
hard link outside `~/Desktop` because a stack sample proved that the
Xcode-launched process was blocked in `open()` at the protected Desktop path.
The linked bytes have the same SHA-256.

## Capture reliability repair

Xcode 26.6 can save a complete time-limited trace and return status 54. The
capture tool now defers that recorder error until five validations succeed:
workload marker, TOC export, required counter schemas, command-buffer
selection, and report analysis. A real status-54 campaign with a completed
JSONL row passed every validation and emitted the accepted-recorder warning.
A shorter campaign without the workload marker and a large truncated campaign
whose export was killed both remained hard failures.

The reusable findings are reported as
[perfscan #866](https://github.com/jxsl13/perfscan/issues/866) and
[perfscan #867](https://github.com/jxsl13/perfscan/issues/867).

## Decision

The current gap remains concentrated in quantized projection work. GoAI's
Q4_K plus Q6_K time nearly consumes llama.cpp's whole selected decode buffer,
while attention and elementwise work are secondary. Earlier speculative leaf
rewrites remain rejected; the next implementation must therefore begin with a
pinned source and dispatch-geometry audit of the exact v0.2.0 Q4_K/Q6_K path,
then clear an interleaved production gate before promotion.
