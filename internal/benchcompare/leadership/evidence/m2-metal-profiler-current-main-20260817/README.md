# M2 Metal encoder profiler: current-main validation

This evidence validates an observability capability, not a model-speed claim. The
candidate adds opt-in per-encoder Metal timestamp sampling while the ordinary
`NewRecorder` path remains the production default.

## Environment and boundary

- Baseline: `2c3606b2b89e3ac2e4786f7c22f29250ced32afa` (`origin/main`).
- Candidate: the focused `codex/m2-metal-profiler` change.
- Host: Apple M2 Pro, macOS 26.5.1 (25F80), Go 1.26.6, darwin/arm64.
- Benchmark boundary: `NewRecorder`, 32 unary encoder constructions, and
  `Recorder.Free`; no command-buffer commit and no GPU execution.
- Protocol: ten fresh-process, order-alternating baseline/candidate pairs at a
  fixed `-benchtime=10000x`; setup buffers are outside the timer.

## Disabled-path gate

| Metric | Baseline | Candidate | Baseline / candidate |
| --- | ---: | ---: | ---: |
| median | 18,697.5 ns/op | 18,555 ns/op | 1.0077x |
| p90 | 19,401 ns/op | 18,938 ns/op | 1.0244x |
| allocations | 1 alloc/op, 8 B/op | 1 alloc/op, 8 B/op | unchanged |

The candidate has no measured disabled-path regression: its median is 0.76%
lower, inside the predeclared two-percent overhead bound. This is treated as
noise-compatible neutrality, not as a speedup claim. The paired geometric mean
is 1.0090x baseline/candidate and the candidate wins six of ten pairs.

Raw samples are in `control.txt` and `candidate.txt`; `benchstat.txt` is the
tool-produced comparison.

## Physical profile gate

`TestProdMetalEncoderProfile`, run against the same local
TinyLlama-1.1B Q4_K_M GGUF used by the production leadership harness, checks:

- exact profiled-versus-ordinary logits;
- no MPS, overflow, or unsupported-event omissions;
- calibrated positive GPU intervals, at least 70% summed encoder busy time, and
  a first-to-last event span within 10% of the command duration;
- at least one quantized-matmul event.

The captured physical-M2 output is in `profile.txt`. The profiler attributed 340
events in one position-1 decode command buffer. Unit coverage also exercises
compute, blit, MPS omission, overflow, explicit lifecycle errors, direct Q4_K
dequantization, the f16 quantized-prefill path, and matrix-unit FlashMM.

## Rejected promotions after current-main revalidation

Two source fusions from the superseded experimental branch were removed rather
than promoted:

| Candidate | Control | Candidate | Control / candidate | Exact result |
| --- | ---: | ---: | ---: | --- |
| residual add + RMSNorm, 200-token decode | 1,205,161,833 ns | 1,215,104,042 ns | 0.991818x | no; all-logit digest changed |
| Q4_K gate/up + SwiGLU, 200-token decode with residual fusion disabled | 1,190,455,583 ns | 1,208,686,833 ns | 0.984916x | yes |

The earlier apparent 2.84x parent result mixed fusion with an obsolete
physical-capacity binary dispatch. Current main already dispatches SwiGLU over
the logical active extent through `BinaryN`; against that corrected parent the
fusion is 1.53% slower. Neither fusion, its public API, nor its claim-bearing
documentation is part of this change.

## Reproduce

```sh
go test ./backend/metal -run '^TestRecorderProfile' -count=1 -v
TINYLLAMA_GGUF=/absolute/path/to/tinyllama-1.1b-q4km.gguf \
  go test ./internal/benchcompare -run '^TestProdMetalEncoderProfile$' -count=1 -v
go test -c -o /tmp/goai-profiler.test ./backend/metal
/tmp/goai-profiler.test -test.run '^$' \
  -test.bench '^BenchmarkMetalRecorderDisabledOverhead$' \
  -test.benchtime=10000x -test.count=1 -test.benchmem
benchstat control.txt candidate.txt
```
