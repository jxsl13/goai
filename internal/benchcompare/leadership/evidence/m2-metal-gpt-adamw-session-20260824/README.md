# M2 Metal resident GPT F32 AdamW session — 2026-08-24

Status: **promoted**. The candidate passes numerical, lifecycle, internal
performance, and corrected torch-mps leadership gates.

## Claim cell

- Hardware: Apple M2 Pro, macOS 26.5.1 (25F80).
- Model: causal pre-LN GPT, vocab 4096, context/sequence 256, dim 512,
  8 heads, FFN 2048, 6 blocks, batch 1.
- Precision: F32 parameters, gradients, AdamW moments, and update arithmetic.
- Optimizer: AdamW, lr 1e-3, betas 0.9/0.999, epsilon 1e-8, weight decay 0.1.
- Timed boundary: mean cross-entropy forward/backward plus one AdamW update.
- Candidate: one resident Metal session; only the scalar loss is materialized
  each step. Parameters materialize on explicit `Sync` or `Close`.
- Internal control: the same one-graph Metal `LossAndGrad`, all Params-order
  gradients materialized, followed by the portable F32 AdamW update.
- External control: torch 2.12.1 MPS, Python 3.14.7, identical model semantics
  and optimizer configuration.

The GoAI control and candidate start from independently constructed identical
weights, warm up once, and advance one update per timed iteration. Every sample
contains both arms; lead order alternates within a sample and across campaigns.

## Frozen gates

1. Three-step loss and synchronized-parameter parity within established F32
   tolerances, including an intermediate checkpoint sync.
2. A resident step must leave host parameters stale until `Sync`; `Close` must
   synchronize, be idempotent, and make later `Step` fail.
3. Three order-alternated count-seven M2 campaigns: paired median speedup at
   least 1.25x and every aligned pair at least 1.10x.
4. Candidate median at most 24.69 ms, which is at least 1.05x faster than the
   corrected pinned torch-mps AdamW median.

All gates pass.

## Environment and immutable pins

- Base commit: `ba513def8dacae2d67a08230839fe56f7b881f2b`.
- Candidate: the commit containing this evidence.
- Go: `go1.27.0 darwin/arm64`.
- Xcode: 26.6, build 17F113.
- Spectackle: 0.10.0.
- PyTorch: 2.12.1; Python: 3.14.7; resolved with
  `uv run --with torch==2.12.1 --python 3.14`.
- perfscan: `github.com/jxsl13/perfscan@v1.81.0`, fetched with
  `GOPROXY=direct` by the canonical repository script.

## Correctness and lifetime commands

Tests were compiled first and filtered only through the resulting binaries:

```text
go test -c ./nn -o /private/tmp/goai-gpt-adamw-nn-final-20260824.test
/private/tmp/goai-gpt-adamw-nn-final-20260824.test -test.run '^TestAdamWF32' -test.v

go test -c ./nlp -o /private/tmp/goai-gpt-adamw-nlp-final-20260824.test
(cd nlp && /private/tmp/goai-gpt-adamw-nlp-final-20260824.test -test.run '^TestGPTAdamWSession' -test.v)

go test -c ./backend/metal -o /private/tmp/goai-gpt-adamw-metal-final-20260824.test
(cd backend/metal && /private/tmp/goai-gpt-adamw-metal-final-20260824.test -test.run '^TestGPTAdamWSession' -test.v)
```

Passing tests:

```text
TestAdamWF32UsesF32StateAndArithmetic
TestGPTAdamWSessionPortableParityAndLifetime
TestGPTAdamWSessionRejectsInvalidConfigurationAndInputs
TestGPTAdamWSessionMetalParitySyncAndLifetime
```

The Metal test covers three updates, compares each pre-update loss, performs
checkpoint synchronization without re-uploading state, compares all 77
parameters, observes stale host parameters before the first sync, closes twice,
and rejects a post-close step.

## Internal paired campaigns

Commands:

```text
GOMAXPROCS=12 /private/tmp/goai-gpt-adamw-metal-final-20260824.test -test.run '^$' -test.bench '^BenchmarkGPTAdamWSessionPaired$' -test.benchtime=20x -test.count=7 -test.cpu=12
GOMAXPROCS=12 GOAI_GPT_ADAMW_CANDIDATE_FIRST=1 /private/tmp/goai-gpt-adamw-metal-final-20260824.test -test.run '^$' -test.bench '^BenchmarkGPTAdamWSessionPaired$' -test.benchtime=20x -test.count=7 -test.cpu=12
GOMAXPROCS=12 /private/tmp/goai-gpt-adamw-metal-final-20260824.test -test.run '^$' -test.bench '^BenchmarkGPTAdamWSessionPaired$' -test.benchtime=20x -test.count=7 -test.cpu=12
```

Raw aligned results:

| Campaign | Pair | Candidate ms | Control ms | Speedup | Candidate tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 16.077221 | 29.416015 | 1.830x | 15,923 |
| 1 | 2 | 15.576965 | 29.114123 | 1.869x | 16,435 |
| 1 | 3 | 15.532217 | 28.591008 | 1.841x | 16,482 |
| 1 | 4 | 15.530154 | 27.693921 | 1.783x | 16,484 |
| 1 | 5 | 15.444538 | 27.752646 | 1.797x | 16,575 |
| 1 | 6 | 15.443340 | 27.894933 | 1.806x | 16,577 |
| 1 | 7 | 15.473931 | 28.636796 | 1.851x | 16,544 |
| 2 | 1 | 15.796525 | 31.016852 | 1.964x | 16,206 |
| 2 | 2 | 16.904717 | 39.867900 | 2.358x | 15,144 |
| 2 | 3 | 16.187648 | 31.729398 | 1.960x | 15,815 |
| 2 | 4 | 16.584212 | 39.860308 | 2.404x | 15,436 |
| 2 | 5 | 15.841719 | 29.689169 | 1.874x | 16,160 |
| 2 | 6 | 15.382810 | 27.817900 | 1.808x | 16,642 |
| 2 | 7 | 15.498481 | 28.190119 | 1.819x | 16,518 |
| 3 | 1 | 15.435202 | 28.714529 | 1.860x | 16,585 |
| 3 | 2 | 15.429237 | 28.009167 | 1.815x | 16,592 |
| 3 | 3 | 15.466102 | 27.932808 | 1.806x | 16,552 |
| 3 | 4 | 15.618456 | 28.063000 | 1.797x | 16,391 |
| 3 | 5 | 15.408263 | 27.875417 | 1.809x | 16,614 |
| 3 | 6 | 15.529600 | 27.972742 | 1.801x | 16,485 |
| 3 | 7 | 15.474844 | 27.811771 | 1.797x | 16,543 |

Aggregate over all 21 aligned pairs:

| Metric | Result | Gate |
|---|---:|---:|
| Candidate median | **15.529600 ms** | <= 24.69 ms |
| Candidate throughput at median | **16,484.65 tok/s** | descriptive |
| Control median | 28.190119 ms | same semantics |
| Paired median speedup | **1.81890x** | >= 1.25x |
| Worst aligned speedup | **1.78324x** | >= 1.10x |

## Corrected torch companion

The previous companion enabled multi-head-attention bias and called the same
pre-attention LayerNorm separately for Q, K, and V. GoAI has no attention bias
and shares one normalized input. The committed companion now matches those
semantics and exposes `--adamw` for the matched full step.

Commands:

```text
uv run --offline --with torch==2.12.1 --python 3.14 python testdata/bench_gpt_train_torch.py --json
uv run --offline --with torch==2.12.1 --python 3.14 python testdata/bench_gpt_train_torch.py --adamw --json
```

Median of 12 after three excluded warmups:

| torch 2.12.1 boundary | CPU ms | CPU tok/s | MPS ms | MPS tok/s |
|---|---:|---:|---:|---:|
| Objective only | 46.853500 | 5,463.84 | 18.340271 | 13,958.35 |
| Objective + AdamW | 65.251021 | 3,923.31 | 25.887938 | 9,888.78 |

The GoAI resident median is **1.66701x** faster than the corrected torch-mps
AdamW median. The materialized-gradient `LossAndGrad` API remains a distinct
cell and is 1.131x behind torch's objective-only cell.

## Attribution and architectural leverage

A temporary native phase probe on the already-fused objective measured a
steady call at approximately 1.91 ms parameter upload, 0.18 ms result-wrapper
setup, 2.49 ms graph encoding, 9.36 ms GPU wait/compute, and 4.56 ms dense
gradient copy-out, or 18.54 ms native. Public Go result allocation raised the
objective median to approximately 20.75 ms. The host optimizer then added
approximately 10.5 ms.

The promoted session removes parameter re-upload, dense host gradient
materialization, Go gradient allocation, and host optimizer traversal from the
step boundary. The existing complete objective graph writes persistent gradient
buffers, then one custom compute encoder applies all Params-order AdamW updates
to persistent parameters and moments on the same `MPSCommandBuffer`.

A retained-activation-only prototype was rejected earlier: forced
materialization erased the intended leverage and measured only 1.0156x. The
winning unit is cross-step state residency plus optimizer placement, not merely
retaining intermediate graph values.

The generalizable finding is reported as
[perfscan issue #879](https://github.com/jxsl13/perfscan/issues/879).

## Limitations

- The resident fast path is intentionally fixed-sequence, contiguous,
  offset-zero F32 causal GPT geometry with one to eight uniform blocks.
- Reading or externally mutating model parameters while the session is live is
  outside the ownership contract; use `Sync`, then finish with `Close`.
- The portable fallback preserves the API and F32 optimizer semantics but does
  not promise accelerator residency.
- This is a leadership claim only for the declared M2 Pro/model/shape/F32/
  AdamW cell, not for other hardware, dtypes, batch sizes, or optimizers.
