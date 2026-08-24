# M2 Metal resident ViT F32 AdamW session — 2026-08-24

Status: **promoted**. The candidate passes numerical, lifecycle, internal
performance, and pinned torch-mps leadership gates.

## Claim cell

- Hardware: Apple M2 Pro, macOS 26.5.1 (25F80).
- Model: pre-LayerNorm ViT, batch 8, RGB 32x32, patch 4, sequence 65,
  dim 128, 4 heads, FFN 512, 4 blocks, 10 classes, 807,306 parameters.
- Precision: F32 parameters, gradients, AdamW moments, and update arithmetic.
- Optimizer: AdamW, lr 1e-3, betas 0.9/0.999, epsilon 1e-8, weight decay 0.1.
- Timed boundary: patch packing, mean cross-entropy forward/backward, and one
  AdamW update.
- Candidate: one resident Metal session; only the scalar loss materializes per
  step. Parameters materialize on explicit `Sync` or `Close`.
- Internal control: the same one-graph Metal `LossAndGrad`, all Params-order
  gradients materialized, followed by portable F32 AdamW.
- External control: torch 2.12.1 MPS, Python 3.14.7, matching model and optimizer.

The GoAI arms start from independently constructed identical weights, warm up
once, and advance one update per timed iteration. Every sample contains both
arms; lead order alternates within each sample and across campaigns.

## Frozen gates

1. Three-step loss and synchronized-parameter parity within established F32
   tolerances, including an intermediate checkpoint sync.
2. A resident step leaves host parameters stale until `Sync`; `Close`
   synchronizes, is idempotent, and makes later `Step` and `Sync` fail.
3. Three order-alternated count-seven M2 campaigns: paired median speedup at
   least 1.20x and every aligned pair at least 1.10x.
4. Candidate median below the measured 9.137708 ms torch-mps AdamW median.

All gates pass.

## Environment and immutable pins

- Base commit: `43219289b2c6f71c0daf11be5d597f813dc666ab`.
- Candidate: the commit containing this evidence.
- Go: `go1.27.0 darwin/arm64`.
- Xcode: 26.6, build 17F113.
- Spectackle: 0.10.0.
- PyTorch: 2.12.1; Python: 3.14.7, installed from the local wheel cache.
- perfscan: `github.com/jxsl13/perfscan@v1.81.0`, fetched with
  `GOPROXY=direct` by the canonical repository script.

## Correctness and validation

Tests were compiled first and filtered only through the resulting binaries:

```text
go test -c -o /private/tmp/goai-vit-vision-20260824.test ./vision
/private/tmp/goai-vit-vision-20260824.test -test.run '^TestViTAdamWSession' -test.v

go test -c -o /private/tmp/goai-vit-metal-20260824.test ./backend/metal
/private/tmp/goai-vit-metal-20260824.test -test.run \
  '^(TestViTAdamWSessionMetalParitySyncAndLifetime|TestGPTAdamWSessionMetalParitySyncAndLifetime)$' -test.v
```

Race binaries compiled with `go test -race -c` and passed the same new vision,
Metal ViT, and existing Metal GPT session tests. The complete short suite and
the external scanner gates passed:

```text
go test -short ./...
make perfscan-check
make perfscan
make perfscan-compat
```

The hard perfscan integration gate reported zero findings. Whole-tree and
compatibility modes remained advisory, as configured. The Vulkan-tagged
cross-backend comparison binary also compiled with the new
`BenchmarkViTAdamWSession` cell.

## Internal paired campaigns

Commands used the same compiled binary:

```text
GOMAXPROCS=12 /private/tmp/goai-vit-adamw-final-20260824.test -test.run '^$' -test.bench '^BenchmarkViTAdamWSessionPaired$' -test.benchtime=20x -test.count=7 -test.cpu=12
GOMAXPROCS=12 GOAI_VIT_ADAMW_CANDIDATE_FIRST=1 /private/tmp/goai-vit-adamw-final-20260824.test -test.run '^$' -test.bench '^BenchmarkViTAdamWSessionPaired$' -test.benchtime=20x -test.count=7 -test.cpu=12
GOMAXPROCS=12 /private/tmp/goai-vit-adamw-final-20260824.test -test.run '^$' -test.bench '^BenchmarkViTAdamWSessionPaired$' -test.benchtime=20x -test.count=7 -test.cpu=12
```

Raw aligned results:

| Campaign | Pair | Candidate ms | Control ms | Speedup | Candidate img/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4.439163 | 5.622952 | 1.267x | 1,802 |
| 1 | 2 | 4.325842 | 5.515554 | 1.275x | 1,849 |
| 1 | 3 | 4.211119 | 5.480248 | 1.301x | 1,900 |
| 1 | 4 | 4.640338 | 5.793025 | 1.248x | 1,724 |
| 1 | 5 | 4.530123 | 5.646256 | 1.246x | 1,766 |
| 1 | 6 | 4.436829 | 5.615354 | 1.266x | 1,803 |
| 1 | 7 | 4.269075 | 5.416373 | 1.269x | 1,874 |
| 2 | 1 | 4.297869 | 5.483696 | 1.276x | 1,861 |
| 2 | 2 | 4.218283 | 5.481369 | 1.299x | 1,897 |
| 2 | 3 | 5.088402 | 6.317063 | 1.241x | 1,572 |
| 2 | 4 | 4.096408 | 5.359333 | 1.308x | 1,953 |
| 2 | 5 | 4.826790 | 5.869848 | 1.216x | 1,657 |
| 2 | 6 | 4.314681 | 5.563267 | 1.289x | 1,854 |
| 2 | 7 | 4.372662 | 5.652571 | 1.293x | 1,830 |
| 3 | 1 | 4.642473 | 5.863558 | 1.263x | 1,723 |
| 3 | 2 | 4.452804 | 5.611310 | 1.260x | 1,797 |
| 3 | 3 | 4.309964 | 5.563637 | 1.291x | 1,856 |
| 3 | 4 | 4.660661 | 5.746625 | 1.233x | 1,716 |
| 3 | 5 | 4.653667 | 5.764512 | 1.239x | 1,719 |
| 3 | 6 | 4.223923 | 5.507379 | 1.304x | 1,894 |
| 3 | 7 | 4.188427 | 5.391571 | 1.287x | 1,910 |

Aggregate over all 21 aligned pairs:

| Metric | Result | Gate |
|---|---:|---:|
| Candidate median | **4.372662 ms** | < 9.137708 ms |
| Candidate throughput at median | **1,829.55 img/s** | descriptive |
| Control median | 5.611310 ms | same semantics |
| Paired median speedup | **1.26875x** | >= 1.20x |
| Worst aligned speedup | **1.21610x** | >= 1.10x |

## Torch companion

The committed companion adds the exact full-step AdamW boundary:

```text
/private/tmp/goai-vit-torch-20260824/bin/python testdata/bench_vision_torch.py --adamw --json
```

Median of 12 after excluded warmups:

| torch 2.12.1 boundary | CPU ms | CPU img/s | MPS ms | MPS img/s |
|---|---:|---:|---:|---:|
| ViT objective only | — | — | 5.618812 | 1,423.79 |
| ViT objective + AdamW | 18.902312 | 423.23 | 9.137708 | 875.49 |

The GoAI resident median is **2.08974x** faster than the pinned torch-mps
AdamW median for this complete step.

## Attribution and architectural leverage

A pre-implementation attribution screen measured approximately 8.554 ms for
the materialized Metal objective and 1.461 ms for the immediate host AdamW
traversal at the same shape. The complete control median was approximately
9.619 ms in that noisier ceiling screen.

The promoted session removes parameter re-upload, dense host gradient
materialization, Go gradient allocation, and host optimizer traversal from the
repeated boundary. The complete ViT graph writes persistent gradient buffers,
then the shared native F32 AdamW compute encoder updates persistent parameters
and moments on the same `MPSCommandBuffer`. Go reuses one packed-patch tensor;
native code retains the input, target, loss, parameter, gradient, and moment
buffers. Only one scalar loss crosses the boundary per step.

The shared backend protocol remains source-compatible with the original GPT
names. GPT and ViT retain specialized public facades and objective graphs; a
generic graph/IR layer remains deferred until a third measured model warrants it.

The generalizable finding is reported as
[perfscan issue #880](https://github.com/jxsl13/perfscan/issues/880).

## Limitations

- The resident fast path is fixed-batch, contiguous, offset-zero F32 ViT
  geometry with one to eight uniform encoder blocks.
- Reading or externally mutating model parameters while the session is live is
  outside the ownership contract; use `Sync`, then finish with `Close`.
- The portable fallback preserves API and F32 optimizer semantics without
  promising accelerator residency.
- This leadership claim is limited to the declared M2 Pro/model/shape/F32/
  AdamW cell, not other hardware, dtypes, batch sizes, or optimizers.
