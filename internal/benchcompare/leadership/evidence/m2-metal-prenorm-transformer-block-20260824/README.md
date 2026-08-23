# M2 Metal complete pre-norm transformer-block evidence

Date: 2026-08-24

## Candidate

- Base HEAD: `5445b5cbdc5a02357482476e2f0543783e5869d2`
- Candidate implementation commit: `ace96a510fe3c795c86e984d36d4d646420a2b55`
- Frozen benchmark binary SHA-256: `5a76bb1634cfc38098986190ea6cf8f4b9cd17d4628480b7ace38fa09311d20d`
- Package: `github.com/jxsl13/goai/backend/metal`
- Shape: batch 8, sequence 65, dimension 128, heads 4, hidden 512, F32
- Control disables only `OpPreNormTransformerBlock` and its backward operation. The independently promoted pre-norm attention and FFN graph fusions remain enabled in both arms.

## Environment

- MacBook Pro Mac14,10, Apple M2 Pro, 12 CPU cores, 32 GB unified memory
- macOS 26.5.1 (25F80)
- Go 1.27.0 darwin/arm64
- `GOMAXPROCS=1` for the benchmark processes, matching the single synchronous submission stream and removing unrelated Go scheduler migration noise

## Protocol

A single final candidate binary was compiled before measurement. Each boundary and full-model measurement ran in a fresh process with fixed 10-iteration samples and count 7. Campaigns 1 and 3 ran control before candidate; campaign 2 reversed the order with `GOAI_PRENORM_BLOCK_CANDIDATE_FIRST=1`. Every process performed both built-in warm-ups before timed work. Medians compare identical work and each aligned pair uses the same list position within one order-controlled campaign.

## Results

| Campaign | Boundary control median | Boundary candidate median | Speedup | Weakest boundary pair | Full control median | Full candidate median | Speedup | Weakest full pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.430 ms | 2.965 ms | 1.157x | 1.150x | 17.455 ms | 15.213 ms | 1.147x | 1.111x |
| 2 | 3.455 ms | 2.994 ms | 1.154x | 1.137x | 17.145 ms | 15.273 ms | 1.123x | 1.112x |
| 3 | 3.456 ms | 2.973 ms | 1.163x | 1.131x | 17.215 ms | 15.405 ms | 1.117x | 1.096x |

Promotion gates: boundary median at least 1.15x, depth-4 ViT training-step median at least 1.08x, and every aligned pair at least 1.03x. All gates passed.

The isolated complete-block forward measured about 1.27x faster and the explicit backward about 1.10x faster. Those decomposition runs diagnose the leverage but are not used for the promotion table.

## Correctness and fallback

- Portable F32 and F64 reference output and all 13 gradients match the two-boundary composite.
- Metal output, all 13 gradients, and zero input mutations match the incumbent tolerance.
- Two runtime epsilon feeds are covered without cache specialization.
- Depth-4 ViT logits and every parameter gradient match.
- Bias, LoRA, mask, causal mode, unsupported dtype/layout/shape, or a backend missing either complete direction routes through pre-norm attention followed by pre-norm FFN.
- The full Metal package passed. The full NLP package passed with only `TestDiffusionLMGrammarE2E` skipped; that test fails identically on untouched base HEAD with the same converged loss and generated string.

## Static performance scan

External perfscan `v1.81.1-0.20260823112112-5af3350190e4` was installed from the pinned mainline commit using `GOPROXY=direct`. The base baseline contained 1,704 findings; all 1,704 were suppressed for the candidate after normalizing a worktree-name defect, leaving 0 candidate-only findings.

The generalizable adjacent-fused-boundary pattern is reported in [perfscan issue #872](https://github.com/jxsl13/perfscan/issues/872). The reproduced worktree-dependent baseline defect is [perfscan issue #873](https://github.com/jxsl13/perfscan/issues/873).
