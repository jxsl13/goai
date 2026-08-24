# M2 Metal pre-norm transformer-stack evidence

Date: 2026-08-24

## Candidate

- Base HEAD: `eabf553f6aebbbb6703d0fccb35cc7cc47399b36`
- Candidate implementation commit: `a2687a46`
- Frozen benchmark binary SHA-256: `76b6d059c3cdb9b7ed897c058663b186dd2e0bd593d1e537b5faa114000adba3`
- Package: `github.com/jxsl13/goai/backend/metal`
- Shape: batch 8, sequence 65, dimension 128, heads 4, hidden 512, depth 4, F32
- Control disables only `OpPreNormTransformerStack` and its backward operation. All four independently promoted complete-block graphs remain enabled in both arms.

## Environment

- MacBook Pro Mac14,10, Apple M2 Pro, 12 CPU cores, 32 GB unified memory
- macOS 26.5.1 (25F80)
- Go 1.27.0 darwin/arm64
- `GOMAXPROCS=1` for the benchmark processes

## Protocol

One final candidate binary was compiled before measurement. Each campaign ran
in a fresh process with Go's adaptive benchmark duration and count 7. Campaigns
1 and 3 ran control before candidate; campaign 2 reversed the order with
`GOAI_PRENORM_STACK_CANDIDATE_FIRST=1`. Every process performed both built-in
warm-ups before timed work. Medians compare identical work and aligned pairs
use the same list position within one order-controlled campaign.

## Results

| Campaign | Boundary control median | Boundary candidate median | Speedup | Weakest boundary pair | Full control median | Full candidate median | Speedup | Weakest full pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.891 ms | 5.456 ms | 2.179x | 1.832x | 15.366 ms | 14.113 ms | 1.089x | 1.051x |
| 2 | 11.869 ms | 5.388 ms | 2.203x | 1.746x | 15.395 ms | 14.032 ms | 1.097x | 1.097x |
| 3 | 13.838 ms | 9.646 ms | 1.435x | 1.302x | 18.496 ms | 16.054 ms | 1.152x | 1.099x |

Promotion gates: boundary median at least 1.12x, depth-4 ViT training-step
median at least 1.05x, and every aligned pair at least 1.03x. All gates passed.

The full-step control reports 17,374,840 B/op and 1,304 allocs/op. The candidate
reports 15,745,792 B/op and 1,133 allocs/op, removing 1,629,048 B/op and 171
allocations/op.

## Correctness and fallback

- Portable reference output and all 49 gradients match the four-block loop.
- Metal output, all 49 gradients, and zero input mutations match the four complete-block submissions.
- Depth-4 ViT logits and every parameter gradient match.
- Two runtime epsilon feeds are shared across the uniform stack without cache specialization.
- Depth outside 2 through 8, nonuniform geometry or epsilon, unsupported features, dtype, layout, or backend directions routes through the exact complete-block helper loop.
- `make preflight`, `make preflight-metal`, and the full Metal package pass.
- The full NLP package passes with only `TestDiffusionLMGrammarE2E` skipped; that test was independently reproduced unchanged on the base in PR 1195.

## Static performance scan

External perfscan `v1.81.1-0.20260823112112-5af3350190e4` was installed from
the pinned mainline commit using `GOPROXY=direct`. It reports no finding on a
new implementation line. The generalizable result was added to
[perfscan issue #872](https://github.com/jxsl13/perfscan/issues/872#issuecomment-5389261981).
