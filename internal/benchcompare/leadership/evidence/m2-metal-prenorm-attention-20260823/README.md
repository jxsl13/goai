# M2 Metal pre-norm attention training evidence

Date: 2026-08-23

## Candidate

- Base HEAD: e4cb791de1494b9b1863058f389c8a67ba4d06ce
- Candidate worktree diff SHA-256: 4513ec3665425b8ce572604ccd2dc243df34b689b142cf40f0d659a9639c4a6c
- Frozen benchmark binary SHA-256: d4ffca65b80c15ca78a8f9f3d980de7dec7ceee6160d04d0c8e5a7fc6a2b8e9b
- Package: github.com/jxsl13/goai/backend/metal
- Shape: batch 8, sequence 65, dimension 128, heads 4, F32
- Control disables only OpPreNormAttention and OpPreNormAttentionBackward; the merged pre-norm FFN fusion remains enabled in both full-model arms.

## Environment

- MacBook Pro Mac14,10, Apple M2 Pro, 12 CPU cores, 32 GB unified memory
- macOS 26.5.1 (25F80)
- Go 1.27.0 darwin/arm64

## Protocol

A single final candidate binary was compiled before measurement. Three independent fresh-process campaigns used count 7. Campaigns 1 and 3 ran control before candidate; campaign 2 reversed that order. Boundary runs used -test.benchtime=20x; full ViT training-step runs used -test.benchtime=10x. Every benchmark process performed the benchmark's built-in warm-up before timed iterations.

## Results

| Campaign | Boundary control median | Boundary candidate median | Speedup | Full control median | Full candidate median | Speedup | Weakest aligned full pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.411 ms | 2.834 ms | 2.968x | 36.297 ms | 17.206 ms | 2.110x | 2.023x |
| 2 | 6.064 ms | 2.031 ms | 2.986x | 36.530 ms | 17.717 ms | 2.062x | 1.793x |
| 3 | 6.089 ms | 1.990 ms | 3.059x | 35.596 ms | 17.262 ms | 2.062x | 2.046x |

Promotion gates: boundary median at least 1.25x, full-step median at least 1.15x, and every aligned full-step pair at least 1.05x. All gates passed.

## Static performance scan

External perfscan v1.81.1-0.20260823112112-5af3350190e4 was run with
GOPROXY=direct against merged main and the candidate. The main baseline
contained 1,704 findings; all 1,704 were suppressed for the candidate, leaving
0 candidate-only findings.

The generalizable trainable-attention fusion pattern is reported upstream in
perfscan issue https://github.com/jxsl13/perfscan/issues/871.
