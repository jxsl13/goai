# M2 Host-Resident Metal Embedding Backward (2026-08-20)

## Outcome

Retained. At GoAI's current synchronous host-resident tensor boundary, a deterministic typed F32
scatter is 3.9x to 30.8x faster by campaign median than the previous Metal atomic route across all
five frozen shapes. It also removes nondeterministic atomic accumulation and is bit-identical to
the reference backend's per-add F32 rounding, including repeated indices.

Spectackle proposal: `P-01M0FNKC7DEJZ`. Task: `T-01M0FNQVFTFQ0`. Decision:
`ADR-01M0FNMY7XF3S`. Contract: `HOST-RESIDENT-EMBED-BACKWARD-001`.
General perfscan finding: [jxsl13/perfscan#771](https://github.com/jxsl13/perfscan/issues/771).

## Why the route wins

The old implementation did all of the following for one synchronous operation:

1. allocate and upload float indices;
2. upload the upstream gradient;
3. allocate and upload a zeroed full output table;
4. submit and synchronize a Metal atomic scatter kernel;
5. download the full output table.

The retained implementation zero-allocates the output table once and scatters typed host rows
directly. It performs no Metal command submission and no boundary copy. This is a route decision
for the present host-resident backend contract, not a claim that CPU scatter beats a future graph
whose embedding state and gradients remain GPU-resident across operations.

## Frozen setup

- Base: `fd6d08bb03188224ce3e63761b71b8581f8752bb`.
- Machine: Apple M2 Pro, macOS 26.5.1, Go 1.26.6, darwin/arm64.
- Control: preserved pre-change upload/atomic/download implementation.
- Candidate: deterministic typed host scatter with reference-order F32 rounding.
- Five shapes cover ViT sequence/position/class gathers and conventional embedding tables.
- Gate: every campaign median at least 1.20x faster; candidate max/min spread at most 3.0x;
  exact repeated-index cross-reference; full Metal and repository tests.

Command, run as three independent processes:

```text
go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalEmbedBackwardRoute$' -benchtime=1x -count=7
```

## Results

Each cell is control median / candidate median in nanoseconds, followed by control/candidate.

| Shape `[n,d,m]` | Campaign 1 | Campaign 2 | Campaign 3 | Worst speedup |
|---|---:|---:|---:|---:|
| `[513,128,520]` | 222,750 / 44,500 = 5.006x | 224,000 / 37,292 = 6.007x | 213,416 / 42,916 = 4.973x | 4.973x |
| `[65,128,520]` | 184,542 / 34,084 = 5.414x | 182,875 / 36,042 = 5.074x | 184,125 / 33,583 = 5.483x | 5.074x |
| `[520,128,8]` | 182,042 / 6,208 = 29.324x | 188,416 / 6,125 = 30.762x | 186,542 / 6,916 = 26.973x | 26.973x |
| `[4096,512,128]` | 1,247,250 / 317,250 = 3.931x | 1,195,125 / 265,583 = 4.500x | 1,233,542 / 310,125 = 3.978x | 3.931x |
| `[32768,128,512]` | 1,569,291 / 397,708 = 3.946x | 1,628,458 / 287,541 = 5.663x | 1,607,583 / 288,333 = 5.576x | 3.946x |

All 15 medians clear the 1.20x gate. Candidate spreads remained below the frozen 3.0x ceiling;
the worst observed was 2.747x for the 3–10 microsecond class-row case where timer/allocation
granularity dominates.

## Correctness

`TestMetalEmbedBackwardCrossReference` passes all existing cases, including repeated-index
collisions. The candidate deliberately reproduces the reference F32 rule by widening each add to
F64 and narrowing each store to F32; unlike the old atomic route, its accumulation order is stable.
