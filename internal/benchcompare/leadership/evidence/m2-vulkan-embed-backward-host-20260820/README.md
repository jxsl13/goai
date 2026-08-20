# M2 Host-Resident Vulkan Embedding Backward (2026-08-20)

## Outcome

Retained. At GoAI's current synchronous host-resident tensor boundary, a deterministic typed F32
scatter is 3.7x to 36.0x faster by campaign median than the previous MoltenVK atomic route across
all five frozen shapes. It also removes nondeterministic atomic accumulation and is bit-identical
to the reference backend's per-add F32 rounding, including repeated indices.

Spectackle proposal: `P-01M0FQ0RWQE8P`. Task: `T-01M0FQ2Z5MFQ1`. Decision:
`ADR-01M0FQ1AP8FBC`. Contract: `HOST-RESIDENT-EMBED-BACKWARD-002`.
General perfscan finding: [jxsl13/perfscan#771](https://github.com/jxsl13/perfscan/issues/771).

## Why the route wins

The old implementation did all of the following for one synchronous operation:

1. allocate and upload float indices;
2. upload the upstream gradient;
3. allocate and upload a zeroed full output table;
4. submit and synchronize a Vulkan atomic scatter kernel through MoltenVK;
5. download the full output table.

The retained implementation zero-allocates the output table once and scatters typed host rows
directly. It performs no Vulkan command submission and no boundary copy. This is a route decision
for the present host-resident backend contract, not a claim that CPU scatter beats a future graph
whose embedding state and gradients remain GPU-resident across operations.

## Frozen setup

- Base: `0911d9d33fe65fa6023d83efe042556f36a00bc1`.
- Machine: Apple M2 Pro, macOS 26.5.1, Go 1.26.6, darwin/arm64.
- Vulkan path: MoltenVK via `/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json`.
- Control: preserved pre-change upload/atomic/download implementation.
- Candidate: deterministic typed host scatter with reference-order F32 rounding.
- Five shapes cover ViT sequence/position/class gathers and conventional embedding tables.
- Gate: every campaign median at least 1.20x faster; candidate max/min spread at most 3.0x;
  exact repeated-index cross-reference; full Vulkan and repository tests.

Command, run as three independent processes:

```text
go test -tags=vulkan ./backend/vulkan -run '^$' \
  -bench '^BenchmarkVulkanEmbedBackwardRoute$' -benchtime=1x -count=7
```

## Results

Each cell is control median / candidate median in nanoseconds, followed by control/candidate.

| Shape `[n,d,m]` | Campaign 1 | Campaign 2 | Campaign 3 | Worst speedup |
|---|---:|---:|---:|---:|
| `[513,128,520]` | 586,708 / 45,292 = 12.953x | 254,291 / 45,209 = 5.625x | 289,958 / 44,208 = 6.559x | 5.625x |
| `[65,128,520]` | 593,417 / 35,625 = 16.657x | 225,000 / 39,917 = 5.637x | 209,875 / 34,500 = 6.083x | 5.637x |
| `[520,128,8]` | 234,458 / 7,042 = 33.294x | 185,875 / 6,500 = 28.596x | 208,417 / 5,792 = 35.984x | 28.596x |
| `[4096,512,128]` | 1,353,500 / 334,042 = 4.052x | 1,596,042 / 211,792 = 7.536x | 838,000 / 224,916 = 3.726x | 3.726x |
| `[32768,128,512]` | 1,936,500 / 315,791 = 6.132x | 2,432,167 / 298,917 = 8.137x | 2,419,042 / 316,833 = 7.635x | 6.132x |

All 15 medians clear the 1.20x gate. Candidate spreads remained below the frozen 3.0x ceiling;
the worst observed was 2.828x for the 3–11 microsecond class-row case where timer/allocation
granularity dominates.

## Correctness and validation

`TestVulkanEmbedBackwardCrossReference` passes all existing cases, including repeated-index
collisions. The candidate deliberately reproduces the reference F32 rule by widening each add to
F64 and narrowing each store to F32; unlike the old atomic route, its accumulation order is stable.

The full tagged Vulkan/MoltenVK suite passed with `-short -count=1 -timeout 15m`, including
training, attention, quantized matmul, convolution, normalization, and backward paths. Repository
`make preflight` passed, and focused perfscan reported no candidate anti-patterns.
