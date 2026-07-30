---
schema: v1
---

## intent
- T-01KYQH929EEFGVGYGEE5B836KP Fuse Swin's per-window slicing — 30k allocations per batched forward: Three steps shipped; the cheap structural wins are spent.

  allocations 30,481 -> 10,145 (3.0x fewer)
  bytes       31.2MB -> 24.0MB
  throughput  397.6 -> 470.5 img/s
BenchmarkSwinForwardBatched/batched, M2 Pro darwin/arm64, 3 alternations, min of 3 runs per
arm.

STEP 1 — fused the score scale, relative-position bias and window mask into one in-place pass
(3 dispatches and 3 [n,n] tensors per (window, head) -> 0). 30,478 -> 24,525.
STEP 2 — reused operand buffers for the per-(window, head) q/k/v blocks, writing kᵀ transposed
during the copy (7 tensors per block -> 0). 24,524 -> 10,960.
STEP 3 — wrote head outputs directly into one [numWin·n, dm] buffer, removing the per-window
head concat and the final window concat. 10,962 -> 10,145, time neutral.

ALL THREE ARE INFERENCE-ONLY, and for TWO distinct reasons that are worth keeping separate.
Step 1 is guarded because the relative-position bias is a LEARNABLE parameter and it is the
OpAdd that records its dependency — fusing it made TestSwinGradcheck report "param 8: nil
grad". Steps 2 and 3 are guarded because a recorder captures tensors BY POINTER, so refilling
a buffer across iterations leaves the graph holding whatever the last window wrote. Same
guard, unrelated hazards; conflating them would make one of the two look removable.

TWO ROUNDING TRAPS from step 1, both caught by the parity test rather than by reading. The
F32 op chain computes in float64 and rounds the STORE, so the fused arm must round after
every term, not once. And the scale arrives as an F32 SCALAR TENSOR, so the chain multiplies
by float32(1/sqrt(dk)) — using the float64 value is a one-ulp divergence. The F64 arm passed
from the first attempt, so a test on F64 alone would have shipped both.

A false start worth recording: step 1's first version showed EXACTLY ZERO change because the
benchmark model is F32 and only the F64 arm had been written. The measurement looked like a
null result rather than a bug. Probing which branch was taken is what distinguished them.

WHAT REMAINS, and why it is a different kind of work: the 10,145 are the softmax and the two
matmuls per (window, head), whose outputs feed the next op rather than being placeable
intermediates. Removing them means reproducing matmul and softmax rounding inside a fused
kernel — precisely what ADR-01KYQ9PHNPEFC declines after the Titans attempt cost three
iterations. Anything further should be weighed against that decision, not treated as a
continuation of these three steps.

## PERF-ORCHESTRATION-HAS-NO-LEVERAGE-001
IF a package benchmark scales poorly and every function of that package profiles at 0% flat, THEN the implementing agent SHALL stop: the leverage is in the callee package, so retarget rather than editing the caller.

Rationale: The vision package looked like the best remaining candidate: ViT forward batched scaled 3.06x with about 32ms serial, MLP-Mixer 2.99x, Swin 2.34x, and none of the three had been examined. A GOMAXPROCS=1 profile ended the search in one step. backend/cpu gemmF32Band is 55% flat, mhaFwd 7%, math.erf 5.5%, and EVERY vision function — ViT.Forward, Features, forwardOne, vitBlock.forward, forwardBatched — shows 0% flat with only cumulative time. They are pure orchestration over the backend, so nothing inside vision can move the number and the serial fraction belongs to gemm, which is another lane and already tuned. The cheap test is the per-package attribution of a one-core profile: a package whose own functions are all 0% flat has no work to optimize, however badly its benchmarks scale. Three steps and no edits — sweep the scaling ratio, profile at one core, attribute by package — is enough to reject a whole lane, and doing it before writing code is what kept this from becoming a wasted refactor.
