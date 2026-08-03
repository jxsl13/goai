---
schema: v1
prefix: CPU
---

## CPU-002
WHERE the f32-native GEMM path under the SIMD experiment build, the cpu backend SHALL be compared against the f64 reference at rel 2e-3 and abs 1e-4 rather than at tolerance 0.

Rationale: This path accumulates in f32, so it amends the general f64-accumulation rule; observed maximum deviation is about 1e-4 for K up to 128. The default build keeps scalar f64 accumulation and stays bit-exact, f64 is bit-exact in both builds, and convolution through the banded GEMM is untouched. Migrated from the worker spec Iw4 and ADR-01KYCZF2W8F3GS2X0410JSHPKZ.

## intent
- R-01KZ15RPY3E9DT1W3ZXXGDA8Q8 Round T1060: masked-MHA mixed-dtype panic fixed; the check for that class drafted and withheld: Consumed: the mixed-dtype panic fixed on both arms with a reference-parity gate, and the check for the class withheld with its two noise sources diagnosed (a guard split across an enclosing type switch, and outputs allocated from an input dtype). Also records that the same class of bug is invisible to CI because the only test reaching it is skipped under -short.

## FANOUT-SIZING-PAYS-ONLY-AT-HIGH-CALL-FREQUENCY-001
IF a fan-out helper serves large operations called a few times rather than small ones called thousands of times, THEN the work-sizing transform of SIZE-THE-FANOUT-TO-THE-WORK-001 SHALL not be applied, because it measures neutral there and neutral is not a reason to add a knob.

Rationale: The transform that won on the quantized decode matmul - minus 4.1 percent wall and minus 27 percent system CPU - was applied to the two nn fan-out helpers and measured FLAT across three benchmarks with CPU flat too: HQQuantize 91.2 to 90.3 ms, KAN_256x256 11.99 to 11.91, TPAForward_1024 55.9 to 54.2, user CPU 25.7 to 26.1 s on both arms. Not shipped. What made decode different is CALL FREQUENCY AT NEAR-GATE WORK: thousands of calls per generation, each just above the threshold, so the wakeups dominated. MEASUREMENT DISCIPLINE MATTERS HERE MORE THAN USUAL: a first run at 10 iterations read as a 6 percent REGRESSION on one cell, and 20 iterations showed it was noise - separate the arms before believing either sign. TWO FURTHER NEGATIVES from the same round. The cpu backends worker pool ALREADY sizes itself, which is why perfscan PS3061 does not list it, and raising its per-worker floor made decode monotonically worse - 522, 529, 542 ms at 1<<15, 1<<16, 1<<17 - so its constants are at their optimum and should not be touched. And after the gguf fix, a decode profile still shows 92 percent of samples in pthread_cond_signal and pthread_cond_wait, but the wakeup path is runtime.startm, the Go schedulers own thread starting, not the pools barrier: the remaining lever on decode is FEWER AND LARGER OPERATIONS - kernel fusion - not more tuning of the existing helpers.
