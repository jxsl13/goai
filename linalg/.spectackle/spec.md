---
schema: v1
prefix: INDEPENDENT
---

## INDEPENDENT-CHAINS-COST-WHAT-ONE-CHAIN-COSTS-001
IF a reduction over streamed memory already carries 2 or more INDEPENDENT accumulator chains, THEN the an optimization that removes chains from it SHALL be rejected unless a measurement shows the loop is throughput-bound.

Rationale: Measured twice on the one-sided Jacobi SVD, in two different shapes, both slower. The pair body accumulates alpha, beta and gamma over one streamed pair of columns. Caching the two norms and recomputing them in a second pass after each rotation was 30 to 50 percent slower, and the note that recorded it blamed the extra pass. The obvious repair - a maintained nrm2 array refreshed INSIDE the rotation loop from the values it already writes, adding no pass anywhere - was also slower: SVD_192x192 90.4 to 109.3 ms, SVD_128x128 26.0 to 30.8, SVDPCA 37.6 to 48.7, with Cholesky512 flat as a control, and bit-identical throughout so only the clock disagreed. The extra pass was never the whole reason. Independent chains interleave into the latency of one chain, so three cost what one costs; deleting two buys nothing while the loop that absorbs the work gains chains it did not have. This is the converse of the PS3053 transform, which adds accumulators to remove PASSES and wins for that reason.

## A-BIT-EXACT-DIGEST-MAY-NOT-SURVIVE-A-RACE-BUILD-001
IF a bit-exact digest test covers arithmetic the compiler may contract into a fused multiply-add, THEN the that test SHALL skip under -race with the reason recorded, rather than being re-frozen against the race build.

Rationale: TestSVDIsBitIdentical, added to gate a one-sided Jacobi SVD change, has been RED under go test -race on arm64 since it landed, and CI never saw it. The rotation is written as c*ai - sn*aj, which the compiler may contract into an FMA; whether it does differs between a normal build and a race build, and an FMA rounds once where a separate multiply and subtract round twice. The 8 by 8 case alone digests to 12638833736442736411 under -race against 3416335863526090039 without it. DO NOT RE-FREEZE AGAINST THE RACE BUILD: the frozen value would then be wrong for the build that ships. AND DO NOT GENERALIZE THE SKIP - eight sibling digest tests across classic, nn, autograd and backend/cpu were checked and all pass under -race, including a transpose parity test that does no arithmetic at all, which is what points at contraction rather than at concurrency. A blanket skip would give away gates that work.

## ALLOCATIONS-BLAMED-ON-A-FANOUT-HELPER-ARE-ITS-GOROUTINES-001
WHEN an allocation profile attributes most objects to a fan-out helper, the reader SHALL assume the goroutines it spawns, not any buffer inside the worker body, and confirm before optimizing either.

Rationale: A GPTQ quantization of a 128 by 512 weight allocates 11215 objects per call and an alloc_objects profile put 74 percent of them in linalg.parallelCols. The obvious reading is the per-worker substitution scratch, which perfscan also flags as allocated once per DISPATCH. Pooling that scratch changed NOTHING: 11218 to 11230 objects, time flat. The objects are the goroutines the helper spawns, and pprof attributes a goroutines allocation to the function that started it. The real lever is fewer or smaller dispatches. Sizing the worker count to the work, the transform that won on decode, cut allocations 11216 to 6627 on GPTQ and 12365 to 7778 on SparseGPT - minus 41 and minus 37 percent - with NO time win and a small possible cost, so it was NOT shipped, consistent with FANOUT-SIZING-PAYS-ONLY-AT-HIGH-CALL-FREQUENCY-001. That is now the third measurement of that transform outside the decode regime and the third time it failed to pay.
