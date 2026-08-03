---
schema: v1
prefix: INDEPENDENT
---

## INDEPENDENT-CHAINS-COST-WHAT-ONE-CHAIN-COSTS-001
IF a reduction over streamed memory already carries 2 or more INDEPENDENT accumulator chains, THEN the an optimization that removes chains from it SHALL be rejected unless a measurement shows the loop is throughput-bound.

Rationale: Measured twice on the one-sided Jacobi SVD, in two different shapes, both slower. The pair body accumulates alpha, beta and gamma over one streamed pair of columns. Caching the two norms and recomputing them in a second pass after each rotation was 30 to 50 percent slower, and the note that recorded it blamed the extra pass. The obvious repair - a maintained nrm2 array refreshed INSIDE the rotation loop from the values it already writes, adding no pass anywhere - was also slower: SVD_192x192 90.4 to 109.3 ms, SVD_128x128 26.0 to 30.8, SVDPCA 37.6 to 48.7, with Cholesky512 flat as a control, and bit-identical throughout so only the clock disagreed. The extra pass was never the whole reason. Independent chains interleave into the latency of one chain, so three cost what one costs; deleting two buys nothing while the loop that absorbs the work gains chains it did not have. This is the converse of the PS3053 transform, which adds accumulators to remove PASSES and wins for that reason.
