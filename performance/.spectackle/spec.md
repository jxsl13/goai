---
schema: v1
prefix: A
---

## A-REPEATED-SCAN-IS-BANDWIDTH-BOUND-TILE-IT-001
WHEN per-item work re-reads a shared collection that the item loop does not index, the optimizer SHALL treats the loop as bandwidth-bound and tiles the item loop rather than tuning the arithmetic; on the measured site 4 accumulators bought 7 percent while a tile of 16 bought 24.

## A-REUSED-STAGING-WINDOW-CHECKS-CONSUMER-ACCUMULATION-001
WHEN replacing a whole-tensor staging buffer with a reused per-chunk window, the optimizer SHALL establishes whether the consuming kernel stores or ACCUMULATES into it, and clears the window between chunks when it accumulates; the whole-tensor form wrote each row slot once, so an accumulating consumer is indistinguishable from a store there.

## A-CHUNK-SIZE-IS-CAPPED-AT-ONE-BAND-001
WHEN sizing a per-worker scratch chunk from a cache-residency target, the optimizer SHALL also caps the chunk at one bands worth of rows, so total scratch stays below the buffer it replaces instead of becoming workers times chunk; uncapped it measured -22.7 percent on the largest shape and +14 to +33 percent on small ones.

## A-SERIAL-STRETCH-IS-READ-AGAINST-WALL-CLOCK-NOT-PROFILE-SHARE-001
WHEN ranking a serial function inside an otherwise parallel path, the optimizer SHALL divides its CPU-profile share by the paths average parallelism before dismissing it, because a serial stretch costs its full CPU time in wall clock; the head mix read as 2 percent of a summed profile at parallelism 6.2 and was 12 percent of the benchmark.

## A-RUNTIME-MEMORY-SHARE-IS-NOT-A-TIME-LEVER-001
WHEN a CPU profile attributes a large share to runtime.madvise or GC, the optimizer SHALL treats it as a resource finding until a paired benchmark shows otherwise; removing 90 percent of the AQLM encoder allocations moved 45.7 MB per op to 4.4 and the time not at all, because that work overlapped on other threads.

## A-COLLIDING-SCATTER-SPLITS-ON-ITS-DESTINATION-DIMENSION-001
WHEN a scatter-accumulate whose index is a loop dimension times a stride plus a data-dependent offset, the optimizer SHALL splits the DIMENSION rather than the items, keeping each slot accumulating in ascending item order for a bit-identical result; per-worker partial copies merged afterwards reassociate every sum and are rejected on that ground.

## A-PARALLEL-SCALING-PROBE-RANKS-BEFORE-A-PROFILE-001
WHEN choosing which benchmark cell to optimize, the optimizer SHALL measures each candidate at GOMAXPROCS 1 and at full width first and works the lowest speedup, because a profile share cannot distinguish serial from parallel time; this found GBMHist at 1.29x among cells scaling 6 to 7x.

## A-WORK-GATE-IS-VERIFIED-AGAINST-THE-REAL-SHAPE-001
WHEN a parallel split measures as no change at all, the implementer SHALL prints the gate inputs from the benchmark before concluding anything; the softmax Hessian split read as exactly zero effect because its work estimate came to 252000 against a 262144 threshold and the fork never ran.

## A-TRIANGULAR-INNER-RANGE-IS-BANDED-ON-CUMULATIVE-WORK-001
WHEN splitting a loop whose iteration a costs m minus a, the optimizer SHALL cuts the bands on cumulative work rather than on iteration count, since an equal-count split gives the first band about 2m over workers times the last bands work and the makespan is the first bands.

## A-FANOUT-WIDTH-IS-DERIVED-FROM-THE-WORK-001
WHEN a work-splitting helper picks its worker count, the pool SHALL derives it as min(GOMAXPROCS, total/floor) and runs serial at one, rather than jumping to full width the moment a total-work threshold is cleared; without the floor a per-token decode ran 2.3x slower at 12 cores than at 1, and adding it took 3 decode benchmarks down 27 to 45 percent.

## A-SCALING-PROBE-COVERS-EVERY-PACKAGE-001
WHEN the parallel-scaling probe is run to rank work, the optimizer SHALL runs it over every package rather than the ones with obvious kernels; the nlp decode cells were the only ones in the tree scaling BELOW 1x and were found only after 5 rounds of probing elsewhere.

## A-POOL-DENSE-RESET-ON-SERIAL-OPS-IS-REJECTED-001
WHEN considering a reset of the worker-pool dense flag when an op runs serially, the optimizer SHALL does not: measured on 4 decode benchmarks it moved neither wall time nor CPU seconds (user 9.5 to 10.6 both arms), because after the work floor landed the dense classification already times out and the workers are parked on their mailboxes.
