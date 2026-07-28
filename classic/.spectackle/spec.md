---
schema: v1
prefix: EST
---

## EST-002 {applies: go:classic_test.TestPredictWidthGuardConsistency}
IF a predict-family method receives a row whose width differs from the feature count recorded at Fit, THEN the estimator SHALL return a clean error naming the row index, the got width and the want width, never panicking and never silently truncating.

Rationale: Two live failure modes: LinearRegression.Predict ranged over the input row instead of its coefficients, so a three-feature model given two features silently returned a plausible wrong number with a nil error; SVC panicked with an index-out-of-range while gradient boosting silently mispredicted, because shallow splits still traverse the surviving features. The guard lives at one site per estimator and is proven non-vacuous by stripping it and confirming the test goes red. Migrated from cavekit SPEC.md V37.

## CONC-SCRATCH-FIELD-001
IF a method uses a receiver slice field as a per-call temporary, THEN the buffer SHALL be passed as a parameter before any caller is parallelized; perfscan PS6006 detects it.

## NUM-ARGMAX-TIEBREAK-001
IF an argmax reduction is parallelized and its candidates combined afterward, THEN the combine SHALL be tested on a constructed exact tie, since random data yields no bit-equal scores and prediction goldens stay green under a reversed combine order.

## intent
- T-01KYN61HYKE58V81832PV4NSKT Parallelize the CART sweep in tree.go once the worker's optimization line there settles: DECLINED ON LEVERAGE, not on collision, and the leverage argument is the useful part.

The task deferred parallelizing the CART sweep because classic/tree.go was a collision zone. That zone has since quieted — no open PR touches it — so the deferral reason expired and the work was re-examined rather than started.

IT IS NOT WORTH DOING. BenchmarkForestFit measures 7.85x across GOMAXPROCS 1..12 (1055 -> 134ms) because a forest already parallelizes ACROSS TREES, which saturates the machine. Adding sweep-level parallelism inside each tree would nest under that: internal/parallel's non-blocking submission finds every worker busy and runs the chunk inline, so a forest gains nothing. That is by design — the unbuffered mailbox makes "pool busy" and "send fails" the same event — and is covered by TestRowsNestedDoesNotDeadlock, though the no-gain consequence is derived from the design rather than measured.

The only workload that WOULD gain is a single-tree fit: BenchmarkTreeFit, 10.07ms at 1.00x. Against 10ms, the cost is a shared-scratch split, an argmax-combine with a constructed tie fixture (NUM-ARGMAX-TIEBREAK-001), and a frozen golden for a grower that has none — the same package of work that took the GBM exact path from 1865 to 667ms. Wrong ratio.

THE TRANSFERABLE POINT: when an outer loop already parallelizes to near machine width, an inner parallelization is not additive — it is inert. Check where the existing parallelism lives before splitting a hot loop inside it. The GBM exact grower was worth 2.80x precisely because boosting is SEQUENTIAL across trees, so no outer parallelism existed to saturate the cores; forests are the opposite case.

ALSO MEASURED AND LEFT: BenchmarkKNNFit 3.96ms at 1.00x (ball-tree build, bt.splitKey is the receiver scratch PS6006 flags). Same ratio problem at 4ms.

classic is now swept: ForestFit 7.85x, DBSCANFit 6.16x, GBM exact 2.80x, GMM 4.09x, KNN Predict 7.00x, GBM histogram 1.57x — all either parallel or measured and declined.
- T-01KYNA8X45ERT956JBG620TWX1 Triage the remaining PS3002 sort.Slice sites for the reflect.Swapper allocation: DONE. Three sites converted, five declined, triaged by CALL FREQUENCY as the task specified.

CONVERTED (per node or per query): ballTree.build, ballTree.kNN, nearest.
  KNNPredict  36,004 -> 24,003 allocs (1.50x), 1.63 -> 1.28 MB, 13.90 -> 13.54ms
  KNNFit       1,539 ->  1,029 allocs (1.50x), 3.85 -> 3.68ms

DECLINED (once per call): classic/gbm.go:275 (per builder), linalg/svd.go:100 (per SVD), nlp/beam.go:128, nlp/embed.go:124, nlp/chattemplate.go:172. One swapper allocation each; converting them is churn.

THE TRIAGE CRITERION IS CALL FREQUENCY, NOT SLICE LENGTH, and these numbers show why: the KNN sorts handle SHORT slices — k results, or one node's indices — and still returned 1.50x, because reflectlite.Swapper is allocated per CALL regardless of length. The earlier CART conversion returned 3.11x for the same reason at higher frequency. A long sort called once is worth nothing here; a short sort called a million times is worth everything.

TWO ORDERING CLAIMS PROBED RATHER THAN ASSERTED. The kNN and brute-force comparators are TOTAL orders on (dist, idx) with unique idx, so the permutation is identical. The build sort is unstable and may reorder ties — harmless, and the probe demonstrates it: reversing the build order OUTRIGHT leaves every KNN test green, because the search is exact and orders results by (dist, idx), so the same k neighbours return whatever shape the tree takes. That is a stronger statement than the comment made and it is now measured.

THE PROBE FOUND A REAL GAP. Inverting the (dist, idx) tie-break in EITHER path also left every existing KNN test green — random data never yields two bit-equidistant points, so an order documented as 'identical to a full brute-force sort by (dist, idx)' was never checked. Closed with a constructed fixture: two points mirrored about the query so distances tie to the bit, plus filler to exceed ballLeafSize so the TREE path is exercised and not only brute force. Red under both inversions.

Third instance in this campaign of a comparison-decided output being untestable without a deliberately constructed tie, after the GBM split argmax and the AQLM ICM argmin. NUM-ARGMAX-TIEBREAK-001 covers the pattern; this is its second application.

## PERF-NESTED-PARALLEL-001
IF an outer loop already parallelizes to near machine width, THEN the inner loop SHALL not be parallelized as well; it runs inline under a busy pool and adds nothing, as ForestFit at 7.85x shows for the CART sweep.

## PROC-BENCH-MEMAXIS-001
IF a parallelization allocates scratch inside the parallel body, THEN the change SHALL report allocs and bytes per op alongside the speedup; GBM hid a 31x memory regression (64MB to 2007MB) behind a 2.80x.
