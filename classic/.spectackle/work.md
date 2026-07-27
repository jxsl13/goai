---
schema: v1
---

## T-01KYJPYC1JEKQT3X1F1A7A26VY Carry values and labels in lockstep with the presorted columns in the split sweep
kind: task
state: draft
created: 2026-07-27

Measured baseline on this host: BenchmarkTreeFit 11.80ms (4000x20, 3-class, unlimited depth), BenchmarkForestFit 184.3ms, BenchmarkSVCFit/n4000_rbf 7.25ms.

SITES: classic/tree.go:552 (*cartBuilder).sweep — the value gather at :558-561, regression target gathers at :566-568 and :571, classification label gathers at :595-597, :617, :641. Threshold re-read at tree.go:537-538 in bestSplit. Mirror defect in the GBM exact grower at classic/gbm.go:329-331 and :334.

WHY HOT: Fit -> fitWithSeed -> build -> bestSplit -> sweep, once PER FEATURE PER NODE. Per node of m samples the sweep does d*m value gathers plus d*m (classification) or 2*d*m (regression) target gathers; over a tree that is d*n*depth — for the benchmark shape 20 x 4000 x about 15 levels, roughly 1.2M value gathers plus as many label gathers. This is the dominant memory traffic of a single-tree fit and of every GBM weak learner, since GBM's DEFAULT grower is the exact gbmBuilder, not the opt-in histogram one.

DEFECT: order[k] walks samples in sorted-by-feature order, a pseudo-random permutation of row order. b.x is a slice-of-slices, so each element costs TWO DEPENDENT random loads — the 24-byte slice header at b.x[id], then row[f] inside a separately allocated 160-byte row. At n=4000, d=20 the row data is about 640KB, over L1, so every gather is an L2 round trip with no prefetch possible and the second load cannot issue until the first retires. The same permutation is then re-walked to gather the labels. Comments at tree.go:555-557 show a previous pass halved the COUNT of these gathers; the gathers themselves were never removed. This is precisely where sklearn's C splitter has its structural edge: it holds X as one contiguous Fortran-ordered buffer, so its Xf fill is a strided-but-linear read rather than a pointer chase — and decision tree is one of the two estimators this library currently LOSES on (about 1.3x).

FIX: carry the sorted PAYLOAD alongside the sorted INDEX. In initColumns (tree.go:282) allocate from single backing arrays colVal [][]float64 (d x n) and, classification only, colLab [][]int32 (regression carries colY instead), filled at presort time so colVal[f][p] == b.x[cols[f][p]][f]. In partition (tree.go:324-350) move the value/label element in lockstep with cf[w]=s — one extra sequential read and write per element per feature, introducing NO new gather because the payload travels with the index already being moved. sweep then takes vals := b.colVal[f][start:end] directly and drops the fill loop entirely; class/target reads become labs[p-1]. bestSplit's threshold reads become vals[cut-1], vals[cut]. Apply the same to gbmBuilder (gbm.go:238/:266/:314/:358); GBM's fit filter loop at :279-284 already streams the master column so carrying a value twin there is free. Memory cost about +960KB at the benchmark shape — guard it behind an n*d budget with a fallback to today's gather path, which is result-identical either way.

VALIDATION GATE (benchmark only): BenchmarkTreeFit (classic/treefit_bench_test.go:41) and BenchmarkGBMFit (:64) cover this today. Add BenchmarkTreeFitShapes sweeping {2000,10,3}, {4000,20,3}, {8000,50,3}, {4000,20,8} — THE WIN MUST GROW WITH n*d as the row data leaves L1/L2, and that gradient is the proof the fix is the gather rather than something else. Add the DecisionTreeRegressor variant (it carries 2x the target gathers so it should improve most) and a WithCriterion(Entropy) case.

EXPECTED: 1.4-1.9x on BenchmarkTreeFit (11.8ms -> about 6.5-8.5ms), larger on the regressor and at wider d. High confidence that it is a real win, medium on the exact factor since L2 hit latency is partly hidden by the sweep's arithmetic.

BIT-IDENTITY BAR: bit-identical — a pure data-layout change. Every arithmetic expression, its operand order, and the candidate-split acceptance tests are untouched, so thresholds and chosen splits are unchanged. Pin the invariant colVal[f][p] == x[cols[f][p]][f] across partition with a debug assertion. Does NOT touch any predict-family width guard (tree.go:861-863, :926-928) — those live in Predict, which never sees the builder. Existing goldens classic/testdata/trees.json and gbm.json are the regression net.

PERFSCAN RULE REQUIRED: indexed-payload gather in a hot permutation walk. AST shape: a for loop containing an index expression A[B[i]][c] or A[B[i]] where B is a []int that is itself the loop's iteration source or a slice of one, A is a struct-level [][]T or []T not written in the loop, and the enclosing function is called from a loop nest at least 2 deep. Report when the gathered result is written into a dense scratch slice (scratch[k] = A[B[k]][c]) — that write is the tell that the value is being materialized in permuted order, which means it could have been maintained in that order instead. Highest-confidence sub-case is A being [][]T (double indirection). Suppress when B is provably the identity or ascending.
