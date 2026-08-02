---
schema: v1
prefix: EST
---

## EST-002 {applies: go:classic_test.TestPredictWidthGuardConsistency}
IF a predict-family method receives a row whose width differs from the feature count recorded at Fit, THEN the estimator SHALL return a clean error naming the row index, the got width and the want width, never panicking and never silently truncating.

Rationale: Two live failure modes: LinearRegression.Predict ranged over the input row instead of its coefficients, so a three-feature model given two features silently returned a plausible wrong number with a nil error; SVC panicked with an index-out-of-range while gradient boosting silently mispredicted, because shallow splits still traverse the surviving features. The guard lives at one site per estimator and is proven non-vacuous by stripping it and confirming the test goes red. Migrated from cavekit SPEC.md V37.

## intent
- R-01KZ1352M6FV0B7DG80S6XMCYP Presort column-major relayout measured and REJECTED: -3.7% on one cell, nothing on the other: No action: column-major relayout measured -3.7 percent on TreeFit and +0.25 percent on GBMFit and was reverted. The gather is already hoisted once per node by an earlier optimization, so it is the prologue rather than the loop, and relayout helps a gather that IS the loop. Also records the two-construction-path trap: RandomForestClassifier builds per-tree cartBuilders that never call initColumns, [body truncated at tombstone retention cap]
- R-01KZ1382YDF4KVFBEPC7GMD0SN SVM kernel-switch hoist measured and REJECTED: +2.7 percent, slower: No action: measured at +2.7 percent (slower) on BenchmarkSVCFit/n4000_rbf and reverted. The call overhead the hoist removes is small next to the 20-dimension distance plus math.Exp it wraps, and the SMO iteration dominates. Also corrects the package-level claim that this was the route to the 2.0x RBF-SVC gap against sklearn: it is not, and the next attempt should profile SMO rather than kernel eva [body truncated at tombstone retention cap]
- R-01KZ16EHFGEK9SA8QVA2TQCPDW Round T1061: GaussianNB row claiming -5.5%, with PS3011 diagnostics run first: Consumed: GaussianNB row claiming shipped at -5.4 percent with PS3011 own diagnostics run first, and the monotonic GOMAXPROCS sweep recorded as the reason the win is small. The fixture lesson became rule A-PARALLEL-GATE-MUST-CLEAR-THE-SERIAL-THRESHOLD-001. Five sibling loops remain, each needing its own screen rather than an assumption that the shape transfers.
