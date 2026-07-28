---
schema: v1
prefix: EST
---

## EST-002 {applies: go:classic_test.TestPredictWidthGuardConsistency}
IF a predict-family method receives a row whose width differs from the feature count recorded at Fit, THEN the estimator SHALL return a clean error naming the row index, the got width and the want width, never panicking and never silently truncating.

Rationale: Two live failure modes: LinearRegression.Predict ranged over the input row instead of its coefficients, so a three-feature model given two features silently returned a plausible wrong number with a nil error; SVC panicked with an index-out-of-range while gradient boosting silently mispredicted, because shallow splits still traverse the surviving features. The guard lives at one site per estimator and is proven non-vacuous by stripping it and confirming the test goes red. Migrated from cavekit SPEC.md V37.
