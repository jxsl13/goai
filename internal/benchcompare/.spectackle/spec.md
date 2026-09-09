---
schema: v1
prefix: ALLOC
---

## ALLOC-SITE-ANALYSIS-001 {applies: rb:analyze_sites.analyze}
WHEN raw allocation-site snapshots are compared, the analyzer SHALL retain all rows and both collision layers, use checked integer arithmetic, and keep all 3 disjoint stack groups without inferring causation.

## ALLOC-SITE-MONOTONE-001 {applies: rb:analyze_sites.build_raw_keys}
WHEN allocation-site rows are normalized into function stacks, the analyzer SHALL reject a decrease in any of the 4 cumulative counters per raw key before coarser aggregation can hide it.

## ALLOC-SITE-FINITE-PARSE-001 {applies: rb:analyze_sites.parse}
WHEN allocation-site JSON is parsed, the analyzer SHALL reject every nonfinite Float at the root or any nested array or object depth, including overflowed exponent tokens, while preserving exact Integer counters.
