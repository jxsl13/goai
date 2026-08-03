---
schema: v1
---

## intent
- R-01KYZCQMQJEFXSAXKDK258YESC ops has no optimization headroom: it is a dispatch layer with no element loops and no benchmark: No action: closed as a measured null result. ops is a dispatch layer, three loops in the package all over op arity, no element loops, no benchmark anywhere in the tree that a change here would move. The one real finding, three fixed heap allocations per op call, is constant per call and needs a backend API decision plus a new benchmark; recorded in the body rather than turned into a task. Leverage [body truncated at tombstone retention cap]
