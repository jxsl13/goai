---
name: never-idle-during-throttle
description: "user correction 2026-07-15: NEVER no-op loop fires while waiting on the C16 push window — keep building the next task locally; commits accumulate and push as a batch"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: edba7aaa-e251-448f-8aa8-5b1950bfd99c
---

During §C16 push-throttle waits (or any external gate: CI runs, timers), loop fires must
NOT be "holding, no action" replies.

**Why:** the user explicitly corrected this ("continue working, never idle, bro",
2026-07-15) after ~15 no-op fires while waiting ~40 min for a push window. LOOP.md's
never-idle rule already said it: a blocked delivery thread is NOT a reason to idle —
advance another thread; commits accumulate locally and push as a batch when the window
opens.

**How to apply:** when a fire lands during a wait, pick the next §NEXT/§Tw task (or
verify-ahead, hardening, benchmarks) and BUILD it — stacked on the pending branch if it
depends on it, else on a fresh branch off main. Wall-clock waits are for background
timers, never for the worker itself.

Related: [[linux-amd64-worker-role]], [[cuda-q4-arc-state]].
