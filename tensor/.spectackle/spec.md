---
schema: v1
---

## intent
- R-01KZ13PDMMFPXVCV19FGZMMNRD Round T1058: two per-element bounds and branch cleanups, -8.3 and -17 percent: Consumed: both cleanups shipped (-8.3 percent on RoundToHalfF32 with the BCE diagnostic confirming the checks gone, -17 percent on the two StridedCast cells), and the function-value pessimization measured at +51 and +66.5 percent became rule HOIST-A-BRANCH-BY-DUPLICATING-NOT-BY-A-FUNCTION-VALUE-001 — it is the form most people reach for, so the number is worth carrying.
