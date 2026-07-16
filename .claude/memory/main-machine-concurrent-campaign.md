---
name: main-machine-concurrent-campaign
description: "The primary MacBook-M2/main machine runs its OWN fable optimization campaign on the SAME backend/cpu files — collision risk; how PR#44 was resolved"
metadata: 
  node_type: memory
  type: project
  originSessionId: 491e34f2-de40-467f-89a0-f918efb8a6b3
---

The primary dev machine (MacBook M2, owns `main`) runs its OWN parallel fable optimization campaign touching the SAME low-level files I optimize. Commits seen on main ~2026-07-15: `4060f93` (backend/cpu opt + norm-parity bugfix), `cf56280` (backend/ref), `56090de` (lowlevel simd/npy/format + integer-overflow OOB security fix). These land on main independently of my PRs → **collision risk on backend/cpu/{norm,mha,attn_extra,conv,pool,elementwise}.go**.

**PR#44 (my norm/RoPE/attn agent) collided with `4060f93`** which had modified norm.go/mha.go/attn_extra.go. I resolved the merge with `--ours` (my agent's versions), superseding the main machine's edits to those 3 files (its other-file changes intact). Correctness VERIFIED green: full backend/cpu+ref+autograd ref-parity suites pass against my versions (my agent independently respected the f64 ulp budget). The main machine's specific hardening there = explicit `float64()` FMA-fusion FENCES in the norm BACKWARD passes (robustness vs gc contraction drift); my version passes the same ulp tests NOW but lacks those explicit fences → latent robustness risk if gc's contraction choice changes. RECONCILED in PR#45: ported the float64() FMA fence into my column-blocked rmsNormBwd/layerNormBwd (kept the perf); TestNormKernelsMatchRefWithinUlps green. Collision fully closed.

**How to apply going forward:**
- Before merging a PR that touches backend/cpu low-level files, `git fetch` and CHECK whether the main machine changed the same files since my branch's fork; if so, do a proper 3-way reconcile (union both), NOT a blind `--ours`.
- Consider porting the main machine's FMA-fence hardening into my agent's norm.go to restore robustness while keeping the perf (best of both).
- Rebase my optimization branches on LATEST main right before finalizing to minimize collision.

See [[parallel-optimization-campaign]].
