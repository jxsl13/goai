---
schema: v1
---

## ADR-01M0FVWNPKEX6917B1N0VBM0FJ How should synchronous host-resident F32 Metal bias gradients route after the CPU reduction optimization?
kind: adr
state: done
created: 2026-08-20
context: Three independent count-7 M2 campaigns show the production CPU selector is 3.263x to 199.71x faster than direct Metal through 2,097,152 elements, with worst candidate spread 1.788x, exact reference parity, and 0.994x end-to-end GPT throughput.
decision: Route measured shapes through CPU and preserve direct Metal above the bound
consequences: F32 rank-2 gradients with positive dimensions and at most 2,097,152 elements use the exact optimized CPU kernel with recorder suppression. Larger, unsupported, or CPU-unavailable cases retain the isolated direct Metal implementation. Future device-resident graph execution requires a new benchmark boundary and does not inherit this host-resident decision.
status: accepted

kind: radio
option: Route measured shapes through CPU and preserve direct Metal above the bound
option: Retain direct Metal for all shapes
option: Remove the direct Metal implementation
blocks: T-01M0FVGM88EWMRQCHFN4B748AV
choice: Route measured shapes through CPU and preserve direct Metal above the bound

## ADR-01M0M9MPAGFM9T4AMA7D97HWJD Which native Metal Q4_1 kernel shape should be the production baseline on M2?
kind: adr
state: done
created: 2026-08-22
context: Q4_1 uses a 20-byte affine block with d and m. Direct decoding preserves compressed residency and avoids transformation traffic. Q4_0 already proves the scalar and cooperative occupancy shapes.
decision: Separate scalar and two-SIMD-group cooperative pipelines derived from Q4_0
consequences: The kernel decodes Q4_1 directly from 20-byte resident blocks. Synchronous, resident, and recorder paths share the same cached pipelines. Scalar is the capability fallback; cooperative is the M2 path. No dense expansion or transient Q4_0 transformation is allowed.
status: accepted

kind: radio
option: Separate scalar and two-SIMD-group cooperative pipelines derived from Q4_0
option: Reuse Q4_0 after transforming Q4_1 weights or activations
option: Materialize dense F32 weights before Metal GEMM
blocks: P-01M0M9B6FRFCZA18408PMM2WGH
choice: Separate scalar and two-SIMD-group cooperative pipelines derived from Q4_0

## P-01M23TTRCAFA4TFBHCJ0BWE4F2 Keep decode-attention smoke checks in short CI and reserve timing ceilings for full local runs
kind: proposal
state: active
created: 2026-09-09
grilled: 2026-09-09 open=0
targets: go:metal.TestMHADecodeCost

Post-merge CI34394770011 at fe7a9bfcc1b4f6b589aa65bf3d61bbe2998a87ae failed unchanged TestMHADecodeCost: llama7b dk128 sk512817.7us exceeds600us under -short on shared hosted macOS. PR1250 pre-merge all16jobs and executedsteps passed; the repaired prefill test did not fail. Apply existing TIMING-ASSERTIONS-SKIP-ON-RUNNERS-001 to this separate test. Short mode must still execute one recorded MHA operation for each4 model/context combinations with existing recorder/MHA error checks, Commit/Wait/Free and buffers. Full mode retains exact25repetitions, meas32/meas256 slope, four geometries and600us ceiling. No production/workflow changes, no threshold increase, no timing/performance gain claim. Fresh implementer and verifier must execute real Metal short/full default+SIMD; explicit no-Metal skips are not successful execution. Retain CI failure and full evidence, report generalized followup to existing perfscan868, create properPR, merge only when alljobs/steps succeed, then delete exact remote merged branch. Existing user authorization covers valid test correction and publication.

## T-01M23TWGZ5F2EBJRR0CTPCXGZ6 Honor short mode in MHA decode timing test while retaining four operation smoke cases
kind: task
state: active
created: 2026-09-09
parent: P-01M23TTRCAFA4TFBHCJ0BWE4F2
targets: go:metal.TestMHADecodeCost

OWNERSHIP: fresh implementer owns ONLY backend/metal/mha_decode_bench_test.go in /private/tmp/goai-mha-short-lEmJsP/repo. Root owns evidence/docs/lifecycle/Git/PR. Read .claude/commands/spectackle.md completely and fetch this task. apply_patch only; do not edit runtime, workflows, other tests or .spectackle bytes. No commits, pushes or lifecycle mutations. Report actual results and errors, including skips.

IMPLEMENT: existing TestMHADecodeCost violates TIMING-ASSERTIONS-SKIP-ON-RUNNERS-001 under shared CI -short. Introduce repeats=25, set repeats=1 when testing.Short, preserving same recorder creation/error checks, MHA errors, Commit, Wait, time read, Free. For each existing model/context pair, when short call meas(1), log a distinctive smoke-complete line identifying model/dk/sk, then continue before slope or ceiling. There are exactly4 cases: tinyllama dk64 sk36/512 and llama7b dk128 sk36/512. Do not early-skip the whole test on short. Preserve Available skip if noMetal, all buffer lifetimes, unchanged full25repetitions with meas32/meas256, same600us ceiling and original shape/semantics. No production or threshold changes; operation smoke is not a new numerical-parity assertion.

VERIFY: pinned Go1.27.1 via PATH=/private/tmp/goai-go1271-j1cLvq/go/bin:$PATH GOCACHE=/private/tmp/gocache-goai GOMODCACHE=/private/tmp/goai-go1271-j1cLvq/modcache GOTOOLCHAIN=local CGO_ENABLED=1. Execute actual Metal focused TestMHADecodeCost short/full with -count=1 -v in default and GOEXPERIMENT=simd. Inspect logs to require4short smoke lines or4full timings and noSKIP; sandbox noMetal is NOT execution proof, use escalation forrealMetal asneeded. Run full backend/metal -short -count=1 -timeout20m default/SIMD and vet both. gofmt-d/diffcheck. Preserve full literal stdout/stderr for every command under /private/tmp/goai-mha-short-lEmJsP/implement-*.log via apply_patch or capture tool outputs; never summarize them as raw. No profiles/newbenchcampaigns. Root will schedule independent fresh verifier. Heavy testing window granted only to this implementer after dispatch; root allocation campaigns are completed and independent allocation verifier runs only lightweight analysis. Before each full timing run ensure no other owned heavy window. Stop after bounded matrix and report heavywindowclosed. Record code SHA256/diff.
