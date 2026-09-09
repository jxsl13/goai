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

## P-01M23R1VJNEJJA8QB17ET720FM Keep prefill operation smoke checks in short CI while reserving timing ceilings for local runs
kind: proposal
state: active
created: 2026-09-09
grilled: 2026-09-09 open=0
targets: go:metal.TestPrefillOpCosts, TIMING-ASSERTIONS-SKIP-ON-RUNNERS-001

Existing CI run34390073218 at76d889fb failed only cgo+metal/macOS TestPrefillOpCosts: RMSNorm68.02us exceeded60us absoluteceiling. This source is unchanged from mainb9c05946 and prior checkpoint72410b07 passed; do not retry until lucky or weakenfullthreshold. Existing TIMING-ASSERTIONS-SKIP-ON-RUNNERS-001 requires timing budgets skip under testing.Short and correctness assertions remain unconditional; precedent archivedP-01M0QV7XJ3EW6 handles RoPE timing. Independent branch frommain, no GELU runtime or failedqualification waiver. Modify only TestPrefillOpCosts: in short mode run one recorded operation per named case to preserve devicebuffer,recorder,operation-error,commit/wait/free smoke behavior; do not run16/128×15 timing workload or compare wallclockceilings. Nonshort measurement protocol,15repeats,min slope,shapes and all60/90/40/600/1200us thresholds byte/semantic unchanged. Keep all numerical tests unaffected. Freshimplementerandverifier; realM2 shortsmokeandfulltiming tests default+SIMD serialized beforeallocationdiagnostics. Fullmodebaseline/candidate logs retained but no kernelperformanceclaim. Verify APIerrorhandlingandalltestlimits source diff. Proper separatePR, allCIjobsANDsteps greenbeforemerge, remoteexactbranchdeleteaftermerge.

## T-01M23R33X1EC7BA8V687WEM8QH Honor short mode in prefill timing test without dropping operation smoke checks
kind: task
state: active
created: 2026-09-09
parent: P-01M23R1VJNEJJA8QB17ET720FM
grilled: 2026-09-09 open=1
targets: backend/metal/prefill_ops_bench_test.go

OWNERSHIP onefile only backend/metal/prefill_ops_bench_test.go in /private/tmp/goai-prefill-ci-O70L2w/repo branchcodex/metal-prefill-timing-short-20260909 baseb9c05946. Follow .claude/commands/spectackle.md; CLI getparent/task. Root owns docs/evidence/lifecycle/commits/push/PR. No runtime,CIworkflow, thresholds, separatecorrectness tests or unrelatedfiles edited. apply_patch only. No mutations/lifecycle/commits/heavywork until root permission. First return implementation diff and proposedverification commands; root schedules heavyM2 checks to avoid allocationdiagnostic overlap.
CAUSE CI34390073218/GELUevidencehead76d889fb failed existingTestPrefillOpCosts RMSNorm68.02us>60us sharedrunner. Existing rootTIMING-ASSERTIONS-SKIP-ON-RUNNERS-001 mandates shortmode bypassperformancebudgets whilecorrectness remains. Testcurrently performs15repeats of16and128ops for5named cases. Keep all buffer/recorder/rec callback error checks and commit/wait/free smoke calls inshortmode. Minimalimplementation: for each slope's existing meas helper choose repeatcount1 iftesting.Short else15; afterhelperdefinition, shortmodecallsmeas(1), t.Logf namedcase smoke completion, returnsBEFOREmeas16/128 orceil comparison. Nonshortsame15,min,slope,thresholds,opshapes,recorder lifecycle andall error paths; no simplyskippingwholetest. Documentonlyabsolute timingworkloadskip. No new global flags/durations/testhelperframework. Full numerical Metaltests unaffected.
VERIFY once rootgrantsheavywindow: pinnedGo1.27.1 PATH/GOCACHE=/private/tmp/gocache-goai/GOTOOLCHAIN=local/CGO_ENABLED1. Focused go test ./backend/metal -run '^TestPrefillOpCosts$' -short -count=1 -v default andGOEXPERIMENTsimd mustPASSandlog5smokecases; nonshortsamefocused default+SIMD mustretain5timingoutputs andunchangedceilings; baselinebeforeeditlogs usefulbut don't mutate concurrentownedchanges. Full -short Metalpackage default+SIMD once separatelyifruntimeaccessible (otherwise transparentlysurfaceGPUavailability,no fabricatedPASS). vetpackagebothmodes, gofmtdiff/diffcheck. Freshverifierrepeatsfocusedmatrixfromdiff and checksliteral5fullceilingsandmeasprotocolunchanged,APIerrorsstillactiveinshort; no kernelperfclaim fromtestduration. CPU/GPUheavychecks serialized with allothergoai campaigns; aftertests stopheavywork andnotifyroot. Rawlogs /private/tmp/goai-prefill-ci-O70L2w/implement-*.log. AllCIjobsANDsteps mustpassbeforemerge separatePR. Returnsummaryincludinglimits andno productionfilesmodified.
