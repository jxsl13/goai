---
schema: v1
---

## T-01KYJR5XB6FB2SGHXSSER77MAG Drop the interface-dispatched RNG from Dropout's per-element mask loop
kind: task
state: draft
created: 2026-07-27

THE DEFECT WAS ESTABLISHED BY A CONTROL EXPERIMENT, not by inference — read that part before changing anything, because the obvious suspects are verified innocent.

MEASURED on this host: BenchmarkDropoutForward 10,432,895 ns/op, 12,583,603 B/op, 14 allocs. BenchmarkDropPathForward at the IDENTICAL shape 813,237 ns/op, 12,583,614 B/op, 15 allocs.

THE CONTROL: DropPath (nn/droppath.go:71-76) uses the identical tensor.New mask allocation, the identical full mask write, and the identical backend.Execute(OpMul) — differing ONLY in that it draws 16 random numbers instead of 1,572,864. So allocation, memclr, mask stores and the multiply together account for 813 us, and the remaining 9.62 ms (92%) is 1.57M RNG draws at 6.1 ns each — about 21 cycles for a PCG step that should cost about 2. The tensor allocation and mask materialization, which look like the obvious suspects, are VERIFIED NOT TO BE THE PROBLEM.

SITE: nn/dropout.go:72 and :79 — d.rng.Float64() inside the mask loop; d.rng is declared *rand.Rand at nn/dropout.go:33, built at :42.

WHY HOT: (*Dropout).Forward runs per dropout layer, per forward pass, per step — 2-3 times per transformer block. At the benchmark's 16x128x768 activation that is 10.4 ms per call, so a 12-layer block stack pays roughly 250 ms/step.

MECHANISM, confirmed by -gcflags='-m -m': Rand.Float64 and Rand.Uint64 both inline, but Rand.src is a rand.Source INTERFACE FIELD, so src.Uint64() is an indirect call that is neither inlined nor devirtualized. pprof corroborates: (*PCG).Uint64 0.80s flat, (*PCG).next 0.25s, and 1.74s of call-site stall attributed to the loop line — against internal/simd.MulF32 at 0.03s (0.9%).

FIX: store the concrete *rand.PCG (or an inlined 2-word PCG state) on the Dropout struct instead of *rand.Rand, so next() inlines. Then derive the Bernoulli from the raw draw with a precomputed integer threshold: thr := uint64(math.Ceil(d.Rate * (1<<53))), test u<<11>>11 >= thr. This is EXACTLY equivalent to float64(u<<11>>11)/(1<<53) >= d.Rate because u53/2^53 is exact in binary64 — the same comparison result for every draw. Optionally make the store branchless.

VALIDATION GATE (benchmark only): BenchmarkDropoutForward (nn/dropout_fastpath_test.go:143) isolates it. KEEP BenchmarkDropPathForward as the invariant control — it must not move; if it does, the change touched something it should not have.

EXPECTED: 10.43 ms -> about 2.5-3.5 ms (3-4x). High confidence — the control experiment bounds the non-RNG floor at 813 us and the remaining cost is a single known indirect call.

BIT-IDENTITY BAR: BIT-IDENTICAL AND RNG-SAFE. The draw count is unchanged (Rand.Float64 consumes exactly one Uint64), the PCG stream is unchanged, and the integer threshold reproduces the float comparison exactly. The mask, and therefore the output, is bit-for-bit the same. Because this class of change is RNG-adjacent, the existing seeded-determinism tests must be run and named explicitly in the commit rather than assumed to pass.

PERFSCAN RULE REQUIRED, and it has wide reach here: interface-sourced RNG in a per-element loop. AST shape: a SelectorExpr call X.Float64() / .Uint64() / .IntN() / .NormFloat64() where X resolves to *math/rand.Rand or *math/rand/v2.Rand, inside a loop whose bound is a slice length or Numel(). Recommend the concrete source type. THIS IS NOT CONFINED TO DROPOUT: 12 non-test files in nn hold a *rand.Rand field with 32 per-element draw sites — neftune.go, mixup.go, cutmix.go, rso.go, droppath.go, psgd.go, apollo.go, qgalore.go, aqlm.go among them. Run the finished detector and report every site.

## P-01M25KRMYZEH599AP2131M1GD6 Qualify KAN exact output across scalar and SIMD CPU lanes
kind: proposal
state: active
created: 2026-09-10
refs: R-01M25JEGEJFDEVBR7B0YZKQ3MX
grilled: 2026-09-10 open=0
targets: go:nn.TestKANForwardIsBitIdentical

GOAL: make the existing KAN whole-layer frozen-output guard correct for the already-supported CPU scalar and SIMD build-feature lanes without changing production arithmetic or tolerances. Research R-01M25JEGEJFDE reproduces an inherited arm64 SIMD mismatch and isolates SiLU; its exact source-artifact audit is still pending and is a hard prerequisite to assigning final golden values. No new runtime speedup is claimed.

SCOPE AND PHASES: nn/kan_bitidentity_test.go is the sole test source edit; .github/workflows/ci.yml may add an explicit existing TestKANForwardIsBitIdentical execution to each existing SIMD matrix job. The current SIMD CI step builds all packages but tests only internal/simd, so successful CI does not establish nn SIMD correctness. Keep all existing CI commands and failure behavior; no skip, suppression, softening, or broad workflow redesign. First, expose all three current fixtures as independent subtests and bind the registered CPU explicitly, preserving every existing golden and input/parameter formula. Native Linux/Windows default and SIMD CI then supplies actual per-fixture baseline results on pinned Go1.27.1. This is an expected-failing diagnostic draft stage, not a mergeable result. A separately drafted implementation task with exact audited constants may then use archgold.PickSIMD. No constants may be copied from a failed local run without cause/provenance review.

KNOWN EVIDENCE: arm64 default goldens remain 5936029728971432568,15159748691548848689,515177776064738749. Provisional arm64 SIMD measurements are 17265271475585544907,5035091549113534389,12048638696559957597. Inputs/weights/basis/spline are unchanged; a scalar-baseline-only SiLU hybrid restores the three historical arm64 finals while MatMul and Add stay CPU. The reference backend formula is not baseline-bit-identical and must not be called the scalar baseline. Current research patch artifact was found empty during root review; do not claim source verification complete until repaired/reproduced. Rosetta AMD64 default itself disagrees with two stored goldens, so its values cannot establish native AMD64 goldens. Native baseline divergence beyond SiLU requires bounded follow-up diagnosis, not silent replacement.

REUSED APIS: backend.Get(backend.CPU) and backend.NewContext().WithBackend(cpuBE) (already used in autograd/wkv_test.go); archgold.PickSIMD already exists in internal/archgold/archgold.go. Preserve TestKANForwardIsBitIdentical, all three 3/5/7,13/8/6,96/24/32 geometries, seed1, math.Sin inputs, F64 storage and FNV64a little-endian bit digest. No change to NewKAN/KANLayer.Forward/fusedSpline/fusedSplineBand, CPU activation/MatMul, archgold helper implementation, or any VJP production/oracle source.

VALIDATION: pinned Go1.27.1; default and GOEXPERIMENT=simd under CGO0. The diagnostic stage must retain all three fixture outcomes, including known failures, with full commands/stdout/stderr/exits. Final qualification requires full default/SIMD nn short suites, existing M2 SiLU accuracy/vector-tail/edge tests, CGO1 default nn short tests and focused race KAN coverage, unchanged runtime hashes, whole-tree vet/format/doc/API checks, and independent fresh verification including a non-vacuous one-ULP output mutation that the exact digest detects. The existing ARM64 TestVsiluF64Arm64Accuracy gate is <=1e-13, not a tolerance for KAN; root rerun measured3.048e-16 over262145values with vector-tail and edge tests passing. Native AMD64 default/SIMD CI evidence is mandatory. Real CI jobs and executed steps must all pass on the final exact head before merge; soft SIMD job status alone is insufficient.

DELIVERY: extend existing separate diagnostic draft PR1256 with proper staged specifications/evidence. Do not merge the diagnostic red stage. Record baseline scope/compiler/feature provenance and source hashes, consume R only after its actual evidence is independently audited, preserve any rejected alternatives in the journal, and report generalizable feature-partition validation lessons on the existing perfscan issue795 if the new evidence materially adds to it. Six-loop VJP PR1255 remains untouched until a fully qualified corrected test baseline exists.

## T-01M25KTC4EEXKBXZPXCWYEFQAJ Expose all KAN exact fixtures in native SIMD CI
kind: task
state: active
created: 2026-09-10
refs: P-01M25KRMYZEH599AP2131M1GD6, R-01M25JEGEJFDEVBR7B0YZKQ3MX
grilled: 2026-09-10 open=0
targets: go:nn.TestKANForwardIsBitIdentical

GOAL: expose every existing KAN exact-output fixture in native CI without changing any expected digest or production operation. This is the diagnostic phase of P-01M25KRMYZEH5; expected inherited failures are evidence, not permission to mark the final correction qualified.

EXACT LEASE/EDIT SCOPE: nn/kan_bitidentity_test.go and .github/workflows/ci.yml only. No production, archgold implementation, VJP oracle, benchmark, dependencies, other tests, workflow triggers/jobs/permissions, or golden constants. Root owns all Git and Spectackle lifecycle mutations.

KNOWN TEST: nn/kan_bitidentity_test.go TestKANForwardIsBitIdentical contains cases {3,5,7,archgold.Pick(5936029728971432568,14272068029666688409)}, {13,8,6,archgold.Pick(15159748691548848689,6609257596807823200)}, {96,24,32,archgold.Pick(515177776064738749,9025949438388873583)}. It creates NewKAN(c.in,c.out,1), x F64 with math.Sin(float64(i*11+5))*0.6, Forward(NewContext(),x), then FNV64a seed14695981039346656037,mult1099511628211 over Float64bits in little-endian bytes. Preserve all formulas, order, shapes and constants. Existing top-level loop uses t.Fatalf, preventing later fixtures from appearing after the first failure.

TEST CHANGE: add a name string to the table with names 3x5x7,13x8x6,96x24x32 and wrap each complete case body in t.Run(c.name,func(t *testing.T){...}); no t.Parallel. Retrieve cpuBE,ok := backend.Get(backend.CPU) once; t.Fatal if not registered; ctx := backend.NewContext().WithBackend(cpuBE); call l.Forward(ctx,x). APIs are already used in autograd/wkv_test.go. No new CPU import or backend substitution should be needed; use the registry, fail if unavailable. Add a short t.Logf with runtime.Version(),runtime.GOOS,runtime.GOARCH and explicit backend=cpu (runtime import needed); experiment identity comes from CI command/env. Existing t.Fatalf digest text stays unchanged within the subtest. No tolerance, skip, new golden or PickSIMD in this phase. Comments explain that independent subtests retain all fixture failures for feature/architecture qualification. Keep the existing batch/row-order rationale.

CI CHANGE: .github/workflows/ci.yml existing simd job matrix is ubuntu-latest,windows-latest,macos-latest, with pinned setup-go1.27.1 and CGO0 in the compile/test step. Its script currently runs GOEXPERIMENT=simd go build -tags=simd ./... then GOEXPERIMENT=simd go test -short -count=1 ./internal/simd. Keep both unchanged and append exactly GOEXPERIMENT=simd go test -short -count=1 -v -run '^TestKANForwardIsBitIdentical$' ./nn. No suppression, masking, ||true, altered continue-on-error policy, reordered gates or additional matrix. Job-level soft status does not waive executed-step failure; root reads real conclusions.

VERIFY ENV: PATH=/private/tmp/goai-go1271-j1cLvq/go/bin:$PATH GOCACHE=/private/tmp/gocache-goai GOMODCACHE=/private/tmp/goai-go1271-j1cLvq/modcache GOTOOLCHAIN=local CGO_ENABLED=0; default GOEXPERIMENT='' and simd GOEXPERIMENT=simd. Use pinned prefix for Spectackle too. Root must grant the heavy-Go window before any Go command. Capture direct complete stdout/stderr and numeric exit files for each command. Scripts/source must use apply_patch, no shell source writes.

VERIFY: before edits record git status, exact source SHA256s for nn/kan.go,backend/cpu/elementwise.go,backend/cpu/vexp_arm64.go,backend/cpu/vexp_amd64.go,autograd/vjp_elementwise.go,autograd/vjp_bounds_internal_test.go,internal/archgold/archgold.go. Read only the two edit files plus the cited API example. After edits gofmt only nn/kan_bitidentity_test.go; run default go test ./nn -run '^TestKANForwardIsBitIdentical$' -count=1 -v (must pass all three), SIMD same command (must execute all three, retain exact inherited failures), default go test ./nn -short -count=1 and go vet ./nn. Do not change expectations if these fail. Assert all six golden decimal constants unchanged and runtime hashes unchanged, git diff --check clean; inspect workflow diff for only the appended command. Report exact raw logs and source diff, not an abbreviated report labeled raw. No benchmarks, golden regeneration, or performance claims.

PROTOCOL: fresh isolated worktree; pull task and parent, claim exact two-file lease TTL7200, edit via apply_patch. No commits/push/PR/lifecycle/raw .spectackle reads or writes. Retain all reports/logs outside repo, complete genuine git diff for tracked edits. Release lease and Go window. Root independently verifies preservation, commits/pushes the diagnostic stage on draft PR1256, audits native CI outputs, and separately drafts any golden-correction task. Never say final correction is done or mergeable based on expected diagnostic failures.
