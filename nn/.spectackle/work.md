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

## R-01M25JEGEJFDEVBR7B0YZKQ3MX Diagnose inherited ARM64 SIMD KAN whole-layer digest mismatch
kind: research
state: active
created: 2026-09-10
grilled: 2026-09-10 open=0
targets: go:nn.TestKANForwardIsBitIdentical, go:nn.KANLayer.Forward, go:archgold.PickSIMD

Goal: diagnose the inherited ARM64 GOEXPERIMENT=simd failure of nn/TestKANForwardIsBitIdentical before the separately approved six-loop unary VJP performance task can run a valid baseline. Research only; no production implementation, tolerance widening, golden regeneration, or benchmark timing is authorized.

CONTEXT: Pristine default main is c6afe9e4ba8b2a46c254953c921cf97ddf4f72c6. Foundation 6322ae6ac7afae1b4af33f5a16f947258437d2f6 has the same tree as independently qualified 0d7c62fe48cd526f02738df41dca6cd1f532c2bb; changes are autograd tests, specifications, and evidence only. Six-loop draft head 5be9a49c83b8dc078040ccc0dd2f4a54dd3a1a97 changes only specifications from 0d7. No nn, backend/cpu, internal/simd, or internal/archgold source differs between main and these heads. A pre-edit default CGO=0 short test of autograd/backend/cpu/nn passed. SIMD failed only at KAN 3x5x7: got 17265271475585544907, expected 5936029728971432568. The original log exists but a shell reserved-variable mistake prevented a separate exit artifact; reproduce honestly instead of inferring a recorded exit.

KNOWN FILES AND CONTRACTS: nn/kan_bitidentity_test.go freezes whole-layer F64 output for B/in/out 3/5/7,13/8/6,96/24/32 using archgold.Pick, FNV64a little-endian Float64bits, deterministic seed 1, and math.Sin input. nn/kan.go KANLayer.Forward runs OpSiLU, OpMatMul to yBase, buildBasis, then fusedSpline and OpAdd for inference. fusedSpline bands rows via parallelRows and fusedSplineBand; c and i accumulation order and explicit float64 rounding barriers must remain unchanged. internal/archgold/archgold.go has both Pick (architecture only) and PickSIMD (architecture plus experiment); documentation describes intentional transcendental approximation differences. This is a hypothesis, not proof of SiLU responsibility or authority to accept new goldens. Applicable FP-GOLDENS-PER-ARCH-001, AMD64-FP-GOLDENS-COME-FROM-CI-001, and BAND-A-NEST-THAT-WRITES-THROUGH-A-DERIVED-BASE-001 retain exact digest and race requirements.

METHOD: Use Spectackle research/find/get for narrowly scoped missing KAN/CPU SiLU/MatMul/reference/archgold symbols and contracts, then read only returned known paths. First capture clean statuses, source hashes, Go version/env. Run unchanged targeted TestKANForwardIsBitIdentical default and SIMD with -count=1 -v on both pristine c6 and foundation. If inherited mismatch is confirmed, a temporary diagnostic test at nn/kan_simd_baseline_diagnostic_test.go may be added via apply_patch in the isolated research worktree only, under a lease. It may report each fixture's input/parameter/stage/output bit digests, compare CPU and reference paths or operation-specific reference substitutions, and compare serial/full-row fusedSplineBand against banded output at identical basis/parameters. Reuse established backend/reference APIs located through Spectackle; no production edits. Exercise all three fixtures, not merely first fatal mismatch. A hybrid must transparently preserve every non-substituted operation and record dispatch; do not switch wholesale to reference and claim CPU coverage. Diagnose to the narrowest evidence-supported level; compiler/FMA attribution requires disassembly and is optional, not guessed. Restore temporary diagnostic after retaining its exact patch and complete raw outputs.

VERIFY AND ARTIFACTS: Pinned Go1.27.1 executable /private/tmp/goai-go1271-j1cLvq/go/bin/go; prefix PATH=/private/tmp/goai-go1271-j1cLvq/go/bin:$PATH GOCACHE=/private/tmp/gocache-goai GOMODCACHE=/private/tmp/goai-go1271-j1cLvq/modcache GOTOOLCHAIN=local CGO_ENABLED=0. GOEXPERIMENT='' for default, GOEXPERIMENT=simd for SIMD. Exact initial command go test ./nn -run '^TestKANForwardIsBitIdentical$' -count=1 -v. Record direct stdout+stderr and separate numeric exits, including expected failures; no hand-abbreviated output labeled raw. No Go commands overlap other workers; request/release the M2 heavy-Go window. Stop before expensive race/full-suite commands unless needed to discriminate the hypothesis. Retain commands, relevant environment, revisions, hashes, diagnostic patch, and exact raw logs under scratch artifacts outside tracked files. Report a bounded recommended corrective task and alternatives, scope and unresolved evidence, not an implementation. Root independently reviews evidence and owns all lifecycle/Git/PR actions.

BOUNDARIES: Fresh cheaper research agent in its own detached worktree. No commits, push, PR, lifecycle move, raw .spectackle reads/writes, runtime edits, official golden edits, policy skips/tolerances, dependency changes, or performance claims. Claim/release only the exact temporary diagnostic path if used. Scope limited to KAN baseline and directly called backend operations; request a brief amendment before materially broader exploration. Research ends consumed by a separately grilled rule/task or explicit no-action closure.
