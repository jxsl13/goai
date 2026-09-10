# KAN SIMD baseline diagnosis (M2, Go 1.27.1)

Research only. No production operation, official golden, or tolerance was changed.
No performance measurement, speedup, or leadership claim is made.

## Result and limits

The inherited `nn/TestKANForwardIsBitIdentical` failure reproduces on clean
pristine `c6afe9e4ba8b2a46c254953c921cf97ddf4f72c6` and the qualified
test/evidence foundation. Default arm64 passes; SIMD fails before any proposed
six-loop VJP runtime edit.

A fresh researcher used stage-level instrumentation. Root then read the exact
retained final source and independently reran it in a separate worktree with
default and SIMD builds. All three fixtures reproduce the following digests:

| Geometry (batch/input/output) | Historical arm64 default | Active arm64 SIMD |
| --- | --- | --- |
| 3/5/7 | 5936029728971432568 | 17265271475585544907 |
| 13/8/6 | 15159748691548848689 | 5035091549113534389 |
| 96/24/32 | 515177776064738749 | 12048638696559957597 |

Across builds, input, weight, coefficient, basis, and spline-stage digests match.
Within each build, fused, generic two-Einsum, and serial spline routes yield the
same output digest. Replacing only SiLU with the historical CPU scalar formula
`x/(1+math.Exp(-x))` restores every historical final digest while MatMul and Add
remain CPU-dispatched. The reference backend SiLU formula is not bit-identical
to that scalar baseline and is reported separately, not used as its synonym.

Source attribution: `backend/cpu/elementwise.go` selects `vsiluF64` through
`vsiluF64Fast`. The arm64 SIMD partition in `vexp_arm64.go` enables it;
`vexp_default.go` retains the scalar formula. This is an existing feature policy,
not an optimization introduced by the current performance candidate.

The applicable M2 accuracy tests are
`TestVsiluF64Arm64Accuracy`, `TestVsiluF64Arm64VectorTailBitIdentity`, and
`TestVsiluF64Arm64Edges` in `backend/cpu/vsilu_f64_arm64_test.go`.
Root ran these with pinned Go 1.27.1, CGO=0, GOEXPERIMENT=simd: all passed;
the accuracy test reported 3.048e-16 against its existing 1e-13 bound over
262145 values. This is not a proposed tolerance for the KAN digest.
The similarly named `TestVsiluF64Accuracy` is AMD64-only.

## AMD64 remains unqualified

The pinned Go executable is arm64-only; attempting to execute it directly as
x86_64 failed and is retained. Cross-compiling AMD64 test binaries with the
pinned SDK and running them under Rosetta succeeded, but the unchanged default
run disagrees with two stored AMD64 goldens. The SIMD run also differs.

These local AMD64 values must not be copied into official goldens. Native
Linux/Windows default and SIMD evidence, and any necessary causal follow-up,
remain required. `archgold.PickSIMD` already supports four exact lanes; this
research does not authorize filling its arguments from unexplained failures.

Inspection of `.github/workflows/ci.yml` revealed that the existing SIMD lane
builds all packages but executes only `internal/simd`. A green SIMD job therefore
did not establish `nn` SIMD correctness. The separately specified diagnostic
task exposes all three KAN fixtures independently and adds their execution to
the existing SIMD matrix, preserving every current golden during data collection.

## Evidence and reproduction

`research-capture.json` is a map of 47 external research text artifacts. Each
entry contains its verbatim UTF-8 content, byte count, and SHA-256. It includes
complete direct raw outputs, numeric exits, the command/environment/source-version
manifest, final diagnostic source, genuine add-file patch, and provenance caveats.
Compiled AMD64 binaries are not vendored; their exact compile commands are in
the manifest.

Decode a source artifact without adding a newline, for example:

```sh
jq -jr '.["kan_simd_baseline_diagnostic_test.go"].content' research-capture.json
```

The retained final source SHA-256 is
`284e0869f3c96c6188e7a1ab47ddc6156f00718bf28d57092631cb24c76f2fa9`.

`root-qualification.json` retains root's independent final-source default and
SIMD command outputs and numerical exits, source/runtime hashes, and explicit
three-fixture digest comparisons. Each original Go command returned exit 0;
the diagnostic itself logs stage values, while root's separately recorded
comparison asserts the stated equalities and restoration. Its final temporary
source was removed with apply_patch; the isolated worktree is clean.

Both roots use:

```sh
PATH=/private/tmp/goai-go1271-j1cLvq/go/bin:$PATH
GOCACHE=/private/tmp/gocache-goai
GOMODCACHE=/private/tmp/goai-go1271-j1cLvq/modcache
GOTOOLCHAIN=local
CGO_ENABLED=0
```

The command is `go test ./nn -run '^TestKANSIMDBaselineDiagnostic$' -count=1 -v`,
once with `GOEXPERIMENT=''` and once with `GOEXPERIMENT=simd`.
Restore the retained source only in an isolated research worktree for reproduction.

## Capture corrections

The first researcher patch file is empty: ordinary `git diff` omitted the
untracked diagnostic. It is explicitly invalid, not source evidence.
The original `git diff --check` likewise did not validate that untracked file.
A genuine nonempty no-index add-file patch and final source were subsequently
recovered from the recorded patch content, then root independently reproduced
the result from that retained source. Earlier intermediate diagnostic logs have
unretained source revisions; they are labeled accordingly and are not needed
for root's final-source causal reproduction.

The initial six-loop worker's separate baseline failure had no numerical exit
artifact because its shell wrapper used a reserved variable. This research
reran the unchanged baseline with proper direct logs and numeric exits; no
missing original exit was invented.

## Spec and delivery state

- R-01M25JEGEJFDE: inherited KAN diagnosis, consumed by the corrective proposal.
- P-01M25KRMYZEH5: qualify exact scalar/SIMD CPU lanes.
- T-01M25KTC4EEXK: all-fixture native CI diagnostic phase only.
- PR #1256 remains a draft until the final corrected baseline is qualified.
- Six-loop performance PR #1255 remains unchanged and unmeasured.

No diagnostic failure is waived as a successful correctness or merge gate.
