# Six-loop VJP bounds-proof experiment

Status: six-loop runtime candidate implemented; independent verification pending.
No timing or performance qualification yet.

Proposal P-01M25H86Z0FZR and task T-01M25HANWFF8F retain the original exactness,
compiler-proof, and performance gates. The entire tanhVJP declaration and
strict VJP oracle remain byte-identical. The earlier eight-loop candidate is
still rejected; its evidence and the audited runner remain in the sibling
`m2-autograd-bce-20260910` directory.

## Corrected baseline amendment

The frozen task's initial 0d7c62fe source exposed an inherited KAN SIMD digest
mismatch before runtime editing. That mismatch is separately diagnosed and
corrected by PRs #1256 and #1257. The correction is test/CI-only and selects
exact architecture/feature goldens from native evidence, without changing any
production operation, scalar golden, strict VJP oracle, or accuracy limit.

Rule SIXLOOP-QUALIFIED-BASELINE-001 requires identical corrected test sources
and pinned Go 1.27.1 in both measurement arms. The prepared baseline is
`e14a704ec6fa8befb19fdb5cdd2603875b772408`, whose source-qualified correction
is merge-committed into the parent as
`5b121d068a36dd9dfd63e2262e75f026643e362a`. Its final child CI passed all
16 jobs and 175 executed steps. Parent main-target CI 34483003557 also passed
all 16 jobs and 175 executed steps, and PR #1256 merged into main as
`814876f179be22ec71144b2124e47ea5df052390`. The prepared baseline and current
branch have identical Go/module/CI sources. Candidate runtime editing can now
resume under the unchanged task and its baseline amendment.

Root reran default and SIMD short suites for autograd, backend/cpu, and nn,
plus the default CGO-enabled strict VJP race oracle. All passed. Four normal-
flag baseline test binaries (autograd/nn, default/SIMD) were rebuilt, with
embedded build settings and hashes captured. GOFLAGS is empty; the binaries
identify Go 1.27.1, darwin/arm64, GOARM64=v8.0, CGO=0, and the intended
experiment. Building binaries is not running benchmarks or proving leverage.

`qualified-baseline-capture.json` retains the complete original command outputs,
exits, explicit experiment/CGO manifests, build settings, source provenance,
and binary hashes. Compiled binaries are not committed. Root verified every
decoded artifact byte and hash against the originals. Before timing, verify
the eventual candidate has exactly matching test/compiler sources and settings.

The branch update had one conflict, in derived Spectackle anchors only.
The incoming KAN anchors were retained through the Git merge, and the unchanged
six-loop tanh anchor was regenerated through Spectackle. No runtime file
conflicted. Both task/proposal records and all rules remain present; no
work.md union merge or manual specification-file rewrite was used.

The post-update full Spectackle check retains 136 inherited warnings and two
inherited record-only-context errors, with no new drift or orphan bindings.
The re-grill also reports a historical waiver-rate advisory; this amendment
adds no waiver and does not relax any correctness or performance requirement.

## Candidate implementation checkpoint

Source commit `33ca653b` implements the six guarded paths for
T-01M25HANWFF8F. Its runtime SHA-256 is
`9969389cb316d0ba85d0121e7fcbcbc1ecbcc85b34b7b98fc15b2d0b80402d5e`.
Only the F32/F64 loops in unaryVJP, reluVJP and sigmoidVJP change. The original
indexed loops remain as cold fallbacks, and all per-element expressions are
unchanged. No tests, arithmetic policy, tanh code, allocations or APIs change.

The fresh implementer ran the required before/after default and SIMD short
suites for autograd, backend/cpu and nn, the before/after CGO-enabled strict
race oracle, and the candidate CGO-enabled autograd short suite. All pass.
The strict oracle hash remains `bd26d0195b02e23aa1406f01f7df25fff5aa65ea81fea0c3bae9f31d3359648c`;
the entire tanh declaration is byte-identical to immutable baseline 54cc2964.
The root static review reconstructs the original file from all six cold loops
and confirms byte equality outside the replacements, plus exact equality of
the hot and cold per-element bodies. This is source evidence, not timing.

`implementation-capture.json` retains all 135 nonbinary artifacts (549,152
decoded bytes) as lossless gzip/base64, with byte counts and SHA-256 hashes.
It includes complete command manifests, stdout/stderr, numeric exits, source,
working diff, compiler diagnostics and disassembly. Root decoded every artifact
and compared it against the original bytes. Compiled BCE binaries are excluded;
their hashes and the actual compiler executable hash are recorded separately
in the capture. These diagnostic binaries are not benchmark binaries.

The implementer's helper finds no BCE diagnostics or panic calls attributed to
the six hot source spans, and retains cold fallback checks. A fresh independent
verifier is repeating correctness and examining actual assembly control flow;
line attribution alone is not the final proof. No benchmark is authorized until
that verification passes. The unchanged audited runner and full frozen target,
control, allocation and noise gates remain binding.

Capture caveats: an initial lease-schema query returned validation exit 1 before
the successful claim; this is not a test failure. The implementation environment
records its inherited default GOPROXY setting; independent verification explicitly
sets GOPROXY=direct. No dependency download is used as qualification evidence.
The root artifact comparator's initial Ruby string-encoding mismatch and corrected
byte-level check are disclosed in `root-static-check.json`; no source or captured
artifact was modified to obtain the pass.

Spectackle auto-healed seven unchanged-contract hashes; explicit rule relinking
then refreshed the moved source spans without changing any requirement. The full
check still reports only the 136 inherited warnings and two context errors, with
no remaining drift or orphan bindings. Main post-merge CI 34484557957 also passed
all 16 jobs and 175 executed steps. Neither CI nor compiler proof establishes a
performance gain, and this PR remains a draft.
