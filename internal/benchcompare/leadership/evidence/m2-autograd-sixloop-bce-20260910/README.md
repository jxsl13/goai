# Six-loop VJP bounds-proof experiment

Status: qualified test baseline prepared; no candidate runtime edit or timing.

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
