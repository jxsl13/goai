# Apple ARM64 VJP bounds-isolation preflight

## Result and scope

**Eligible for a new implementation proposal only.** Under pinned Go 1.27.1,
a compile-time Darwin/ARM64 constant preserves the six intended M2 bounds-proof
hot loops and the original Linux/Windows AMD64 compiled implementations.
Independent native M2 correctness checks pass. This is compiler-design evidence,
not native AMD64 candidate validation, performance qualification, or a speedup.
No benchmarks ran and no runtime, test, module, or CI changes are promoted here.

Research `R-01M25XNPFEEP58GC32DVNHVB9Q` is consumed by
`VJP-ARCH-CODEGEN-001`: emitted functions and clones must bind to identical
object bytes and named relocation targets before compiler equivalence is claimed.
The prior six-loop and eight-loop candidates remain rejected. In particular,
the six-loop source changed sigmoid F64 NaN payloads in native AMD64 CI;
the new constant is not permission to weaken that oracle or omit sigmoid.

## Frozen inputs

Baseline main is `14fa6d015fa2482eb5926f03013b976aadda1a09`; isolated captures
start from the evidence-only branch commit
`3afc57b0e5db2cd6474e0352c437c89b5eadf380` with the same original runtime.

| Input | SHA-256 |
| --- | --- |
| ORIGINAL `autograd/vjp_elementwise.go` | `07adbf223e44d862d5f9e3e38a743eff54def329280bb8b76aa0ffe3f9b62945` |
| REJECTED six-loop runtime | `9969389cb316d0ba85d0121e7fcbcbc1ecbcc85b34b7b98fc15b2d0b80402d5e` |
| GUARDED experimental runtime | `4427ca8dd0ac65f8724d29382220412c7cbfa53fd170bd9909ba5e86ee4be325` |
| Experimental diff | `22aa62be37ebefb47e33c7157556f7bb2c85e16362bea1d072dc4e1c9fc441c9` |
| Frozen `autograd/vjp_bounds_internal_test.go` | `bd26d0195b02e23aa1406f01f7df25fff5aa65ea81fea0c3bae9f31d3359648c` |
| Complete unchanged tanh declaration | `10b6ee15d51252bff1ce32ddfd089284e77f0240aa985c04955032b8c05261b0` |

The only experimental source variation adds the standard `runtime` import and
`const darwinARM64VJPBounds = runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"`,
then prefixes exactly six existing length guards with the constant. Unary,
ReLU, and sigmoid each retain F32/F64 hot paths and their original cold loops.
Arithmetic, tanh, callbacks, APIs, registry, fallbacks, and tests are unchanged.
The archive contains the exact source captures and inert diff; it is not applied
to this PR's runtime.

## Verification

Host: Apple M2 Pro, 12 cores, 32 GiB, macOS 26.5.1. Every corrected command
selects Go 1.27.1, local toolchain resolution, direct module access, dedicated
cache/modcache, empty `GOFLAGS`, and an explicit default or SIMD experiment.
Full command/environment records and exit statuses are archived.

Both researcher and fresh independent verifier ran these on ORIGINAL, then
GUARDED, in separate worktrees:

- CGO0 default and SIMD: `go test -short -count=1 ./autograd ./backend/cpu ./nn`.
- CGO1 default: `go test -race -count=1 ./autograd -run '^TestUnaryVJPBoundsExact$'`.
- GUARDED additionally: CGO1 default `go test -short -count=1 ./autograd`.

All corrected commands pass. Each guarded source was applied only after the
corrected baseline gates passed. Failed helper attempts are preserved separately;
they are not counted as test passes.

| Compiler pairing | Default and SIMD result |
| --- | --- |
| REJECTED → GUARDED, Darwin ARM64 | All 1,608 actual instruction rows across six emitted functions/clones match after source-location stamps are removed. All six hot loops are retained; the constant disappears. |
| ORIGINAL → GUARDED, Windows AMD64 | Binary comparison passes; independent root symbolic/object binding also passes. |
| ORIGINAL → GUARDED, Linux AMD64 | Initially inconclusive binary references; all 36 differing PC-relative `LEAQ` rows per mode subsequently bind to identical named relocations. |

The M2 row count excludes headers/blanks (the full disassembly has 1,619 lines).
BCE diagnostics and actual hot/cold control-flow spans are recorded in
`independent/REPORT.md`; neither source-line-only diagnostics nor a missing
`panicBounds` string alone is the proof.

The root symbolic recovery audits all four AMD64 OS/experiment pairs. Each pair
has 14 identical compiler STEXT definitions after removing only instruction-row
source-location fields. All six emitted functions bind byte-for-byte outside
declared relocation fields; paired relocation offsets, targets, and addends,
plus 35 target-local data/metadata definitions, match. Windows' 110 trailing
INT3 padding bytes per capture are explicitly checked (Linux has none).
The linker coalesces seven byte-and-relocation-identical hashed unary clones
under one displayed symbol; all possible origins and paired origin sets match.
The unused, differently sized un-hashed definition is not substituted for them.

No unexplained data reference, numeric immediate, unknown instruction row, or
closure clone is silently discarded. Independent comparator fixtures check
branch relocation, ordinary immediates that resemble addresses, negative source
labels, and rejection of unknown rows.

## Evidence and reproduction

`evidence.tar.zst` is a content-addressed tar archive: `manifest.json` plus
439 unique `blobs/<sha256>` regular files. The manifest maps 893 logical artifact
names to their blob, exact byte count, and SHA-256. Repeated artifacts share
storage; logical decoded size is 546,691,828 bytes. All 893 artifacts were
decompressed and compared byte-for-byte with the frozen originals, including
tar checksums, entry completeness, padding, hashes, and decompressor exit status.
The archive is 5,845,848 bytes, compressed with zstd 1.5.7. No intermediate
uncompressed tar was created. Sixty-one compiled binaries remain outside Git;
their names, byte counts, and hashes are listed in `excluded_binaries`.

| Published artifact | SHA-256 |
| --- | --- |
| `evidence.tar.zst` | `8405e6319df2e4c1917b077b743b4cf1af9bff1c109dc8289ca2f37c0088f838` |
| `manifest.json` | `8b22a13fd877b03110e302ddd6340c5df9e38f459fa0cf04a2b1ad404afc2617` |

Run `ruby verify.rb` from this directory to stream-verify the published archive
without writing the decoded payload to disk. It requires Ruby and `zstd` on
PATH, but not the original worktrees. `verification.json` records the stronger
original-byte comparison; its original verifier and compression command are
published as provenance. Absolute temporary paths in captures are historical,
not expected to exist on another machine.

To read one logical artifact without extracting everything:

```sh
set -o pipefail
blob=$(ruby -rjson -e 'm=JSON.parse(File.binread("manifest.json")); a=m.fetch("artifacts").find { |x| x.fetch("name") == ARGV.fetch(0) }; abort "unknown artifact" unless a; puts a.fetch("blob")' independent/REPORT.md)
zstd -d -c --check evidence.tar.zst | tar -xOf - "$blob"
```

Key logical artifacts:

- `independent/REPORT.md` and `independent/SYMBOLIC-ANALYSIS.md`: independent
  native checks, M2 hot-loop spans, corrected all-row comparison and Linux
  relocation proof. Final verdict is proposal eligibility only.
- `root-source/source-proof.json`: exact full-source reconstruction,
  oracle/tanh preservation, and six-path source constraints.
- `root-symbolic/REPORT.md`, `root-symbolic/artifacts/symbolic-comparison.json`,
  and `root-symbolic/artifacts/symbolic-binding-v3.json`: complete recovery
  compiler/object proof, including clone provenance and padding.
- `research/guarded_experimental.diff`: inert candidate for a future task.
- `research/REPORT.md`: deliberately preserved **interim inconclusive** report;
  incomplete researcher symbolic metadata is not qualification evidence. The
  later independent and root symbolic captures resolve the comparison.

Initial environment/tool lookup, Ruby encoding/parser, shell-loop, comparator,
ambiguous clone-binding, and packaging-helper failures remain in the archive.
Early checksums/reports are interim snapshots; the published manifest is the
complete final inventory. The root recovery supplements independent verification;
it does not turn cross-compilation into native AMD64 execution.

## Required next gate

A separate implementation proposal/task/PR must add actual native strict SIMD
oracle execution to CI, then pass native ARM64/AMD64 default/SIMD/race checks and
fresh independent verification before timing. The existing soft SIMD CI jobs do
not execute this strict autograd oracle. Runtime/oracle/tanh constraints and all
six paths stay frozen.

The existing [paired benchmark protocol](../m2-autograd-bce-20260910/README.md)
is unchanged: three alternating seven-pair campaigns; target direct ReLU cells
at least 1.05x and taped ReLU cells at least 1.03x, each with `p < 0.05`; no
allocation increase, no reproducible control regression above 3%, and near-5%
spread for sub-10% claims. A positive compiler preflight bypasses none of these
gates. This evidence-only PR's CI validates the original runtime, not GUARDED.
