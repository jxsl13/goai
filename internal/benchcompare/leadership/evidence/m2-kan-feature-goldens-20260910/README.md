# Exact KAN CPU feature goldens

Test-only correction; no runtime performance claim. Task T-01M25NHXPAEYA
reuses `archgold.PickSIMD` for the existing F64 SiLU feature policy. All six
scalar constants, fixture geometry, seed, input formula, CPU binding, digest
arithmetic, and exact assertions remain unchanged. The existing SIMD CI step
also executes the applicable CPU SiLU accuracy tests without changing limits.

## Source and baseline

The source commit is `3530758fd01961064dad0d21366531d0964c2df3`, based on the
diagnostic branch at `81bfae535480e144495dc54fcf93ae4b66171745`.
Native Linux/Windows default and SIMD evidence, independent M2 causal
isolation, and Rosetta limitations are retained in the sibling
`m2-kan-simd-baseline-20260910` evidence directory. No new expected value was
derived from the candidate implementation or from Rosetta.

- Test SHA-256: `03b9565cc76737bafc500617ca0e392d0994c3c1f6dcae1269e445ac0518fe92`.
- CI SHA-256: `a2f436db4e5bb2358c72ee7245255b83ebd42646e1fdbb51c9359926cb5ada84`.
- Production and strict VJP oracle sources remain unchanged.

## Implementation capture limitations

`implementation-capture.json` retains 112 original text artifacts verbatim,
with byte counts and SHA-256. Root compared every decoded artifact with the
original bytes. The runner records numeric exits inside its manifests.

The initial complete VERIFY commands passed on the committed test hash above.
The implementer subsequently refined only the comment in its isolated tree;
that unadopted revision has test SHA-256
`f77c95cfd45d01a98e9ddba9e33d6be42b3c34e809db316bd8f72dde1b6baa1a`.
The `final-*` captures concern that later comment-only revision. The PR retains
the original comment, and independent verification targets the exact PR source.

The implementer runner omitted `GOEXPERIMENT` from its manifests. Focused KAN
stdout identifies the compiler experiment, but a full-suite artifact label alone
does not prove the build environment. Those original manifests are not rewritten
to invent missing provenance. Independent verification must record the explicit
experiment for every command and rerun the complete final-source VERIFY block.

## Qualification status

The fresh verifier reran the complete final-source VERIFY block successfully:
default/SIMD full short nn and focused KAN tests; the applicable ARM64 SiLU
accuracy, vector-tail, and edge tests; CGO-enabled nn; both KAN race modes;
the strict VJP race oracle; whole-tree vet; and Markdown/API checks. All three
KAN fixtures executed in both modes. Root separately rebuilt the exact source
with Go 1.27.1 using `go build ./...` in default and SIMD modes; both passed.

`independent-verifier-capture.json` retains 112 original artifacts, including
explicit experiment/CGO manifests and numeric exits. Entries `30-*` and `31-*`
are root's separate rebuilds, not verifier-authored commands. Root compared all
decoded artifact bytes and hashes with their originals.

The verifier initially saved the committed diff in `13-mutation-state.stdout`,
not the temporary working mutation diff. That original artifact is retained
but is not evidence of the temporary edit. The repaired evidence includes the
complete mutated source `21-mutation-source.go` (SHA-256
`cce577ff3f5d7a40d17107019c090c705f6e172d269a3c816ea9a15fb882eb52`)
and genuine working diff `22-mutation-working-tree.diff` (SHA-256
`1db1bcb084bb8ac707032041fec53c002ab635ef33612a8dd8e6bc0e2d64fc09`).
The only mutation is `math.Nextafter` of actual output element zero toward
positive infinity. Fresh runs `23` and `24` each fail all three digest assertions
in default and SIMD builds; restored runs `25` and `26` pass all three. Final
state is clean and matches the exact committed source hash. Root read the
complete retained mutation source, genuine diff, and raw assertion outputs.

Native CI run 34479783082 on the source commit passed all 16 jobs and all
175 executed steps, including the three soft SIMD jobs. Full status is in
`native-ci-source-final.json`; `native-simd-capture.json` retains complete native
Linux, Windows, and macOS job logs with IDs and byte/hash metadata. Root verified
the logs byte-for-byte and confirmed actual KAN and architecture-specific SiLU
test execution. ARM64 accuracy remains 3.048e-16 against the existing 1e-13 bound.
The final records/evidence push and parent PR still require exact-head CI before
merge. No performance candidate or benchmark result is included here.

`spec-check.json` retains the complete Spectackle check after explicit anchor
refresh: no new drift, with 136 inherited warnings and two inherited
record-only-context errors. It is not a globally clean specification claim.
The validate pack did not supply a source diff even with working edits; root
and the independent verifier use genuine Git diffs, not absent-pack evidence.
An explicit passing validate verdict attributed to `kan_feature_goldens_verify`
records the independent report; rendering a pack alone was not counted.
