# ARM64 SIMD scalar-parity test correction

Status: test-only implementation passes root and fresh independent verification.
Final exact-head CI remains the merge gate. No runtime change, timing measurement,
or speedup claim.

## Baseline and scope

Base: `4ad981bfe801f22e8b057d2682a93bc51d012048`.
Implementation: `c3b4d1f1`, stacked PR1254 on the strict-oracle/evidence PR1253.
Spectackle: P-01M24BYJB6EEP / T-01M24C1S91FWX, consuming archived
research R-01M249HFAVFQR.

The original full SIMD autograd suite failed F64 focal/composite and WKV/host
comparisons on pristine main as well as the test-foundation branch. The archived
baseline diagnostic measured every fixture element against bounds declared
before measurement. Maximum scaled errors were 2.7755575615628914e-17 for
focal forward, 5.082197683525802e-21 for focal VJP, and
1.7347435098673699e-16 for WKV. See the sibling
`m2-autograd-bce-20260910/simd-baseline-diagnostic-report.txt` and raw evidence.

Only five test files change. F64 focal active-composite comparisons on
ARM64 plus `goexperiment.simd` use
`abs(got-want)/max(1,abs(want)) <= 1e-13`. WKV CPU/scalar comparisons on
the same build use `abs(got-want)/max(1e-6,abs(want)) <= 1e-10`.
These match existing approximate-leaf/scan contracts, not newly measured budgets.

The comparator accepts identical raw bits first, then rejects nonfinite
mismatches and opposite zero signs before applying the finite bound. Tests
retain actual CPU execution and check tensor dtype, shape, count and dense
storage. Focal target gradients remain detached.

Default and non-ARM64 F64 focal paths still use `requireGradBits`; the existing
F32 policy is unchanged. Default/non-ARM64 WKV retains its historical exact
numeric comparison. Separate fused CPU/reference focal and same-SIMD WKV
state/chunk/range exact tests are unchanged.

## Root verification

Pinned official Go1.27.1, darwin/arm64 Apple M2 Pro. Commands, combined raw
outputs and process exits are retained in `root-gates.txt`. Go work was
serialized. The initially delegated implementation worker exhausted its quota
before editing files; root implemented the declared five-file scope locally.
Independent verification was performed in a fresh isolated worktree; it is not
inferred from the root tests.

Passing commands include full short autograd under default CGO0, SIMD CGO0 and
default CGO1; focused default CGO1 race including the historical unary oracle;
SIMD Softplus numerical/edge/vector-tail and fused focal CPU/reference checks;
all tests matching TestWKV in internal/simd; whole default go vet; mdlint/apicheck.

Three temporary, exact-anchor mutations compiled and failed their intended
assertions:

- Always-true comparison helper: special-value and above-bound unit cases fail.
- Corrupt the real F64 focal loss by +1: both F64 attribute cases fail at the
  actual focal comparator.
- Corrupt the real WKV output by +1: actual CPU/host comparison fails at [0,0].

Each patch is retained in `root-mutant-*.txt`, with its failing output in the
combined gate log. All three were restored; full SIMD autograd then passed.
These are test-sensitivity checks, not performance measurements.

Frozen unchanged source SHA-256:

- `autograd/vjp_elementwise.go`:
  `07adbf223e44d862d5f9e3e38a743eff54def329280bb8b76aa0ffe3f9b62945`
- `autograd/vjp_bounds_internal_test.go`:
  `bd26d0195b02e23aa1406f01f7df25fff5aa65ea81fea0c3bae9f31d3359648c`

The independently rejected eight-loop BCE candidate remains inert evidence,
not runtime. In particular, this test policy does not permit NaN normalization
or tolerance in the historical unary VJP oracle.

## Independent verification and final local gates

The first independent report summarized some failing output. It is retained as
`independent-initial-summary.txt`, explicitly not raw or complete evidence.
The verifier then reran every declared command and all three compiling mutation
probes with direct stdout/stderr capture and separate process-exit files.
All normal and restored-suite exits are 0; the three mutations exit 1 at the
intended assertions. The independent corruption is +2 at each actual comparator,
separate from the root's +1 tensor-output corruptions. The restored tree is clean.

`independent-capture.json` preserves 30 complete UTF-8 artifact values: command
manifest, raw outputs, exact exits and captured unified mutation diffs. Decode a
value without adding a newline (for example with `jq -j`) to recover its bytes.
The scratch directory's stale preparatory README is not execution evidence and
is deliberately excluded. Source/runtime/oracle bytes remain unchanged.

The canonical `make perfscan-check` passes with direct fetching of pinned
github.com/jxsl13/perfscan@v1.81.0: 53 compatibility checks, zero findings in its
focused scan and passing fixtures. This is not a clean whole-tree scanner claim.
The first attempt failed with sandbox DNS denial and a misleading registry-diff
message; the complete failure is retained separately from the successful retry.

Pinned reindex reports 6715 typed call edges with zero skipped packages. The
Spectackle check retains 136 inherited warnings and two inherited record-only
context errors; it heals four intentional policy anchors. Explicit rule refresh
also updates their line spans. There is no new source/type error. Whole tracked
Go formatting is clean; the final markdown/API tests pass. The relevant commands
and outputs are in the numbered evidence files.

Final exact-head CI still must be audited. CI's soft SIMD lane builds the tree
and tests internal/simd,
not autograd, so its success cannot replace the local full SIMD autograd gate.
Merge commit only after qualification; delete exact remote feature branches
only after the result reaches default main.
