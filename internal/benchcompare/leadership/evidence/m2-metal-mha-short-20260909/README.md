# MHA decode short-mode CI repair — September 9, 2026

Status: implementation and fresh independent verification passed; CI publication
pending. Test-only CI
reliability correction, not a kernel speedup or external leadership claim.

Spectackle proposal `P-01M23TTRCAFA4`, task `T-01M23TWGZ5F2E`, and
`METAL-MHA-DECODE-SHORT-001-001` apply the existing root contract
`TIMING-ASSERTIONS-SKIP-ON-RUNNERS-001` to `TestMHADecodeCost`.

## Observed failure

PR #1250's pre-merge run 34393416718 at a222b045 passed all 16 jobs and
every executed step, including Metal and soft SIMD. Its post-merge run
[34394770011](https://github.com/jxsl13/goai/actions/runs/34394770011) at
`fe7a9bfcc1b4f6b589aa65bf3d61bbe2998a87ae` failed a different, unchanged
test: `TestMHADecodeCost` measured llama7b dk=128 sk=512 at 817.7 us/op,
above its 600 us ceiling, under shared-host `go test -short`. The prefill
test fixed in #1250 did not fail. The exact relevant CI excerpt is retained
in `ci-failure-excerpt.txt`; it is not represented as the entire job log.

## Required behavior

Short mode executes one actual recorded MHA operation for each of four
model/context cases, retains recorder/MHA error checks, Commit, Wait, Free,
and buffer lifetimes, and logs each completed smoke case. It performs no
slope comparison or timing ceiling assertion. The unavailable-Metal skip
remains; it is not proof of execution on a GPU.

Full mode retains 25 repetitions, measurements at 32 and 256 operations,
the original slope calculation, all four geometries, and the unchanged
600 us ceiling. No production code, dispatch rule, shader, or workflow changes.
This operation smoke is not a new numerical-parity assertion.

## Verification plan

Pinned Go 1.27.1, Darwin ARM64, CGO enabled. Fresh implementer runs actual
Metal focused short/full tests in default and SIMD modes, the full short
Metal package in both modes, vet in both modes, formatting and whitespace
checks. A fresh independent verifier repeats focused real-Metal short/full
tests and audits the diff. Raw stdout/stderr, outcomes, source hash, and
the original failure remain available. No skips are counted as execution.

Publication requires all CI jobs and individual executed steps successful
for the exact final PR head. The remote feature branch is deleted only
after verified merge. Generalizable follow-up belongs on existing
[perfscan #868](https://github.com/jxsl13/perfscan/issues/868), not a duplicate
issue or an unverified claim about current detector coverage.

## Implementation verification

All ten bounded verification commands exited 0. Focused short default/SIMD
runs each executed four smoke cases with no skips. Full default timings, in
tinyllama sk36/sk512 then llama7b sk36/sk512 order: 13.22, 16.24, 66.84,
278.52 us/op. SIMD: 12.71, 15.38, 63.97, 278.36 us/op. These verify the
unchanged local ceiling; they are not a before/after speedup estimate.

Full short Metal package: default 40.501s, SIMD 40.572s, both successful.
Vet default/SIMD, gofmt, and diff checks passed. `implementation.txt` retains
command descriptions and literal outputs, including empty outputs.
Test source SHA-256:
`5bd34b91fe3e64e9dafa9c43037c247f84c5eeb4a7575ecb30f8f572a040337c`.

Fresh independent verifier: four real-Metal focused runs passed without skips,
with four smoke cases per short run and four timings per full run. Full default:
12.10, 16.22, 65.65, 278.34 us/op; SIMD: 11.93, 16.37, 66.17, 279.51 us/op.
Independent source/hash/format/diff audit passed. The verifier did not repeat
the full package or vet; those results above remain implementer evidence.
See `verification.txt` and `verification-report.txt` for retained records.

Spectackle check healed the changed test anchor and reported audit=0. It also
reported 136 inherited warnings and two inherited context-only errors in
performance/testing; this is not represented as a globally clean spec audit.
