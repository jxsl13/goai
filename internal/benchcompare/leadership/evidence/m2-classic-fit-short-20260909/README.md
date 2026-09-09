# Classic Fit guard short-mode repair

Status: implementation and fresh independent verification PASS; final-head CI
and merge remain separate release checks. No production algorithm change or
performance-gain claim. Base: main
`4f17f5a4df8957c1521e5a5a4575440f2a87f853`, Go 1.27.1 on M2 Pro.

Spectackle proposal `P-01M23Y1K1TEXF`, task `T-01M23Y32PDEKB`, new contract
`CLASSIC-FIT-SHORT-001`, existing root contract
`TIMING-ASSERTIONS-SKIP-ON-RUNNERS-001`.

## Trigger and scope

Hosted macOS cgo+metal CI run `34400229272` for draft PR1249 at
`a84cdb98ef117afdf020d9cee8383720b9baf3df` failed in the classic package, not the
Metal package: RandomForest100 took 312.445542 ms against a 300 ms ceiling.
`ci-failure-excerpt.txt` is explicitly a filtered excerpt, not a complete job log.
That observation does not establish an algorithmic regression or its cause.

The repair makes `TestClassicFitTimeGuard` obey the existing short-mode contract:
all five original models still Fit the same data once and every Fit error still
fails. Short mode logs each completed Fit smoke, then bypasses the wall-clock
ceiling. Full mode retains the 50/300/800/60/10 ms ceilings and architecture
reporting policy. No model, data, option, seed, solver or workflow is changed.
A Fit smoke does not add prediction parity or solver iteration-count assertions.

The isolated branch is based on main and does not include the rejected GELU
runtime candidate. PR: [1252](https://github.com/jxsl13/goai/pull/1252).

## Local implementation verification

`implementation.txt` retains every gate command, literal output and exit status,
including unsuccessful CLI setup calls. The four default/SIMD short/full focused
tests passed. Each short run emitted exactly five Fit smoke completions without
SKIP; each full run executed all five unchanged ceilings. Both full classic short
suites and vet passed; the CGO-enabled short race suite passed in 13.185 seconds.
These are guard-execution checks, not a before/after performance comparison.

Reviewed source SHA-256:
`91b4c285b42cc399d83b195497026c33220e326d65226d1cd0f270f4aa10ac57`.

The concrete table-driven threshold/short-continue fixture is reported on
[perfscan issue 868](https://github.com/jxsl13/perfscan/issues/868#issuecomment-5608528657).
The current detector was not rerun on this exact fixture; no false-negative
claim is made.

## Fresh independent verification

`verification.txt` retains the complete 348-line rerun and source review from a
separate detached checkout at `5f60d52b1beb7fc45179723be73e0d74bbdb27e6`, without
using the implementer transcript as evidence. All four focused modes, both
complete classic short suites, both vet modes and default CGO short race passed.
The race rerun took 13.057 seconds. Every short focused run had five smoke logs
and no SKIP; each full run had five timing lines.

Two separately compiled mutations validate the distinction:

- Replacing `el > c.ceiling` with `el > 0` still passed all five short smokes, but
  full mode exited 1 with all five forced ceiling assertions.
- Replacing the DecisionTree Fit closure with a synthetic error failed in both
  short and full modes before subsequent cases, as required.

Both mutations were restored byte-for-byte to the source SHA above, with clean
Git status. Mutant hashes, compiler success, literal failure outputs and source
restoration are in the report. Expected mutation failures are not failed
candidate gates. No production or benchmark workload was run. Non-arm64
report-only behavior was reviewed as unchanged source, not executed locally.

## CI checkpoint and release boundary

`ci-implementation-checkpoint.json` retains all job/step results for run
`34403018535` at exact source/evidence head
`5f60d52b1beb7fc45179723be73e0d74bbdb27e6`: all 16 jobs and every executed step
passed, including soft SIMD lanes. Final evidence/lifecycle commits require
their own final-head CI check before merge; this checkpoint is not substituted
for that check. Spectackle retains inherited repository-wide warning/CTX debt;
the new contract and source change must have no new drift.
