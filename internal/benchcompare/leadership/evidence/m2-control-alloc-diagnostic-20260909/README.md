# Fixed-count control allocation diagnosis — September 9, 2026

Status: diagnostic evidence independently verified PASS, including implementation,
matched builds, and both complete measurement phases. This is
not a performance qualification rerun and cannot promote rejected GELU V2.

Spectackle task `T-01M23QYMYVE0M` consumes research `R-01M23Q8QJDE3Y`.
The [original V2 data and rejection](../m2-f64-gelu-neon-v2-20260909/README.md)
remain immutable. Its fixed parallel Softplus B/op medians increased by 2, 3,
and 2 bytes across three campaigns, with unchanged allocs/op and individually
nonsignificant differences. Source inspection found no control-path changes.
Go 1.27.1's benchmark result divides process-wide heap totals by adaptive N;
accounting effects are plausible, not an established cause.

## Prospective protocol

Add the identical, opt-in test-only diagnostic file to the old source
`dd1e779eb085bb621ed5dafff0a4351636b6e656` and V2 source
`361955ac35eaada696045f2cf828817b650174ec`. Build distinct matched diagnostic
binaries with Go 1.27.1, Darwin ARM64/v8.0, `GOEXPERIMENT=simd`, `CGO_ENABLED=0`.
Record all source, harness, compiler, and binary pins before measurement.
Never overwrite the original old, V1, or V2 binaries.

The guarded `TestCPUControlAllocationDiagnostics` invokes `testing.Benchmark`
on the unchanged SiLU backward, Sigmoid, and Softplus control benchmarks, in
that order. `-test.benchtime=1024x` fixes final N at 1024 for every cell and arm.
The normal one-iteration calibration is excluded from returned benchmark totals;
there is no additional warmup. This is a fresh-process, calibrated-loop heap
accounting diagnostic, not a warm/cold latency leadership measurement.

After each benchmark returns, emit exact integer N, total ns, `MemBytes`,
`MemAllocs`, and quotient/remainder for both allocation metrics. Nested benchmark
failures must fail the enclosing test. JSON emission and validation occur outside
the measured loop. Ordinary tests skip this diagnostic unless explicitly enabled.
Matched instrumentation changes both binaries; it is not interchangeable with
the original qualification binary or its measurements.

Run [run.sh](run.sh) in two serialized phases, both mandatory when valid:

1. `old-old`: both A and B invoke the exact same matched old diagnostic binary.
2. `old-v2`: A invokes that old binary; B invokes the matched V2 diagnostic binary.

Each phase uses campaigns **1–3** and pairs **1–7** (one-based indexing).
Odd campaigns use GOMAXPROCS 1 then 12; even campaigns reverse it. Arm order is
A then B when `(pair+campaign)` is even, otherwise B then A. All three controls
run in every invocation. Each phase has 84 invocations and 252 records; both
phases total 504 records. No owned concurrent build, test, profile, or other
benchmark. Preserve every sample and failure; do not replace invalid or unwanted
observations silently. Record environmental observations without claiming
continuous host idleness.

## Predeclared interpretation

Validate phase, protocol, frozen hashes, every invocation/arm/process/pair order,
exact fixed N, all three expected JSON records, PASS/FAIL, integer types and
quotient/remainder arithmetic. Parse integers losslessly. Reject incomplete or
malformed streams before analysis.

Report every paired B-minus-A total-byte and total-allocation delta, per campaign,
process count and benchmark. Retain medians, ranges, positive/zero/negative pair
counts, rounded quotients and remainders. Exact tied-rank p-values are descriptive,
not proof of equivalence. No post-hoc noise subtraction or chosen tolerance.

Repeated old-old movement demonstrates that the symptom is possible without a
binary difference; its absence cannot prove an absence of noise. A repeatable
old/V2 raw-total difference would require allocation-site evidence before causal
attribution. Neither outcome changes V2's rejected verdict. Any future gate
revision requires its own reviewed proposal and prospective full qualification.

Fresh independent plan review passed, with explicit source-absent disclosure.
Implementation and independent Go verification passed: full CPU suites and vet
in default/SIMD modes, guarded default skips, pre-measurement invalid-config
failures, and nested benchmark failure propagation (also race-checked).
The initial independent report caught an analyzer mismatch: zero allocation
totals were incorrectly rejected. The retained original FAIL report and
supplementary PASS report document its correction. Independent synthetic
analysis tests now pass 3 tests and 1,214 assertions, including exact integers
above 2^53, complete zero-allocation streams, and malformed-input rejection.
See `implement-raw.txt`, `verifier-raw.txt`, `allocdiag-verifier-report.txt`, and
`allocdiag-verifier-repair-report.txt` for evidence, including expected guard
skips/failures rather than treating exit codes alone as successful execution.
Matched-binary verification also passed. `build-pins.txt` and the matched-build
report retain exact source, diagnostic, compiler, and binary hashes. The file
`allocdiag-verifier-matched-build-tests.log` is a structured summary;
`allocdiag-verifier-matched-build-tests-stdout.log` contains literal captured
stdout, preserved without rerunning commands.

## Independently verified complete observations

Both serialized phases completed with exit 0: 84 successful invocations and 252
records each, all fixed at N=1024. No invocation failed or was replaced. The
predeclared analyzer validated all 504 records and 36 comparison cells. All
samples, exact paired deltas, quotient/remainder arrays and descriptive p-values
are retained in `old-old.txt`, `old-v2.txt`, `analysis.csv`, and
`analysis-audit.txt`. No owned build, test, profile, or other benchmark overlapped
either phase. Ordinary desktop background activity remained; host idleness was
not continuously monitored.

The same binary can exhibit the rounded-allocation symptom: at GOMAXPROCS=12,
Sigmoid's B-minus-A median B/op difference was +1, -1, and +1 across the three
old-old campaigns. Exact totals also varied; this is not merely information lost
by the final integer division. It demonstrates possibility without a binary
change, not the cause of the original V2 measurements.

Candidate trends remain unresolved. Parallel Softplus's median paired total-byte
deltas (over 1024 iterations, not per operation) were -2880, -5848, +1784 in
old-old, versus +3264, +3424, +10864 in old-v2. Its old-v2 median paired allocation
count deltas were +2, +9, +27. Separately, its old-v2 differences of median rounded
B/op were +1, +6, +6; a difference of medians is not a median paired delta.
Parallel Sigmoid also had positive old-v2 median paired byte deltas in all three
campaigns (+1424, +64, +672). These descriptive trends cannot be dismissed by
pointing to old-old variability or by treating nonsignificance as equality.

This diagnostic supplies no allocation-site attribution and does not establish
equivalence, a noise allowance, or a runtime optimization. Both original GELU
rejections remain unchanged. Any causal explanation needs allocation-site
evidence; any revised qualification protocol needs separate review and new
prospective complete measurements. No runtime or external-library performance
claim is made.

Fresh independent data review passed a separate strict parser and raw-to-CSV
recomputation of every field in all 36 cells; regenerated analyzer output was
byte-identical to `analysis.csv`. The reviewer also reran all 1,214 synthetic
assertions and rechecked source/protocol/binary pins. See
`allocdiag-data-verifier-report.txt` and `allocdiag-data-verifier-commands.txt`.
This is a diagnostic evidence PASS, expressly not a runtime qualification PASS.

Implementation checkpoint `c3a77bdb331b2150ab8e41d1110fd7d5d6baca2c` passed
all 16 CI jobs and every executed step, including soft SIMD lanes, in
[run 34395490441](https://github.com/jxsl13/goai/actions/runs/34395490441).
The exact-head audit is retained in `ci-implementation-checkpoint.jsonl`.

Generalizable accounting/attribution finding:
[perfscan #968](https://github.com/jxsl13/perfscan/issues/968). This reports a
validation opportunity, not an established cause of V2's byte differences.
