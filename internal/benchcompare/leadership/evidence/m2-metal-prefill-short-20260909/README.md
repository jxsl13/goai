# Preserve prefill smoke coverage in short-mode CI

Status: implementation and fresh independent focused verification passed on
real Metal. Final CI remains mandatory. Test-only change, no kernel performance
claim.

Proposal `P-01M23R1VJNEJJ`, task `T-01M23R33X1EC7`, based on main
`b9c059464edb4a339073c75ea8037c5472bf9aea`. Existing contract
`TIMING-ASSERTIONS-SKIP-ON-RUNNERS-001` reserves timing budgets for non-short
runs while keeping correctness checks active. New anchored
`METAL-PREFILL-SHORT-001` requires one recorded operation in each of five
short-mode cases, preserving operation-error checks and asserting no timing
ceilings.

## Observed CI failure

[Run 34390073218](https://github.com/jxsl13/goai/actions/runs/34390073218) at
GELU evidence checkpoint `76d889fbd7b182070e172cbd2fb4a476f0a152d4` failed the
macOS cgo+Metal test step. The exact assertion was:

```text
2026-09-09T18:41:53.4929240Z --- FAIL: TestPrefillOpCosts (4.43s)
2026-09-09T18:41:53.4931460Z prefill_ops_bench_test.go:106: RMSNorm rows=64 d=2048: 68.02 us > 60 us ceiling
```

The test file is unchanged between main and that checkpoint; a previous
checkpoint passed all jobs. This establishes a shared-runner timing failure,
not a new GELU or Metal numerical failure. No retry-until-pass, raised threshold,
or integrity-check bypass is used. The separate rejected GELU qualification
remains rejected regardless of this CI repair.

## Change and verification boundary

`TestPrefillOpCosts` retains all five cases: RMSNorm, SwiGLU, residual add, and
attention at key lengths 64 and 128. Under `testing.Short`, each still records,
commits, waits for, and frees one operation, preserving recorder creation and
operation-error checks and logging each completed smoke case. It returns before
the repeated timing measurements and ceiling comparison.

Non-short mode still uses 15 repetitions, the minimum of 16- and 128-operation
runs, the same slope calculation, shapes, and ceilings: 60, 90, 40, 600 and
1200 microseconds. Production code, workflows, and numerical tests are unchanged.
This is not an AI-kernel speedup or a relaxation of a local performance gate.

The [implementation logs](implementation.txt) show real-Metal focused short and
non-short runs in default/SIMD modes, full Metal package short-mode tests and
package vet in both modes. All passed; short mode emitted five smoke messages,
and non-short mode retained five timing outputs. Pinned Go 1.27.1 was used.
Fresh independent verification must rerun the focused matrix from the actual
diff, check the unchanged full-mode protocol and thresholds, and distinguish
real GPU execution from an unavailable-device skip.

That fresh [independent review](verification-report.txt) now passes: all four
real-Metal executions passed with zero skips, five smoke cases in each short
mode and five timing results in each non-short mode. It confirmed the unchanged
full protocol, thresholds and operation-error handling directly from the diff.
The [raw verification](verification.txt) also retains the first four sandbox
attempts, which exited zero but **skipped** for unavailable Metal. The initial
exit-code interpretation was corrected before validation or publication; those
skips are not passing GPU evidence. The verifier did not independently rerun
the full package or vet, so those remain implementation checks only.

Spectackle refreshed the one changed function anchor (`healed=1`, `audit=0`).
It still reports the inherited 136 warnings and two context-mode errors in
the performance/testing bundles; this is not a globally clean spec-check claim.
Those pre-existing findings are outside this single-test repair.

Before merge, every final CI job and individual step, including soft SIMD lanes,
must succeed. Delete only the exact remote feature branch after verified merge.
