# M2 VJP architecture qualification — inconclusive

Task `T-01M26474J0EDX`, measured September 10, 2026.

The architecture-gated bounds-check candidate is **rejected for promotion**.
The original runtime is restored. This change retains hard native CI coverage
and a small, sanitized measurement fixture, not a performance optimization.
Neither a speedup nor a regression is established; no external competitor was
measured.

## Measurement scope

Apple M2 Pro, darwin/arm64, Go 1.27.1, pure-Go default build,
`GOMAXPROCS=1`. One diagnostic pilot ran seven alternating old/new pairs
(old first in odd pairs, new first in even pairs). Each invocation used
`-benchtime=200ms -count=2` for eight direct/taped ReLU F32/F64 cases at
2,048 and 262,144 elements. The first sample of each case is a warmup; the
second is retained. All 14 invocations exited successfully.

`pilot.csv` contains all 224 samples: 112 warmups and 112 retained samples,
without selection or replacement. It contains only invocation/pair identity,
arm, sample role, benchmark name, iteration count, ns/op, B/op, and allocs/op.
Nominal `SetBytes` throughput is omitted: it is not measured memory bandwidth.
Full local captures, environment records, compiler artifacts and review logs
are intentionally not distributed. Books and private reference documents are
not part of this package.

The measurement host had filesystem-service activity and prior swap use,
recorded before timing. No owned build, test or other benchmark overlapped.
The background activity is a possible confound, not an isolated explanation.
This pilot was diagnostic only; no full campaign or replacement sampling ran.
SIMD, higher concurrency, non-ReLU controls and training are not covered.

## Results

Independent medians, seven retained samples per arm and case. The old/new
ratios are descriptive, **not qualified speedups**. P-values are two-sided
benchstat comparisons. The minimum ratios were fixed before measurement.

| Case | Old ns/op | Candidate ns/op | Old/new | Required ratio | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct F32 N2048 | 2351 | 2314 | 1.0160 | 1.05 | 0.383 |
| Direct F32 N262144 | 246607 | 255329 | 0.9658 | 1.05 | 0.805 |
| Direct F64 N2048 | 2784 | 2851 | 0.9765 | 1.05 | 0.535 |
| Direct F64 N262144 | 264803 | 265773 | 0.9964 | 1.05 | 0.902 |
| Taped F32 N2048 | 2455 | 2460 | 0.9980 | control | 0.710 |
| Taped F32 N262144 | 246391 | 249639 | 0.9870 | 1.03 | 1.000 |
| Taped F64 N2048 | 2939 | 2849 | 1.0316 | control | 0.097 |
| Taped F64 N262144 | 269836 | 258114 | 1.0454 | 1.03 | 0.383 |

Five of six target ratio minima were missed; every target failed `p < 0.05`.
Within-arm `(maximum - minimum) / median` ranges span 5.48–44.02%, too noisy
for the intended small gains. Benchstat median-confidence widths are a
different statistic. B/op and allocs/op are unchanged in every sample:
direct cases use four allocations, taped cases six.

## Source and verification bindings

- Original and restored runtime SHA-256:
  `07adbf223e44d862d5f9e3e38a743eff54def329280bb8b76aa0ffe3f9b62945`.
- Candidate runtime SHA-256:
  `4427ca8dd0ac65f8724d29382220412c7cbfa53fd170bd9909ba5e86ee4be325`.
- Unchanged exact oracle SHA-256:
  `bd26d0195b02e23aa1406f01f7df25fff5aa65ea81fea0c3bae9f31d3359648c`.
- Frozen local pilot runner (`run.rb`),
  SHA-256 `824cadb31abf90465e4978ec82d7c88ac1970c08fc7edfe0079a6f9c5ca6d662`.
- Analysis: `golang.org/x/perf/cmd/benchstat`,
  `v0.0.0-20260709024250-82a0b07e230d`.

Native CI ran the exact oracle on Linux, Windows and macOS, each in
CGO-disabled default/SIMD and CGO-enabled race modes. Each execution had 218
matching RUN/PASS records (parent plus 217 children), zero skips and exit 0.
All 19 jobs and 214 executed steps succeeded at these checkpoints, including
the soft SIMD steps:

| Checkpoint | Source commit | CI run |
| --- | --- | --- |
| Original | `f1020ed63435139582e5d3dffc39cab0038cbc2f` | [34510578531](https://github.com/jxsl13/goai/actions/runs/34510578531) |
| Candidate | `47289163459f27e5e8586c190651f97a7b4a5d65` | [34514069758](https://github.com/jxsl13/goai/actions/runs/34514069758) |
| Restored | `f43d9fcb02b273e7e0676832970320e34bd99b5c` | [34522969503](https://github.com/jxsl13/goai/actions/runs/34522969503) |

The restored source also passed local Go 1.27.1 default/SIMD builds and short
tests, a strict race oracle, CGO-enabled build/tests, and CGO-disabled vet.
CI is a correctness gate, not performance evidence. Later publication-only
commits require their own exact-head CI before merge.

The generalizable lesson is recorded on
[perfscan #904](https://github.com/jxsl13/perfscan/issues/904#issuecomment-5625045104):
removing bounds checks in an inspected hot loop does not establish an
end-to-end gain. Inspect compiler output, then measure the actual workload.

## Recheck the small fixture

From this directory, Ruby's standard library and Minitest suffice:

```sh
ruby verify_pilot_csv.rb
ruby pilot_csv_test.rb
ruby verify_pilot_csv.rb pilot.csv old > old.txt
ruby verify_pilot_csv.rb pilot.csv new > new.txt
benchstat old.txt new.txt
```

The last three commands analyze existing retained data; they run no timing.
`build_pilot_csv.rb LOCAL_CAPTURE_DIRECTORY` regenerates the CSV on stdout
when the original private capture directory is available. It verifies the
numbered capture set, hashes, exits, arm schedule and all sample identities
before emitting the allowlisted fields. Local captures are not required to
validate or analyze the public fixture. Schema checks do not independently
authenticate measurement provenance; the independent review compared all
224 public rows to the original captures.

## Public artifact policy

Public Git contains code, small sanitized test data and this authored result
summary. Private documents, books, compiler binaries and large local evidence
bundles stay local. This applies to outgoing history, not merely the final
tree; the unpublished full-evidence commit is not an ancestor of this release.
