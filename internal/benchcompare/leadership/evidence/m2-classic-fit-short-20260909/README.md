# Classic Fit guard short-mode repair

Status: implementation and independent verification in progress. No production
algorithm change or performance-gain claim. Base: main
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
runtime candidate. Its full verification and CI evidence will be recorded before
marking the PR ready or merging it.
