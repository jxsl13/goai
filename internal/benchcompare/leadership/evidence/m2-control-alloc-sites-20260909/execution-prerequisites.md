# Allocation-site execution prerequisites

No live preflight or matrix has run. The capture and per-invocation analyzer are
independently verified under `T-01M24137G7EHM` and `T-01M241R5P3EH8`; implementation
completion does not satisfy the remaining execution prerequisites below.

1. Independently review and rerun each implementation's declared pure tests.
   Bind prospective rules to actual indexed symbols. Preserve initial failures
   and corrections. Confirm the helper is byte-identical in both future builds
   and that production source is unchanged by diagnostic instrumentation.
2. Build a standalone pprof decoder with the pinned Go 1.27.1 SDK in a serialized
   build window. Its `pkg/tool/darwin_arm64/pprof` executable is absent locally;
   do not treat `go tool pprof` as an already verified executable. Use the SDK's
   bundled `cmd/pprof` source, record the literal build command and hash, and
   verify the selected offline decode flags before profiling. Do not download a
   different decoder implicitly.
3. Draft a separate, retained live **preflight** task after implementation
   review. Build a distinct helper binary and use fresh output paths. Check
   startup guards, real caller/worker names, snapshot status, raw/frame JSON,
   cumulative pprof decoding, exact integer offline analysis and artifact sizes.
   A preflight is not a matrix observation or comparative performance result.
   Failed preflights remain labeled and retained; never relabel one as a sample.
4. Before any matrix process, review and freeze a runner/manifest/paired-summary
   implementation. It must encode the exact protocol order, pin source/helper/
   SDK/binaries/analyzers/decoder and schemas, use exclusive output creation and
   retain literal argv, environment, child exit/PASS status, stdout/stderr, raw
   capture/tail/pprof plus offline decoder output and derived JSON per invocation.
   Decoding and analysis occur after the child exits, never inside the profiled
   accounting window. All input/output paths and hashes must be validated.
5. Validate runner completeness and failure behavior with synthetic processes
   before actual use: missing, repeated, reordered, replaced, interrupted,
   failed or skipped invocations; wrong pins/flags/count; decode/analyzer failures;
   partial artifact writes and reused paths. No silent reruns or partial-matrix
   success. Stop on invalidity and retain the unfinished planned phase as such.
6. Create **new** detached old/V2 source worktrees at the protocol pins and build
   separately named binaries. Do not reuse or mutate the completed fixed-count
   diagnostic's worktrees, binaries, transcripts or analyses. Freeze all hashes
   and the complete expected 168-entry order before old/old starts. Record disk
   capacity and expected retained artifact volume based on the labeled preflight;
   do not solve an evidence-storage problem by silently dropping raw records.
7. Reserve one owned profiling window. No concurrent owned build/test/profile/
   benchmark. Record ordinary background-load limitations without claiming an
   idle operating system. Run complete old/old before old/V2, then independently
   validate every invocation and paired result before drawing an association.

The paired summary must retain all immediate B-minus-A totals and all normalized
site deltas, including unchanged sites, for every seven-pair campaign/process
cell. Report ranges, medians and signs as specified in `protocol.md`. Do not
subtract old/old variation, select only favorable stacks, replace failed runs,
turn nonsignificance into equivalence, infer a unique allocation statement from
function-stack aggregation, or revise the original GELU rejection. Substantiated
generalizable findings should extend existing perfscan #968 rather than create
a duplicate issue.
