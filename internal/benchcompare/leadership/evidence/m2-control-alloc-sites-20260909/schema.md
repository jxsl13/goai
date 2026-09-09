# Allocation-site capture schema v1

Prospective implementation contract supplementing `protocol.md`. No measurement
has run. Freeze this format and the helper/runner/analyzer hashes before the
old/old and old/V2 matrix. All byte/object arithmetic is integer arithmetic.

## Capture process

New test-only file: `backend/cpu/control_alloc_sites_test.go`, package `cpu_test`.
Entry: `TestCPUControlAllocationSites`. Noinline region:
`allocationSiteExecuteRegion(ctx *backend.Context, ins []*tensor.Tensor) (int, error)`.
It directly calls Execute exactly 1,024 times, returning the number of successful
calls and any error without formatting. Setup and one warmup are outside it.

Pre/post buffers each have **length and capacity 65,536** public
`runtime.MemProfileRecord` values. No buffer resize/retry. This capacity is a
prospective resource choice, not a result-selected setting.

Opt-in requires all of:

- `GOAI_ALLOC_SITE_DIAGNOSTIC=softplus`; empty disables before any setup or output.
  Any other nonempty selector is an error.
- `GOAI_ALLOC_SITE_OUT` is an absolute path that does not exist. The test creates
  this directory, then opens all three files below with exclusive creation before
  the first accounting boundary. Existing paths are rejected, not overwritten.
- `GODEBUG=memprofilerate=1`, `GOGC=100`, `GOMEMLIMIT=off` and actual
  `runtime.MemProfileRate == 1`. The last two explicitly pin the ordinary default
  GC policy for this new experiment; they are not a tuning sweep.
- Actual GOMAXPROCS is 1 or 12, `-test.count=1`, and
  `-test.run=^TestCPUControlAllocationSites$`. Opt-in short mode is rejected.
- Neither `-test.memprofile` nor `-test.memprofilerate` is explicitly supplied.
  Detect supplied flags with `flag.Visit`; do not confuse defaults with changes.

The test never assigns runtime.MemProfileRate, changes GC policy, or changes
GOMAXPROCS. The matrix runner sets the environment and exact test flags. A
separate explicitly labeled preflight may exercise the artifact pipeline before
matrix freeze; it is not a matrix observation or performance comparison.

## Files

`capture.json`, `tail.json`, and `allocs.pprof` are opened during setup. After
capturing both raw snapshots, serialize raw rows/frames and capture JSON, then
write one cumulative `pprof.Lookup("allocs").WriteTo(file, 0)` profile. Then read
tail MemStats and serialize `tail.json`. The tail-report write and file closes
are outside that tail read. All write/close errors fail the test and remain in
stdout/stderr; files are never deleted on failure.

Every JSON object below has exactly its listed keys. Arrays are arrays, not
null. No decimal/exponent representation or floating conversion for counters.
Field order is not identity. JSON string keys must be unique; analyzers reject
duplicate or unknown keys rather than silently overwriting values.

### Counters

Each counters object has unsigned 64-bit integers:
`total_alloc`, `mallocs`, `frees`, `heap_alloc`, `heap_objects`, `num_gc`.
Values come from the corresponding runtime.MemStats fields (NumGC widened).

### Raw row

Each raw row contains:

- `ordinal`: zero-based integer, consecutive in returned snapshot order.
- `alloc_bytes`, `free_bytes`, `alloc_objects`, `free_objects`: the literal
  signed 64-bit public fields. Retain invalid values for diagnosis; the analyzer
  rejects them under the reviewed protocol rather than normalizing them away.
- `pcs`: all 32 Stack0 values as canonical lowercase unpadded hex strings,
  including trailing `"0x0"` values. Preserve the complete public array. A
  nonzero value after the first zero is invalid for this fresh-buffer protocol.
- `frames`: every runtime.CallersFrames frame for the nonzero prefix, in order,
  retaining inline expansion and repeated frames. Each frame has `pc` (same hex
  syntax), `function` (string), `file` (string), and `line` (nonnegative integer).
  An empty stack has an empty frame array. Empty function names are retained,
  then make active nonzero rows invalid for comparison; files/lines are evidence,
  not identity. No requested-payload-size or internal ObjectSize is fabricated.

### Snapshot

Each snapshot contains `reported_count` (nonnegative integer), `ok` (boolean),
and `rows`. When ok is true, reported_count must be no greater than 65,536 and
rows has that exact length. On overflow/failure, rows is empty because MemProfile
does not copy any records; reported_count and ok retain the actual return values.
Do not serialize unused zero-filled buffer capacity as observed rows.

### capture.json

Top-level keys:

- `schema`: `"goai-control-alloc-sites-v1"`.
- `control`: `"SoftplusF64_256K"`.
- `n`: 1024; `warmup_calls`: 1; `completed_calls`: successful region calls.
- `go_version`, `goos`, `goarch`: runtime values, retained literally.
- `gomaxprocs`: actual process count; `godebug`, `gogc`, `gomemlimit`: environment
  strings validated above; `profile_rate_before`, `profile_rate_after`: actual
  runtime values before setup and after raw post capture.
- `raw_capacity`: 65536.
- `caller_function`: exact built function name from
  runtime.FuncForPC(reflect.ValueOf(allocationSiteExecuteRegion).Pointer()).Name(),
  obtained during setup. Expected package-qualified name is
  `github.com/jxsl13/goai/backend/cpu_test.allocationSiteExecuteRegion`; verify it
  in a built preflight before freezing the matrix.
- `worker_function`: `"github.com/jxsl13/goai/backend/cpu.poolWorker"`.
- `boundaries`: exactly `pre_raw_before`, `start`, `end`, `post_gc`, `post_raw`,
  each a counters object. Follow protocol.md order, using preallocated MemStats
  destinations; no formatting or assertion inside these boundaries.
- `pre`, `post`: snapshots defined above.
- `region_error`: empty on success, otherwise the returned error string,
  formatted only after post raw capture.
- `errors`: ordered strings describing any detected capture invalidity (rate
  change, snapshot failure, wrong completion count, pre-snapshot counter movement,
  or region error); empty on a successful capture. Preserve raw data on invalidity
  before failing the test. Analyzer validation remains independent and stricter.

Snapshot calls and all boundary reads remain in a fixed straight-line order even
if a snapshot reports insufficient capacity. The noinline helper returns early
only on an Execute error; immediate end counters are read before inspecting or
formatting that error. Post GC/raw capture still follows, preserving invalid data.

### tail.json

Exactly `schema` = `"goai-control-alloc-sites-tail-v1"`, `tail` = counters object,
and `report_write_excluded` = true. This is the post-capture/pprof serialization
boundary, not a recursive measurement of its own final write.

## Offline validation and derived output

The analyzer owns derivation, not the capture formatter. It validates every raw
row, preserves zero-active size-unknown rows separately, derives positive slot
sizes with exact division/checked multiplication, sums collisions at both the
raw-PC+size and function-stack+size layers, retains contributors/cardinalities,
and rejects decreasing cumulative fields. Use checked signed 64-bit arithmetic
for profile sums and deltas; cumulative MemStats fields are unsigned 64-bit,
and signed discrepancies must fit signed 64-bit. Do not key across binaries on
PC/file/line. Keep all sites, including zero-delta sites and the three disjoint
caller/worker-temporally-associated/other groups.

Per invocation retain a derived JSON document with immediate totals, raw-key and
normalized aggregates, zero-active row references, all group totals, profile
sums, immediate-minus-profile discrepancies and serialization tail counters.
The derived output is not a substitute for the raw capture. A valid capture must
have a positive caller allocation contribution in this allocation-producing
Softplus workload; this verifies the built marker is usable without prescribing
a particular per-site count or equating profile and immediate totals.

Runner/analyzer implementation must freeze their own exact manifest and derived
schemas before use. Validate artifact completeness, child exit/PASS and pprof
decoding as well as JSON; no hidden replacement or partial-matrix claim. The
standard cumulative pprof is only corroborative, not a region delta. No result
changes the original GELU rejection or any performance gate.
