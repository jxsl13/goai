# M2 SiLU leadership evidence — 2026-08-14

This directory preserves two complete benchmark sessions rather than selecting
the quieter result after observing both. Each session contains ten alternating,
prebuilt GoAI/PyTorch samples. The first ten records in `goai.txt` and
`pytorch.txt` are session A; the last ten are session B. `benchstat.txt` and
`summary.json` combine all 20 samples per implementation.

Session B slowed for both implementations near its end. Keeping those samples
makes the combined result conservative and exposes system/thermal drift instead
of silently discarding it. The combined medians are 137,040 ns/op for GoAI and
208,354 ns/op for PyTorch: GoAI is 1.520x faster (`p=0.000`, `n=20`).

The GoAI benchmark executes `backend.ExecuteInto(OpSiLU)` on caller-owned F64
input/output tensors. PyTorch executes `aten.silu.out` on caller-owned F64
input/output tensors. Inputs, outputs, and thread counts are fixed before the
timer. `quality.txt` records bitwise Execute/ExecuteInto and vector/tail parity,
edge handling, and a 262,145-value accuracy sweep. The Python records include
their independent quality result.

`pytorch-thread-sweep.txt` verifies that eight intra-op threads are the measured
PyTorch winner on this host and shape. It retains five one-second samples for
each serious candidate (4, 6, 8, 10, and 12 threads); the median at eight threads
is 205,719 ns/op versus 227,084 at six and 240,049 at twelve. The sweep selects
configuration only and is not counted in the 20-sample comparison.

The cell remains provisional. This checkout is backed by a bare Git repository,
so the new source has no immutable commit even though the production-source and
prebuilt-binary SHA-256 values are recorded. Session A was captured before the
collector learned to label that condition explicitly and therefore records the
base commit with `-dirty`; session B records `bare-workspace-uncommitted`.

Reproduce from the repository root:

```sh
go install golang.org/x/perf/cmd/benchstat@v0.0.0-20260709024250-82a0b07e230d
go run ./internal/benchcompare/leadership collect-silu \
  -root . -out /tmp/goai-silu-evidence -samples 10 -seconds 1 \
  -go-procs 12 -torch-threads 8 -python .venv/bin/python \
  -benchstat "$(go env GOPATH)/bin/benchstat"
```
