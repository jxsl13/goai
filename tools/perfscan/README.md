# perfscan

A static + benchmark lint for GoAI's recurring hot-path performance
anti-patterns. It exists so the patterns we keep rediscovering by profiling are
written down and checkable. See **PATTERNS.md** for the catalog, evidence, and
measured speedups.

## Run

```
go run ./tools/perfscan ./...          # static scan (P1 per-element closure, P2 closure-comparator sort)
go run ./tools/perfscan -tests ./nn    # include _test.go
go run ./tools/perfscan -json ./...    # machine-readable
go run ./tools/perfscan -strict ./...  # exit 1 on any finding (opt-in CI gate)

tools/perfscan/perfscan-bench.sh ./nn 2.0 500x   # P4: flag Fast/Slow pairs below 2.0x
```

## Discipline

Every hit is a **candidate**, not a bug. Confirm with a pre/post benchmark on
representative data before changing anything, and skip cold paths (one-time
init, eval-only). Prefer bit-identical fixes. Mark intentional/handled sites
with a `//perfscan:ignore` comment.

## Extending

When a new generic pattern proves worth codifying: add it to PATTERNS.md with
its evidence, and teach the scanner — extend `perElemVisitors` / `closureSorters`
or add a detector in `main.go`.
