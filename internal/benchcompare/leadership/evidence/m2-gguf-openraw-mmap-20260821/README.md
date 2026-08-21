# M2 mmap-backed quantized GGUF open — 2026-08-21

## Verdict

`gguf.OpenRaw` retains regular-file tensor bytes in one read-only mapping instead
of copying the encoded data section into Go heap. On the pinned 638 MiB TinyLlama
Q4_K_M file, mapped-view open improves 78.824 to 8.860 ms (**8.90x**) and a strict
open-plus-full-consumer-copy cell improves 113.81 to 72.72 ms (**1.57x**). Both
comparisons are `p=0.000`, n=10. Heap traffic falls from 652.25 to 15.07 MiB/op
(**−97.69%**, about 637 MiB removed).

At matched raw mapped-view semantics, GoAI is **89.17x faster** than gguf-py
0.19.0 for open and **11.86x faster** when both copy every encoded tensor.

## Claim cells and pins

- Hardware: Apple M2 Pro, 12 logical CPUs, 32 GiB; darwin/arm64.
- OS: macOS 26.5.1, build 25F80.
- Go: 1.26.6; Spectackle: 0.9.3; external perfscan: v1.71.0.
- Model: `tinyllama-1.1b-q4km.gguf`, 668,788,096 file bytes, 201 tensors,
  667,078,656 encoded tensor bytes, SHA-256
  `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
- Incumbent: gguf-py 0.19.0, NumPy 2.5.1, Python 3.14.7.
- Cache state: warm page cache. Every fresh process performs an untimed warm load
  and then one timed load. The Go campaign alternates which arm runs first.

The buffered Go control and mmap candidate are compiled into the same binary.
`openRawFile(path, false)` preserves the former `ReadRaw(bufio.Reader)` file
boundary; `openRawFile(path, true)` is the production `OpenRaw` route.

## Go A/B

| cell | buffered control | mmap candidate | result |
|---|---:|---:|---:|
| mapped-view open | 78.824 ms ±2% | 8.860 ms ±10% | −88.76%, 8.90x; `p=0.000`, n=10 |
| full encoded-tensor copy | 113.81 ms ±2% | 72.72 ms ±8% | −36.11%, 1.57x; `p=0.000`, n=10 |
| heap, either cell | 652.25 MiB/op | 15.07 MiB/op | −97.69%; `p=0.000`, n=10 |

The full-copy benchmark allocates its consumer buffer before timing and reuses it.
It sorts tensor names once, copies all 667,078,656 encoded tensor bytes, and reads
the destination sink before closing. This proves the win after complete payload
consumption. The open-only cell intentionally measures the lazy mapped-view API;
its derived B/s is not a claim about physical storage.

`campaign.txt` contains every order-alternated process, `control.txt` and
`candidate.txt` are normalized inputs, and `benchstat.txt` is the tool output.

## Incumbent race

`testdata/bench_gguf_raw_open.py` uses `gguf.GGUFReader` on the identical file.
It warms once in each fresh process, retains raw mapped tensor arrays, and under
`GGUF_RAW_COPY=1` copies each encoded array into independent NumPy storage.

| cell | GoAI median | gguf-py median | GoAI advantage |
|---|---:|---:|---:|
| mapped-view open | 8.8595 ms | 789.9960 ms | 89.17x |
| full encoded-tensor copy | 72.7198 ms | 862.2366 ms | 11.86x |

`ggufpy.jsonl` contains all ten fresh-process observations per mode, including
versions, file size, tensor count and encoded byte count.

## Correctness and ownership boundary

- `TestOpenRawMatchesBuffered` compares every metadata value, tensor name, type,
  shape and encoded byte, and proves every mapped view is capacity-clamped.
- `TestOpenRawMappingLifetime` proves no early release, exactly one release after
  repeated closes and handle value-copies, and cleanup on malformed mapped input.
- `RawFileHandle.Close` invalidates every `QuantTensor.Data` view. There is no
  finalizer because a copied tensor can outlive the handle.
- Unsupported platforms, special/empty files and mapping failures use buffered
  `ReadRaw`; its handle has a no-op `Close`.
- Existing `ReadRaw(io.Reader)` and eager `ReadFile` behavior are unchanged.

The repository contract is `GGUF-RAW-MMAP-LIFETIME-001`. The generalizable
regular-file-copy pattern is reported as
[perfscan issue 798](https://github.com/jxsl13/perfscan/issues/798).

## Reproduce

```sh
go test -c -o /tmp/goai-gguf-openraw.test ./format/gguf

cd format/gguf
TINYLLAMA_GGUF=/absolute/path/to/tinyllama-1.1b-q4km.gguf \
  /tmp/goai-gguf-openraw.test -test.run '^$' \
  -test.bench '^BenchmarkOpenRawModel(Path|Copy)$' \
  -test.benchtime=1x -test.count=1 -test.benchmem

GGUF_BENCH_FILE=/absolute/path/to/tinyllama-1.1b-q4km.gguf \
  .venv/bin/python testdata/bench_gguf_raw_open.py
GGUF_RAW_COPY=1 GGUF_BENCH_FILE=/absolute/path/to/tinyllama-1.1b-q4km.gguf \
  .venv/bin/python testdata/bench_gguf_raw_open.py
```

Alternate `GOAI_GGUF_MMAP_CANDIDATE_FIRST=0/1` across ten fresh Go processes.
Run each Python mode in ten fresh processes. The release gates additionally cover
the complete package suite, race detector, CGO-disabled build/test, vet, non-Unix
cross-compiles, API check, Spectackle drift check, and repository preflight.
