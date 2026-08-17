# M2 GGUF ReadFile mmap evidence — 2026-08-17

## Claim cell

- Hardware: Apple M2 Pro, arm64
- OS: macOS 26.5.1, build 25F80
- Go: 1.26.6
- Model A/B: `models/tinyllama-1.1b-q4km.gguf`, 638 MiB, full eager F32 materialization
- Incumbent fixture: deterministic 64 MiB F32 GGUF, 16 tensors of `[1024,1024]`
- Incumbent: gguf-py 0.19.0, NumPy 2.5.1, Python 3.14.7
- Cache state: warm page cache; each timed arm performs the same full parse and materialization

## Model A/B

Ten fresh processes were run with `-benchtime=1x -count=1`; odd runs used control
first, even runs set `GOAI_GGUF_MMAP_CANDIDATE_FIRST=1`. The benchmark warms and
validates each path before starting its timer.
The raw files normalize the `/buffered` and `/mmap` subtest suffixes to one common
benchmark name so `benchstat` treats the two files as the arms of one comparison.

```sh
go test ./format/gguf -run '^$' -bench '^BenchmarkReadFileModelPath$' \
  -benchtime=1x -count=1 -benchmem

GOAI_GGUF_MMAP_CANDIDATE_FIRST=1 \
go test ./format/gguf -run '^$' -bench '^BenchmarkReadFileModelPath$' \
  -benchtime=1x -count=1 -benchmem

benchstat buffered.txt mmap.txt
```

`benchstat` reports 182.00 ms versus 97.12 ms: −46.64%, 1.87x faster,
`p=0.000`, n=10. Heap bytes fall 13.14%, from 4.735 GiB to 4.113 GiB;
allocations are statistically unchanged.

## Incumbent race

Each session ran one side and then the other, alternating order across six fresh
shells. Both programs materialize every F32 tensor. The Go harness also checks the
fixture's deterministic values.

```sh
GGUF_BENCH_FILE=/private/tmp/gguf_bench.gguf \
  .venv/bin/python testdata/bench_gguf_load.py

GGUF_BENCH_FILE=/private/tmp/gguf_bench.gguf \
  go test ./internal/benchcompare -run '^TestGGUFLoadCompare$' -count=1 -v
```

Six-session medians: GoAI 3.04 ms; gguf-py 6.22 ms; GoAI is 2.05x faster.

## Correctness and scope

The result covers regular-file `ReadFile`, not streaming `Read`. Mapped and
buffered results compare exactly after `ReadFile` returns, proving no returned
value aliases the released mapping. Mapping failure and unsupported platforms use
the existing buffered reader. Concurrent truncation is excluded from the contract.
