# M2 safetensors single-tensor direct-load evidence (2026-08-17)

## Claim boundary

- Hardware: Apple M2 Pro, darwin/arm64, macOS 26.5.1.
- Go: 1.26.6.
- Incumbent: safetensors 0.8.0, NumPy 2.5.1, Python 3.14.7.
- Fixture: `testdata/bench_safetensors_load.py`, 16 deterministic F32 tensors of shape
  `[1024,1024]`; file size 67,110,088 bytes. The partial workload eagerly materializes `t15`
  (4 MiB) from a warm page cache and verifies its reference value.
- Semantics: both implementations open the file, locate one named tensor, and return independent
  materialized NumPy/Go storage. Header parsing, file-size validation, and open/close time are
  inside the measured region.

## Internal A/B

The unchanged code at `b47b3a5f` was archived and compiled as a separate test binary. Candidate
and baseline binaries then ran in 10 fresh-process pairs with order reversed every pair. The
committed `BenchmarkLoadTensorFile` reports time and allocation cost.

| 4 MiB tensor from 64 MiB file | Baseline median | Direct-load median | Effect |
|---|---:|---:|---:|
| Latency | 614,468 ns | **280,561 ns** | **2.19x faster**, -54.34% |
| Heap bytes/op | 12,602,318 | **4,200,899** | **-66.67%**, about 8.4 MiB/op removed |
| Allocations/op | 124 | **80** | **44 fewer** |

All 10 candidate latency samples are below all 10 baseline samples. The implementation removes
the selected-range payload allocation, synthetic JSON header, in-memory container copy, second
JSON parse, and full-loader map construction. Common little-endian dtypes now `ReadAt` directly
into final tensor storage. Widening dtypes retain one selected-range scratch buffer and share the
same centralized decoder as full loading.

## Incumbent comparison

Ten fresh Go/Python pairs, again reversing order each pair:

| Workload | GoAI median | safetensors 0.8.0 median | GoAI effect |
|---|---:|---:|---:|
| One 4 MiB tensor | **0.2825 ms** | 0.4195 ms | **1.48x faster**, -32.66% latency |
| Full 64 MiB / 16 tensors | 5.670 ms | 5.675 ms | parity (1.001x) |

This is a shape- and contract-scoped leadership claim, not a universal loader claim.

## Rejected rung

A transient whole-file read-only mmap was implemented and compared with the already optimized
direct `ReadAt` path. Across five 2-second samples, direct `ReadAt` was 265,247 ns/op median and
mmap was 347,943 ns/op: mmap was 31.18% slower with identical allocations. Per-call map/unmap
overhead costs more than the saved read syscall for this 4 MiB selected-range workload, so the mmap
code was removed. The full-file GGUF result does not generalize to single-range extraction.

## Reproduction

Generate the shared fixture and run the production benchmark/reference harness:

```sh
ST_BENCH_FILE=/private/tmp/st_bench.safetensors \
  .venv/bin/python testdata/bench_safetensors_load.py
ST_BENCH_FILE=/private/tmp/st_bench.safetensors \
  go test ./format/safetensors -run '^$' -bench '^BenchmarkLoadTensorFile$' -benchmem
ST_BENCH_FILE=/private/tmp/st_bench.safetensors \
  go test ./format/safetensors -run '^TestSafetensorsLoadCompare$' -count=1 -v
```

`baseline-candidate.tsv` and `go-python.tsv` retain the raw paired samples. Correctness is gated by
package round trips, exact `LoadTensor` versus `LoadFile` checks for verbatim and every widening
dtype, independent F16/BF16 reference fixtures, hostile-entry tests, and the no-other-tensor
allocation test.
