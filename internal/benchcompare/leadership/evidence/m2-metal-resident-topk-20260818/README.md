# M2 Metal resident Top-K sampling

This bundle is the promotion evidence for Spectackle proposal
`P-01M09DYYWEEW9` and task `T-01M09E0P15ERA`. It adds `TopKN` to Metal's
resident f32 buffer so the existing backend-neutral generation fast path can
return only a bounded candidate set instead of materializing and sampling all
vocabulary logits in Go.

## Verdict

The coherent-UMA implementation passes its Apple M2 Pro gates:

- exact index/value parity against a host reference for `n` from 1 through
  128,000 and `k` through 256, including a first-`n` prefix of an
  over-allocated buffer;
- deterministic descending-value, ascending-index tie ordering and explicit
  guards for invalid ranges, released buffers, and f16 storage;
- exact public `Decoder.Generate` token parity against the forced
  full-vocabulary fallback in both a hermetic model and the trained
  TinyLlama campaign;
- 32,000-logit/K=56 leaf medians of 62.669-66.691 us/op, 448 B/op, and two
  O(k) Go allocations;
- ten alternating trained-model pairs improve sampled tg64 from 153.97 to
  163.67 tok/s, a 1.06301x speedup, with exact 70-token sequence parity;
- two additional independent ten-pair campaigns reproduce 1.06615x and
  1.06778x, and a thermally slower third invocation remains at 1.06343x;
- matched forward-only tg64 moves only 1.00181x with the host copy removed,
  proving that readback alone is not the remaining llama.cpp gap.

The implementation scans `MTLResourceStorageModeShared` contents after the
decoder wait with a bounded native min-heap. It is O(n log k), uses fixed
stack storage for at most 256 candidates, crosses cgo once, and returns only
the k indices and values. It clears the end-to-end gate without adding another
command-buffer dispatch/wait, so a GPU-reduction prototype was not promoted.
No implementation claim is made for discrete-memory systems.

## Semantic boundary correction

Pinned llama.cpp b10450 `tools/llama-bench/llama-bench.cpp::test_gen` calls
`llama_decode`, synchronizes, and selects the next benchmark token randomly.
It never calls a logits accessor or sampler. GoAI's historical production
harness calls `Decoder.Step`, which additionally allocates 32,000 f32 values
and copies the complete resident logits buffer to Go.

Ten alternating same-decoder campaigns bound that mismatch rather than
silently crediting it as a kernel gain:

```text
Step + 32000-logit copy: 357.622417 ms / tg64 = 178.96 tok/s
device-only stepInto:     356.975542 ms / tg64 = 179.28 tok/s
boundary factor:          1.00181x
resident DownloadF32:     3.993 us/copy
```

The mismatch is real but too small to explain the pinned incumbent factor.
The leadership matrix therefore keeps forward-only llama-bench and sampled
generation as separate cells.

## Leaf protocol

```sh
go test ./backend/metal \
  -run '^$' \
  -bench '^BenchmarkMetalResidentTopKN32K56$' \
  -benchtime=500ms -count=5
```

Apple M2 Pro raw samples:

```text
66691 ns/op  448 B/op  2 allocs/op
66094 ns/op  448 B/op  2 allocs/op
65233 ns/op  448 B/op  2 allocs/op
62669 ns/op  448 B/op  2 allocs/op
63161 ns/op  448 B/op  2 allocs/op
```

The established host work on the same trained logits measured 29.151 us for
greedy and 459.297 us for temperature 0.8 + Top-K 40 + Top-P 0.9. The latter
includes the full f32-to-f64 widening and existing sampler, which is the
boundary the candidate replaces.

## Trained-model protocol

Model:

- path: `/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf`;
- bytes: 668,788,096;
- SHA-256:
  `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`;
- f16 K/V cache, Apple M2 Pro, macOS 26.5.1;
- GoAI baseline: `b26af00fa49fb6e22cca2f9807de249544d7bea8`.

```sh
GOAI_TINYLLAMA_GGUF=/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
  go test ./llamagpu \
  -run '^TestMetalLogitsReadbackAttribution$' -count=1 -v
```

The test alternates arm order every round. `GOAI_DEVICE_TOPK_SAMPLE=0` forces
the full-vector fallback; the original `GOAI_CUDA_TOPK_SAMPLE=0` spelling is
retained as a compatibility alias. The candidate arm type-asserts Metal's new
`deviceTopKer` capability.

```text
resident-TopK samples:
392.842958ms 373.646667ms 376.552291ms 402.241750ms 391.084083ms
390.280167ms 388.613958ms 389.581125ms 401.044042ms 391.033875ms

full-vocabulary fallback samples:
400.159583ms 399.777000ms 401.174209ms 446.733375ms 417.401125ms
415.077875ms 415.672875ms 415.569250ms 428.018208ms 417.407834ms

upper medians: 391.033875ms vs 415.672875ms
throughput:    163.67 tok/s vs 153.97 tok/s
speedup:       1.06301x
```

Two subsequent invocations produced 1.06615x and 1.06778x. A third campaign
whose absolute throughput fell to 151.55/142.51 tok/s under thermal pressure
still retained a 1.06343x paired gain. Every invocation preserved the same
70-token fast/fallback sequence.

## Contracts and generalizable finding

- `METAL-RESIDENT-TOPK-001` pins exact deterministic selection.
- `METAL-RESIDENT-TOPK-TRANSFER-001` prohibits full-n Go materialization.
- The O(n)-materialization-before-O(k)-consumption pattern is reported as
  [perfscan issue 767](https://github.com/jxsl13/perfscan/issues/767).
