# M2 Metal mixed-quant QKV fusion

This bundle is the promotion evidence for Spectackle proposal
`P-01M09A28JWF5Z`, implemented at
`53fd8c2751a1052ceabf8e6d5a91477f126331a0`. It targets the ten
TinyLlama-1.1B Q4_K_M blocks whose q/k projections are Q4_K while v is Q6_K.
The raw quantized weights remain resident for single-token decode. At prefill
M>=24, their exact f16 expansions occupy disjoint columns of one budgeted
matrix and feed one MPS GEMM.

## Verdict

The path passes its corrected promotion gates on Apple M2 Pro:

- the combined Q4_K/Q4_K/Q6_K expansion is bit-exact against three
  independently expanded and converted controls;
- the grouped GEMM stays finite with per-segment NRMSE at or below
  `0.000897414` at M64/M512;
- the profile contains three typed expansion events, two conversion events,
  and exactly one omitted MPS GEMM; M<24 retains the separate quant kernels;
- grouped expansion bytes equal the sum of the three replaced f16 expansions
  and remain charged to the existing cache budget across cache resets;
- the ten-layer projection stage improves 1.7378x at M64 and 1.2198x at M512
  across ten interleaved samples per arm;
- three independent trained-model invocations preserve 76/76 greedy tokens
  and exact checked logits; pp64 improves 1.0408-1.0572x, pp512 improves
  1.0093-1.0164x, tg64 stays at 0.9980-1.0050x, and unchanged-control spread
  stays below 1.036x.

The required five-pair shipping comparison measures GoAI at median 178.5
tok/s tg64 and 1,592.4 tok/s pp64. Pinned llama.cpp b10450 measures
193.311717 and 1,770.109320 tok/s. The remaining incumbent factors are
therefore 1.082979x at decode and 1.111598x at prefill. Relative to the prior
f16-KV shipping median, GoAI pp64 improves 1.050257x while tg64 is effectively
unchanged, matching the feature's prefill-only selector.

This is a measured prefill gain, not an overall leadership claim. llama.cpp
still leads these pinned shipping shapes by 8.30% and 11.16% when expressed as
incumbent/GoAI factors.

## Correctness and path protocol

```sh
go test ./backend/metal \
  -run 'TestMixedQGroup|TestRoPEPairSplitExact' -count=1 -v
```

`TestMixedQGroupExactExpansionAndBoundedParity` separates exact operand
validation from vendor-GEMM reassociation. The first frozen proposal,
`P-01M098NFP0FXGVPQN54J2QV2QR`, was rejected because it incorrectly required
bit-identical GEMM output after changing result width. MPS chooses a different
reduction schedule: storage remains exact, but synthetic segment NRMSE is
`0.000229429` to `0.000897414`. ADR `ADR-01M09A3S9JFMH` records the corrected
gate: exact stored operands plus bounded numerical and trained-model semantic
equivalence.

`TestRoPEPairSplitExact` proves the fused RoPE-and-scatter epilogue is
bit-identical to RoPEPair followed by three Copy2D operations. That epilogue is
required leverage, not decorative fusion: the first whole-model campaign with
the three copies reached only 1.0241x pp64 and regressed pp512 to 0.9750x.
Removing the layout debt produced the promoted ranges above.

## Leaf and trained-model protocols

```sh
GOAI_MIXED_QKV_PERF=1 \
  go test ./backend/metal \
  -run '^TestMixedQGroupProjectionGate$' -count=1 -v

GOAI_MIXED_QKV_REAL=1 \
GOAI_TINYLLAMA_GGUF=/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
  go test ./llamagpu \
  -run '^TestMetalMixedQKVRealModelQualityAndSpeed$' -count=3 -v
```

Each trained-model arm constructs a fresh otherwise-identical f16-KV decoder,
warms the expanded-weight cache outside the timed window, and alternates arm
order between campaigns. The selector proof counts zero groups in the control
and ten in the candidate. Model parsing, upload, and expansion fill are outside
the throughput boundary.

## Pinned incumbent campaign

`shipping.tsv` contains five alternating fresh-process pairs. Each GoAI process
uses one measured repetition and the shipping `NewQuantF16KV` constructor. The
llama.cpp process is Homebrew build 10450 / upstream `ece963f41`, f16 K/V,
FlashAttention auto, on the identical pinned model.

```sh
TINYLLAMA_GGUF=/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
GOAI_PROD_KV=f16 GOAI_PROD_REPS=1 \
  go test -tags vulkan ./internal/benchcompare \
  -run '^TestProdDecodeGGUF$' -count=1 -v

/opt/homebrew/Cellar/llama.cpp/10450/bin/llama-bench \
  -m /Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
  -p 64 -n 64 -r 1 -ctk f16 -ctv f16 -fa auto -o json
```

## Validation

- `go test ./backend/metal ./llamagpu -count=1`: pass;
- `go test -short ./... -count=1`: pass;
- `go test ./internal/apicheck -count=1`: pass;
- `CGO_ENABLED=0 go vet ./...`: pass;
- the only failure in the unshortened no-CGO full tree is
  `TestDiffusionLMGrammarE2E`; it reproduces identically on the baseline
  worktree and is outside this change's surface.

## Generalizable findings

- Vendor GEMM shape changes invalidate generic bit-exact output gates even
  when operands are exact: [perfscan #765](https://github.com/jxsl13/perfscan/issues/765).
- Producer fusion can lose all leverage to downstream layout materialization;
  fuse an existing epilogue with the required scatter:
  [perfscan #766](https://github.com/jxsl13/perfscan/issues/766).
