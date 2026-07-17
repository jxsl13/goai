# GGUF loading in GoAI

GoAI loads [llama.cpp](https://github.com/ggml-org/llama.cpp) GGUF checkpoints —
the single-file format the local-inference ecosystem distributes models in —
straight into runnable, trainable GoAI models. Nineteen architectures load in
full precision; fourteen families additionally decode
**directly from the quantized ggml blocks**, never materializing full-precision
weights.

This document is the reference for *which* architectures load, *how* each one's
on-disk layout maps to GoAI's, and — most importantly — the methodology that
keeps those mappings honest. Every convention below was verified against
llama.cpp master source (the `conversion/*.py` converters, `src/models/*.cpp`
runtime, `src/llama-arch.cpp` tables, and `gguf-py/gguf/constants.py`), not
inferred from a round-trip.

## Using it

```go
f, err := gguf.ReadFile("model.gguf") // float tensors dequantized to f32
model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
tok, err := nlp.BPEFromGGUF(f.Metadata) // the embedded tokenizer
// model.Generate(...), or fine-tune it — a loaded model is a normal GoAI model.
```

For quantized decode the tensors stay in their ggml block form:

```go
r, err := os.Open("model.Q4_K_M.gguf")
raw, err := gguf.ReadRaw(r) // tensors kept in their ggml Q-block bytes
qmodel, err := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
```

## Float loaders (19 architectures)

Each loader has a matching `*ToGGUF` inverse (used for round-trip tests and for
writing GoAI models out in llama.cpp's convention). "Permute" refers to the
rotary row layout: llama.cpp rotates *consecutive* value pairs (`ROPE_TYPE_NORM`)
for some archs and *split-half* pairs (`ROPE_TYPE_NEOX`) for others; GoAI's
`OpRoPE` is split-half, so NORM-rope archs are un-permuted at load.

| Arch string | GoAI loader | Load-critical conventions |
|---|---|---|
| `llama` | `LlamaFromGGUF` | q/k **and their biases** permuted on disk (NORM rope) → un-permuted at load (§B67/§B68) |
| `qwen2` | `Qwen2FromGGUF` | llama block structure + q/k/v biases; NEOX, no permute |
| `qwen3` | `Qwen3FromGGUF` | + per-head QK-norm; NEOX, no permute |
| `gemma` | `GemmaFromGGUF` | (1+γ) norms pre-folded on disk; √dim embed scale at runtime; tied head |
| `gemma2` | `Gemma2FromGGUF` | +1 pre-folded sandwich norms; soft-caps default 50/30 when absent; sliding window via Ctx clamp; 27B query-scale from block_count |
| `phi3` | `Phi3FromGGUF` | packed `attn_qkv` [q;k;v] and `ffn_up` [gate;up], unpacked by row-slice; NEOX |
| `llama` (+`expert_count`) | `MixtralFromGGUF` | rides the llama arch; q/k permuted → un-permuted; fused 3-D expert banks |
| `starcoder2` | `StarCoder2FromGGUF` | LayerNorm γ+β pairs; biased projections; NEOX; tied-head fallback; fused-qkv accepted |
| `gptneox` | `GPTNeoXFromGGUF` | converter **de-interleaves** packed qkv → `[all-q; all-k; all-v]`, sliced in thirds; partial rotary; parallel-residual required |
| `mpt` | `MPTFromGGUF` | pure rename (qkv keeps HF chunk order — the anti-`gptneox`); ALiBi via `max_alibi_bias`; no rope |
| `falcon` | `FalconFromGGUF` | "jploski" fused-qkv transform is the identity for MQA `n_head_kv=1`; 40B dual-norm rejected; NEOX |
| `stablelm` | `StableLMFromGGUF` | NEOX partial rotary (`rope.dimension_count` = rotated *channel* count); sequential form gated by `ffn_norm` presence |
| `olmo2` | `OLMo2FromGGUF` | pure rename (first-gen `olmo` IS permuted, `olmo2` is NOT); post-norms `post_attention_norm`/`post_ffw_norm`; full-width QK-norm |
| `deepseek2` | `DeepSeekV2FromGGUF` | MLA: `kv_b` split on disk (`attn_k_b` pre-transposed for absorption); `head_count_kv`=1 while split tensors carry all heads; `key/value_length` are MQA cache widths, true head dims under `*_mla`; NORM rope de-interleaved |
| `command-r` | `CohereFromGGUF` | interleaved q/k rows + NORM rope → `permuteInterleaveToSplit`; `logit_scale`; tied head |
| `nemotron` | `NemotronFromGGUF` | (1+γ) LayerNorm1P pre-folded on disk; NEOX partial rotary; untied head |
| `granite` | `GraniteFromGGUF` | inherits the llama converter's q/k permute → un-permuted; four scalar multiplier keys; `logit_scale` required |
| `mamba` | `MambaFromGGUF` | SSM: `A_log`→−exp(A_log); conv1d squeezed `[d_inner,1,k]→[d_inner,k]`; `ssm_a`/`ssm_d` carry no `.weight` suffix; packed `ssm_in`/`ssm_x` |
| `jamba` | `JambaFromGGUF` | hybrid: mixer interleave in the per-layer `head_count_kv` vector (0=Mamba); NoPE attention; dedicated `ssm_{dt,b,c}_norm`; fused-expert MoE |

## Quantized decode (14 families)

`QuantLlamaFromGGUF` (llama), `QuantQwen2FromGGUF`, `QuantQwen3FromGGUF`,
`QuantPhi3FromGGUF`, `QuantGemmaFromGGUF`, `QuantMixtralFromGGUF`, `QuantGraniteFromGGUF` (the
scalar-multiplier Granite type — its four scalars survive the metadata and are
applied on the quantized forward) `QuantStarCoder2FromGGUF`, `QuantGPTNeoXFromGGUF`, `QuantFalconFromGGUF`,
`QuantMPTFromGGUF`, `QuantNemotronFromGGUF`, `QuantStableLMFromGGUF` and
`QuantOLMo2FromGGUF` (the LayerNorm/dedicated-type quant twins: biased
projections as Q-block matmuls plus f32 bias adds, parallel residuals, ALiBi,
pre-folded LayerNorm1P, partial rotary, OLMo 2 post-norm with full-width
QK-norms, every fused qkv unpacked losslessly on the quantized bytes) decode
straight from ggml Q-blocks (Q8_0, Q4_0, and the K-quants) with no dequantize
step: each projection is a `nn.QuantLinear` over the raw block bytes, and only
the small precision-sensitive pieces (norm gains, the embedding lookup table,
MoE routers) are dequantized to f32.

The enabling insight is that **ggml block quantization is row-granular** — every
block covers consecutive elements of a single row, and a row width that is not a
block multiple is rejected everywhere in the pipeline, so blocks never span
rows. Therefore any *row operation* on a quantized tensor is an exact byte-range
manipulation, provably identical to quantizing the transformed float weights:

- **row-slice** (`quantSliceRows`) unpacks Phi-3's packed `attn_qkv`/`ffn_up`;
- **row-permute** (`quantPermuteRows`) undoes the llama-arch q/k interleave for
  quantized Llama and Mixtral;
- **expert-slice** un-fuses Mixtral's 3-D expert banks.

None of these ever dequantize-and-requantize (which would compound quantization
error); each is proven byte-equal to the direct quantization in the test suite.

## The verification methodology: why round-trips are not enough

Two hard-won lessons (SPEC §B67, §B68) govern every loader's tests, after a real
bug slipped through symmetric testing:

1. **A round-trip test proves nothing about a foreign convention.** `LlamaFromGGUF`
   silently mis-loaded *every* real llama/mistral file for a time, because its
   only fixtures were GoAI's own `ToGGUF → FromGGUF` round-trips — a symmetric
   test cannot detect that both sides share the *same wrong* assumption about
   the on-disk layout. The fix: **every loader carries at least one fixture whose
   convention-critical transform is implemented independently, test-side**, from
   raw Hugging Face tensors — e.g. an independent re-implementation of
   llama.cpp's rotary permute (`llamaCppPermute` in the tests), the literal
   Falcon "jploski" transform, or the GPT-NeoX de-interleave — and the loaded
   model must reproduce the *transformers* golden logits.

2. **Fixtures need nonzero values in every convention-critical tensor.** The
   same bug hid in the q/k *biases* even after the weight fix, because the
   available golden's attention biases were all zero — and a zero vector is
   invariant under permutation, so it can never gate a layout convention. Every
   fixture now carries nonzero values (or synthesizes them) in each
   permute/fold/pack/split target.

These are enforced by a class audit: after any convention bug, the whole class
of loaders is re-checked, because the class is usually larger than the single
find.

## Further reading

- The per-loader godoc in package `nlp` carries the full source-verified
  convention bullets for each architecture.
- [`SPEC.md`](../SPEC.md) §R (the ggml quant formats) and §B67/§B68 (the
  verification lessons) hold the caveman-encoded record.
- Quantized serving performance is in [`docs/benchmarking.md`](benchmarking.md).
