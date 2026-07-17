// Package nlp builds transformer language models — the components that turn a
// stream of token ids into next-token predictions, and the decoding strategies
// that turn those predictions back into text.
//
// # For AI practitioners
//
// nlp sits at layer L4 on top of the tensor/backend/autograd/nn stack. It
// provides the pieces a GPT- or Llama-family model is made of and the machinery
// to run and train them:
//
//   - Tokenization: a byte-level BPE (byte-pair encoding — merging frequent character pairs into tokens) Tokenizer (GPT-2 compatible, encode
//     bit-exact vs tiktoken, decode∘encode byte-exact under fuzzing, §T37), a GPT-2/HF (HuggingFace)
//     byte-level BPETokenizer, a Unigram tokenizer (SentencePiece, 1-best Viterbi over
//     a scored vocabulary, §T108), and BERT's WordPiece — loadable from a GGUF (llama.cpp’s single-file model format) model's
//     embedded vocabulary (BPEFromGGUF / UnigramFromGGUF) or from HuggingFace
//     tokenizer JSON, so a .gguf or HF checkpoint tokenizes end-to-end.
//   - Attention: fused multi-head attention (MHA) with a hand-derived SDPA VJP,
//     grouped/multi-query variants (GQA/MQA), causal masking, ALiBi and
//     sliding-window biases, and a KV (key/value attention cache)-cache for O(1)-per-step incremental decode.
//   - Models: the GPT decoder (pre-LN, causal MHA, GELU FFN) and the Llama/Llama-2
//     decoder (pre-norm RMSNorm, RoPE queries/keys, grouped-query attention, SwiGLU,
//     no biases), assembled from the same verified primitives and loadable straight
//     from safetensors weights or GGUF files — including quantized GGUF, which
//     QuantLlama decodes directly from the ggml Q-blocks without dequantizing to f32.
//     Beyond the autoregressive decoders: DiffusionLM, a LLaDA-style masked-diffusion
//     model that generates non-autoregressively by iterative unmasking (bidirectional,
//     not causal), CLA, cross-layer attention that shares one group's keys/values
//     across its layers to shrink the decode KV cache, and BLT, the Byte Latent
//     Transformer — a tokenizer-free model over raw bytes that entropy-patches the
//     byte stream and runs a latent transformer over the patches.
//   - Loading & adapting real Hugging Face checkpoints: converters that build the
//     above models straight from a HF checkpoint — thirty-one transformers-anchored
//     architectures spanning attention transformers, state-space models, and
//     linear-attention recurrence. Decoders through the Llama loader: LlamaFromHF
//     (Llama/Mistral); Qwen2/Qwen2.5 (adds q/k/v bias); Qwen3 (per-head QK-norm, no
//     bias); Phi3FromHF (packed qkv / gate-up); GraniteFromHF (four scalar
//     multipliers). Decoders as dedicated types: GemmaFromHF (√dim scale, (1+w)
//     RMSNorm, GeGLU); Gemma2FromHF (sandwich norms + logit soft-caps); OLMo2FromHF
//     (post-norm, full-width QK-norm); CohereFromHF (parallel residual, LayerNorm,
//     interleaved-RoPE via load permutation, logit scale); GPTNeoXFromHF (parallel
//     residual, packed qkv, partial rotary); StableLMFromHF (LayerNorm, partial
//     rotary); PhiFromHF (Phi-1/2); StarCoder2FromHF (LayerNorm, biased, GELU MLP);
//     NemotronFromHF (LayerNorm1P, ReLU² MLP); MPTFromHF (ALiBi position bias, no
//     rotary); FalconFromHF (multi-query attention, fused qkv); DeepSeekV2FromHF
//     (Multi-head Latent Attention + un-normalized MoE with shared experts). Sparse
//     Mixture-of-Experts: MixtralFromHF, Qwen3MoeFromHF, Qwen2MoeFromHF (shared
//     expert) and GraniteMoeFromHF — the KV-cached decode of every MoE evaluates only
//     the routed top-k experts (SparseMoE.ForwardDecode). Recurrent, non-transformer:
//     MambaFromHF and Mamba2FromHF (selective-scan state-space models) and RWKVFromHF
//     (WKV linear attention). Hybrid: JambaFromHF (interleaved Mamba/NoPE-attention
//     mixers with MoE/dense FFNs). Encoders: BertFromHF, RobertaFromHF,
//     DistilBertFromHF; and T5FromHF + T5DecoderFromHF (the full seq2seq
//     encoder–decoder). Each anchored bit- or tolerance-exact against transformers,
//     and each decoder supports generation with its architecture-native serving
//     mechanics: batched Prefill (one block-stack pass seeds the KV-cache,
//     bit-identical to stepwise decode), amortized-O(T) KV-cache growth, sparse
//     top-k MoE decode, O(1) constant-size state for the recurrent families,
//     Jamba's hybrid KV+SSM state, and DeepSeek-V2's absorbed-latent cache
//     (6.7× less KV memory). Weights load from safetensors, from .pt/.bin via the safe
//     no-code-execution PyTorch loader (package format/pytorch), or — for the most
//     common quantized community checkpoints — from llama.cpp GGUF files
//     (LlamaFromGGUF, Qwen2FromGGUF, Qwen3FromGGUF, GemmaFromGGUF, Phi3FromGGUF,
//     MixtralFromGGUF, StarCoder2FromGGUF, GPTNeoXFromGGUF, MPTFromGGUF,
//     FalconFromGGUF, StableLMFromGGUF, OLMo2FromGGUF — each verified
//     against that converter's layout conventions in llama.cpp source — plus
//     quantized decode straight from the ggml Q-blocks, no dequantized copy of
//     the weights ever materialized: QuantLlamaFromGGUF, QuantQwen2FromGGUF,
//     QuantQwen3FromGGUF, QuantPhi3FromGGUF (unpacks the packed qkv/gate-up
//     projections by lossless quantized row-slicing), QuantGemmaFromGGUF (a
//     dedicated quantized twin of the Gemma type — √dim scale, pre-folded (1+γ)
//     norms, tied head served from the same Q-block bytes) and
//     QuantMixtralFromGGUF (sparse top-k MoE decode; the llama-arch q/k row
//     interleave and the fused 3-D expert tensors are undone losslessly on the
//     quantized bytes)); LlamaConfigFromHF
//     (also Qwen2/Qwen3/Phi-3), GemmaConfigFromHF, Gemma2ConfigFromHF,
//     MixtralConfigFromHF (also Qwen3-MoE), GraniteConfigFromHF and BertConfigFromHF
//     read the checkpoint's config.json so loading is config-driven. Because a loaded model's forward runs through the
//     autograd choke-point, it is not just runnable but trainable: fine-tune it
//     directly, attach parameter-efficient LoRA adapters (ApplyLoRAGPT / ApplyLoRABert,
//     base frozen), or wrap it in BertClassifier for sequence classification.
//   - Sampling: a Sampler implementing the standard HuggingFace pipeline
//     (temperature → top-k → top-p nucleus → min-p → epsilon/eta → locally-typical →
//     multinomial), Mirostat adaptive sampling, and repetition / frequency / presence
//     logit penalties — all behind the TokenSampler interface that every sequential
//     generation loop accepts.
//   - Search & contrast: beam search and Diverse Beam Search, contrastive search
//     (degeneration penalty), contrastive decoding (expert−amateur logits), DoLa
//     layer-contrast decoding (DoLaDecode, via early-exit layer logits), and
//     Classifier-Free Guidance with negative prompts (CFGDecode).
//   - Constrained & marked output: regex/FSM-guided constrained decoding and
//     red/green-list LLM watermarking with statistical detection.
//   - Accelerated decoding: lossless speculative decoding with a draft model,
//     draft-model-free prompt-lookup (n-gram), Medusa multi-head drafting
//     (trainable MedusaHeads over ForwardHidden + MedusaGenerate with typical
//     acceptance; MedusaGenerateTree verifies a topK candidate tree in ONE
//     masked forward; the GPU loop lives in llamagpu), EAGLE feature-level
//     autoregressive drafting (EagleHead over ForwardHidden + lossless
//     EagleGenerate), and Jacobi parallel decoding (JacobiDecode /
//     GPT.JacobiGenerate).
//   - Long-context inference: attention-sink streaming (StreamGenerate), bounded
//     KV-cache eviction policies, SnapKV prompt compression, PyramidKV per-layer
//     cache budgets, an 8-bit quantized KV-cache, and Self-Extend length
//     extrapolation (Llama.SelfExtendForward / SelfExtendGenerate — grouped
//     attention, no fine-tuning; generation stays coherent at 4× training length).
//   - Training objectives & data: BERT masked-LM (language model), T5 span corruption, UL2
//     mixture-of-denoisers, Fill-in-the-Middle transformation, and sequence packing.
//
// Every algorithm is validated on the §V16 ladder: tier-1 bit- or tolerance-exact
// parity against an official reference (torch / tiktoken / gguf), tier-2 the
// defining paper, cited in SPEC.md §R. Forward and backward paths are checked
// against real PyTorch at f64 rtol ~1e-12 (inference) and ~1e-9 (gradients).
//
// # For everyone else
//
// A language model reads text as a list of numbers ("tokens"), and for each
// position it outputs a score for every possible next token — think of a giant
// weighted dice with one face per word-piece, re-weighted after every token.
// This package contains both halves of that process:
//
//   - the model that produces the scores (attention, the mechanism that lets each
//     word look back at the earlier words, plus the surrounding layers), and
//   - the "decoder" that chooses the next token from the scores. Choosing always
//     the single highest-scoring token is greedy decoding; adding a little
//     controlled randomness (temperature, top-k, top-p, min-p) makes the output
//     more varied and natural. Beam search explores several candidate
//     continuations at once; speculative decoding uses a small fast model to
//     guess ahead and a large model to check the guesses, going faster without
//     changing the result.
//
// The runnable Example functions below show each of these at three levels: a
// one-line trivial call, a realistic use case, and a piece embedded in a larger
// decode pipeline.
//
// Further reading: Jurafsky & Martin, "Speech and Language Processing" (3rd ed.) for language modeling and decoding foundations; Mielke et al. 2021, "Between Words and Characters", for the tokenization landscape; Holtzman et al. 2019, "The Curious Case of Neural Text Degeneration", for why sampling strategies matter.
//
// In plain terms: everything language — tokenizers that turn text into numbers, transformer models that continue a sequence, and the decoding strategies that turn model probabilities back into readable text.
package nlp
