package nlp_test

import (
	"fmt"
	"os"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

// The examples in this file load the tiny randomly-initialized HuggingFace
// checkpoints committed under testdata/ (the same goldens the parity tests use).
// They demonstrate the load → Forward / Generate API workflow for each
// architecture family; because the weights are random, generated token ids are
// meaningless — deterministic, but not language.

// LlamaFromHF is the loader for the Llama family — RMSNorm, RoPE positions and a
// SwiGLU MLP, the template most modern decoder LLMs follow. Weights come from a
// HuggingFace safetensors checkpoint; layer count, dims and vocab are inferred
// from the tensors, while head layout and RoPE come from the config. Forward
// returns next-token logits for every prompt position: [seq, vocab].
func ExampleLlamaFromHF() {
	ts, _, err := safetensors.LoadFile("testdata/llama_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.LlamaFromHF(ts, nlp.LlamaConfig{
		Heads: 2, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		panic(err)
	}
	logits, err := model.Forward(backend.NewContext(), []int{1, 4, 7, 2, 9, 0, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape()) // [seq, vocab]
	// Output: (7, 16)
}

// LlamaConfigFromHF parses a HuggingFace config.json into a LlamaConfig, so a
// checkpoint's own metadata — head counts, norm epsilon, RoPE base, context —
// drives the loader instead of hand-copied numbers. Pair it with LlamaFromHF for
// fully config-driven loading of any Llama-family checkpoint.
func ExampleLlamaConfigFromHF() {
	b, err := os.ReadFile("testdata/llama_tiny_config.json")
	if err != nil {
		panic(err)
	}
	cfg, err := nlp.LlamaConfigFromHF(b)
	if err != nil {
		panic(err)
	}
	fmt.Println(cfg.Heads, cfg.KVHeads, cfg.Ctx)
	// Output: 2 2 32
}

// Gemma2FromHF loads Google's Gemma 2 — a Llama-style decoder with extra
// stabilizers: four "sandwich" RMSNorms per block, a query_pre_attn_scalar score
// scale, and soft-caps that squash attention scores and final logits through
// tanh. All of them come from the HF config; head_dim is inferred from the
// weights. The tiny random checkpoint demonstrates the workflow only.
func ExampleGemma2FromHF() {
	ts, _, err := safetensors.LoadFile("testdata/gemma2_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.Gemma2FromHF(ts, nlp.Gemma2Config{
		Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		QueryPreAttnScalar: 8, AttnLogitCap: 1.0, FinalLogitCap: 2.0,
	})
	if err != nil {
		panic(err)
	}
	logits, err := model.Forward(backend.NewContext(), []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape()) // [seq, vocab]
	// Output: (7, 32)
}

// Mixtral is the sparse Mixture-of-Experts family: Llama attention, but each
// block's MLP is replaced by several expert MLPs of which a router activates
// only the top-k per token — big-model quality at a fraction of the compute.
// Generate runs KV-cached greedy decoding: the prompt is echoed first, then
// maxNew tokens follow. The tiny model's random weights make the continuation
// meaningless ids — but deterministic, since greedy decoding has no randomness.
func ExampleMixtral_Generate() {
	ts, _, err := safetensors.LoadFile("testdata/mixtral_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.MixtralFromHF(ts, nlp.MixtralConfig{
		Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		panic(err)
	}
	out, err := model.Generate([]int{3, 7, 1}, 3 /*maxNew*/, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Println(out)
	// Output: [3 7 1 0 3 0]
}

// DeepSeek-V2 replaces classic multi-head attention with Multi-head Latent
// Attention (MLA): keys and values are compressed through a low-rank latent, so
// the decode cache stores one small latent per token instead of full K/V.
// GenerateLatent uses the absorbed-MLA cache — the mathematically identical but
// memory-lean decode path — for greedy generation (random weights, meaningless
// but deterministic ids).
func ExampleDeepSeekV2_GenerateLatent() {
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2moe_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.DeepSeekV2FromHF(ts, nlp.DeepSeekV2Config{
		Heads: 4, QLoraRank: 24, KVLoraRank: 16,
		QKNope: 16, QKRope: 8, VHead: 16,
		Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		FirstKDense: 1, NRoutedExperts: 4, NSharedExperts: 1,
		TopK: 2, MoEHidden: 32, RoutedScale: 1.0,
	})
	if err != nil {
		panic(err)
	}
	out, err := model.GenerateLatent([]int{3, 7, 1}, 3 /*maxNew*/, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Println(out)
	// Output: [3 7 1 24 28 26]
}

// Mamba is a state-space model (SSM), not a transformer: a selective scan
// carries a fixed-size recurrent state across the sequence, so decoding needs
// O(1) memory and time per token — no KV cache growing with context. MambaFromHF
// infers every dimension from the weights; only the norm epsilon is supplied.
// Greedy Generate on the tiny random checkpoint is deterministic API demo output.
func ExampleMamba_Generate() {
	ts, _, err := safetensors.LoadFile("testdata/mamba_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		panic(err)
	}
	out, err := model.Generate([]int{3, 7, 1}, 3 /*maxNew*/, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Println(out)
	// Output: [3 7 1 1 1 1]
}

// RWKV blends transformer training with RNN inference: its WKV time-mixing is a
// linear-attention recurrence, so like Mamba it decodes with a constant-size
// state instead of a growing KV cache. RWKVFromHF wires the time-decay/
// time-first parameters and token-shift interpolators from a HuggingFace RWKV-4
// checkpoint; greedy Generate on random weights yields deterministic demo ids.
func ExampleRWKV_Generate() {
	ts, _, err := safetensors.LoadFile("testdata/rwkv_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.RWKVFromHF(ts, nlp.RWKVConfig{Eps: 1e-5})
	if err != nil {
		panic(err)
	}
	out, err := model.Generate([]int{3, 7, 1}, 3 /*maxNew*/, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Println(out)
	// Output: [3 7 1 16 4 23]
}

// Jamba is the hybrid family: the stack interleaves Mamba (SSM) mixer layers
// with GQA attention layers, and dense SwiGLU MLPs with sparse-MoE MLPs — the
// layer types are detected from the checkpoint's tensor names. Generate
// threads both cache kinds (KV rows for attention layers, recurrent state for
// Mamba layers) through one decode loop (random weights → deterministic demo ids).
func ExampleJamba_Generate() {
	ts, _, err := safetensors.LoadFile("testdata/jamba_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.JambaFromHF(ts, nlp.JambaConfig{Heads: 4, TopK: 2, Eps: 1e-6})
	if err != nil {
		panic(err)
	}
	out, err := model.Generate([]int{3, 7, 1}, 3 /*maxNew*/, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Println(out)
	// Output: [3 7 1 25 30 30]
}

// MPT is the ALiBi family: instead of rotary position embeddings it biases each
// attention score by a head-specific slope times the key distance, which lets
// the model generalize to sequences longer than it was trained on. No RoPE
// parameters are needed — just heads, norm epsilon and context.
func ExampleMPTFromHF() {
	ts, _, err := safetensors.LoadFile("testdata/mpt_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.MPTFromHF(ts, nlp.MPTConfig{Heads: 4, Eps: 1e-5, Ctx: 32})
	if err != nil {
		panic(err)
	}
	logits, err := model.Forward(backend.NewContext(), []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape()) // [seq, vocab]
	// Output: (7, 32)
}

// IBM Granite is a Llama with four scalar multipliers — on the embedding, the
// attention scores, every residual branch, and the final logits — that keep
// activations in range at depth. GraniteFromHF reuses the whole Llama machinery;
// the scalars ride in on the ordinary LlamaConfig, so everything Llama can do
// (Generate, DecodeStep, streaming) works on a Granite checkpoint too.
func ExampleGraniteFromHF() {
	ts, _, err := safetensors.LoadFile("testdata/granite_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.GraniteFromHF(ts, nlp.LlamaConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		EmbeddingMult: 12, AttentionMult: 0.5, ResidualMult: 0.22, LogitsScale: 8,
	})
	if err != nil {
		panic(err)
	}
	logits, err := model.Forward(backend.NewContext(), []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape()) // [seq, vocab]
	// Output: (7, 32)
}
