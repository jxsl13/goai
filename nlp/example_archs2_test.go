package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// The examples in this file continue example_archs_test.go's roster: they load the
// same kind of tiny randomly-initialized HuggingFace checkpoints committed under
// testdata/ and demonstrate the load -> Forward/Generate workflow for architecture
// families that don't yet have one there. Random weights make the numbers
// meaningless — deterministic, but not language.

// Bert is the bidirectional transformer-encoder family: BERT, RoBERTa and
// DistilBERT all share this type (see [BertFromHF], [RobertaFromHF],
// [DistilBertFromHF]) since they differ only in embedding/tensor-naming details the
// loaders absorb. Unlike the causal [Llama]/[GPT] decoders, every position attends
// to every other position — the shape retrieval and sentence-embedding tasks want.
// Forward returns the last hidden state [seq, dim]; pool it with [MeanPool] for a
// sentence embedding (see [ExampleMeanPool]).
func ExampleBert() {
	ts, _, err := safetensors.LoadFile("testdata/bert_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.BertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		panic(err)
	}
	tokens := []int{1, 5, 8, 3, 9, 2, 7}
	segments := []int{0, 0, 0, 1, 1, 1, 1} // two segments, e.g. sentence-pair input
	hidden, err := model.Forward(backend.NewContext(), tokens, segments)
	if err != nil {
		panic(err)
	}
	fmt.Println(hidden.Shape()) // [seq, dim]
	// Output: (7, 16)
}

// BertClassifier packages a [Bert] encoder plus a linear head for the common
// "fine-tune a loaded HF encoder for classification" task: Logits mean-pools the
// encoder's last hidden state and applies the head to produce class scores.
func ExampleBertClassifier() {
	ts, _, err := safetensors.LoadFile("testdata/bert_hf.safetensors")
	if err != nil {
		panic(err)
	}
	bert, err := nlp.BertFromHF(ts, nlp.BertConfig{Heads: 2, Eps: 1e-12})
	if err != nil {
		panic(err)
	}
	clf := nlp.NewBertClassifier(bert, 2 /* classes */, 3 /* seed */)
	logits, err := clf.Logits(backend.NewContext(), []int{1, 5, 8, 3}, nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape()) // [1, classes]
	// Output: (1, 2)
}

// Cohere is Command-R's architecture: a PARALLEL residual block (one input norm
// feeds both attention and the SwiGLU FFN, their outputs summed onto the raw
// stream), a mean-centered weight-only LayerNorm (not RMSNorm), interleaved RoPE,
// GQA, and a tied LM head scaled by LogitScale. Generate runs KV-cached greedy
// decoding: the prompt is echoed first, then maxNew tokens follow.
func ExampleCohere() {
	ts, _, err := safetensors.LoadFile("testdata/cohere_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.CohereFromHF(ts, nlp.CohereConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, LogitScale: 0.0625, Ctx: 32,
	})
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

// Falcon exercises the SINGLE-norm parallel residual (one input_layernorm feeds
// BOTH attention and the MLP, both summed onto the raw stream), a full LayerNorm
// WITH bias, and the fused multi-query attention split (all query heads, then one
// shared key head, then one shared value head).
func ExampleFalcon() {
	ts, _, err := safetensors.LoadFile("testdata/falcon_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.FalconFromHF(ts, nlp.FalconConfig{
		Heads: 4, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
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

// GPTNeoX (the Pythia family) pairs a PARALLEL residual with TWO separate norms
// (one feeding attention, one feeding the MLP) with PARTIAL rotary position
// embeddings — RotaryPct selects how much of each head rotates, the rest passing
// through unrotated.
func ExampleGPTNeoX() {
	ts, _, err := safetensors.LoadFile("testdata/gptneox_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.GPTNeoXFromHF(ts, nlp.GPTNeoXConfig{
		Heads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32,
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

// Gemma is Google's Llama-style decoder with a DECOUPLED head_dim (inferred from
// the weights rather than Dim/Heads) and a tied embedding/output matrix. Mutating
// TokEmb in place — e.g. an optimizer step during fine-tuning — invalidates the
// cached tied-head transpose Forward/DecodeStep read; RefreshTiedHead drops that
// cache so the next call recomputes from the updated weights.
func ExampleGemma_RefreshTiedHead() {
	ts, _, err := safetensors.LoadFile("testdata/gemma_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.GemmaFromHF(ts, nlp.GemmaConfig{
		Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		panic(err)
	}
	// Simulate a fine-tuning step mutating the tied embedding in place.
	model.TokEmb.SetF64(model.TokEmb.AtF64(0, 0)+0.01, 0, 0)
	model.RefreshTiedHead()
	logits, err := model.Forward(backend.NewContext(), []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape()) // [seq, vocab]
	// Output: (7, 32)
}

// Gemma2 adds four "sandwich" RMSNorms per block, a query_pre_attn_scalar score
// scale, and tanh soft-caps on both attention scores and final logits, on top of
// Gemma's tied head. RefreshTiedHead drops the cached tied-head transpose after the
// embedding is mutated in place (fine-tuning); see [ExampleGemma_RefreshTiedHead].
func ExampleGemma2_RefreshTiedHead() {
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
	model.TokEmb.SetF64(model.TokEmb.AtF64(0, 0)+0.01, 0, 0)
	model.RefreshTiedHead()
	logits, err := model.Forward(backend.NewContext(), []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape()) // [seq, vocab]
	// Output: (7, 32)
}

// GraniteMoE composes IBM Granite's four scalar multipliers (embedding, attention,
// residual and logits) with a sparse top-k Mixture-of-Experts FFN over a Llama-style
// GQA attention stack.
func ExampleGraniteMoE() {
	ts, _, err := safetensors.LoadFile("testdata/granitemoe_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.GraniteMoeFromHF(ts, nlp.GraniteMoeConfig{
		Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
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

// Mamba2 is the state-space-duality successor to Mamba: NewDecodeState allocates the
// O(1) recurrent state (a fixed-size conv window plus per-head SSD hidden state)
// that DecodeStep advances one token at a time — no KV cache growing with context.
func ExampleMamba2_NewDecodeState() {
	ts, _, err := safetensors.LoadFile("testdata/mamba2_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.Mamba2FromHF(ts, nlp.Mamba2Config{N: 8, NGroups: 2, Eps: 1e-5})
	if err != nil {
		panic(err)
	}
	ctx := backend.NewContext()
	st := model.NewDecodeState()
	var last *tensor.Tensor
	for _, tok := range []int{3, 7, 1} {
		if last, err = model.DecodeStep(ctx, st, tok); err != nil {
			panic(err)
		}
	}
	fmt.Println(last.Shape()) // [1, vocab]
	// Output: (1, 32)
}

// Nemotron pairs LayerNorm1P (gain (1+weight), WITH bias — not RMSNorm, not
// weight-only) with a ReLU² MLP (down(relu²(up(x))), no gate) and a PARTIAL rotary
// (only RotaryPct of each head rotates).
func ExampleNemotron() {
	ts, _, err := safetensors.LoadFile("testdata/nemotron_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.NemotronFromHF(ts, nlp.NemotronConfig{
		Heads: 4, KVHeads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32,
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

// OLMo2 uses a POST-norm residual (attention/FFN read the raw stream; a norm then
// scales the sublayer output before the residual add — the reverse of Llama's
// pre-norm) plus full-width q_norm/k_norm applied before RoPE.
func ExampleOLMo2() {
	ts, _, err := safetensors.LoadFile("testdata/olmo2_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.OLMo2FromHF(ts, nlp.OLMo2Config{
		Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
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

// OLMoE is OLMo's sparse-MoE sibling: the standard Llama pre-norm residual, the
// same full-width QK-norm as OLMo2, and a top-k fused Mixture-of-Experts FFN in
// place of the dense SwiGLU.
func ExampleOLMoE() {
	ts, _, err := safetensors.LoadFile("testdata/olmoe_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.OLMoEFromHF(ts, nlp.OLMoEConfig{
		Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
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

// Phi (the original Phi-1/1.5/2 family, distinct from the Phi-3 GGUF path) has a
// SINGLE-norm parallel residual, separate biased q/k/v/dense projections, a PARTIAL
// rotary and a biased, untied lm_head.
func ExamplePhi() {
	ts, _, err := safetensors.LoadFile("testdata/phi_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.PhiFromHF(ts, nlp.PhiConfig{
		Heads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32,
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

// Qwen2MoE is Qwen2's biased attention (q/k/v carry a bias, o does not) plus a
// top-k sparse MoE FFN AND a sigmoid-gated shared expert that always contributes.
func ExampleQwen2MoE() {
	ts, _, err := safetensors.LoadFile("testdata/qwen2moe_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.Qwen2MoeFromHF(ts, nlp.Qwen2MoeConfig{
		Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
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

// StableLM has a biased LayerNorm (not RMSNorm) in a SEQUENTIAL two-norm residual,
// a PARTIAL rotary, GQA and a SwiGLU FFN.
func ExampleStableLM() {
	ts, _, err := safetensors.LoadFile("testdata/stablelm_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.StableLMFromHF(ts, nlp.StableLMConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32,
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

// StarCoder2 has a biased LayerNorm in a SEQUENTIAL two-norm residual, BIASED
// attention (q/k/v/o all carry a bias), FULL rotary and a biased 2-layer GELU MLP
// (not SwiGLU).
func ExampleStarCoder2() {
	ts, _, err := safetensors.LoadFile("testdata/starcoder2_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.StarCoder2FromHF(ts, nlp.StarCoder2Config{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
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

// T5Decoder is the causal half of a T5 encoder-decoder pair: self-attention with a
// learned relative-position bias, cross-attention over the [T5] encoder's output,
// and a ReLU (or gated) FFN. Decode returns the last hidden state [tgtSeq, dim];
// Logits projects that through the (possibly tied) LM head.
func ExampleT5Decoder() {
	ts, _, err := safetensors.LoadFile("testdata/t5full_hf.safetensors")
	if err != nil {
		panic(err)
	}
	cfg := nlp.T5Config{Heads: 2, HeadDim: 8, Eps: 1e-6}
	enc, err := nlp.T5FromHF(ts, cfg)
	if err != nil {
		panic(err)
	}
	dec, err := nlp.T5DecoderFromHF(ts, cfg)
	if err != nil {
		panic(err)
	}
	ctx := backend.NewContext()
	encOut, err := enc.Forward(ctx, []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		panic(err)
	}
	hidden, err := dec.Decode(ctx, encOut, []int{0, 5, 8, 2, 6})
	if err != nil {
		panic(err)
	}
	fmt.Println(hidden.Shape()) // [tgtSeq, dim]
	// Output: (5, 16)
}
