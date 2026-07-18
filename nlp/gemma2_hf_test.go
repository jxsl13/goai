package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// TestGemma2FromHF is the forward-parity anchor for the Gemma 2 converter (§V16): a
// real transformers Gemma2ForCausalLM's weights loaded through Gemma2FromHF must
// reproduce that model's logits. The golden exercises every Gemma 2-specific detail —
// the four sandwich RMSNorms, GQA (kv=1), the query_pre_attn_scalar score scale, and
// BOTH soft-caps (attn_logit_softcapping=1.0, final_logit_softcapping=2.0). The q/k
// projections in the golden are amplified so the pre-softmax scores reach O(3), where
// the attention-score soft-cap (cap=1.0) bites hard (it changes scores by up to ~2.2);
// removing that soft-cap moves the logits by ~1e-1, far above the 2e-3 gate, proving
// the golden is not vacuous. The only systematic (non-round-off) error source is
// GoAI's exact-erf OpGELU vs Gemma's tanh-approx gelu_pytorch_tanh.
func TestGemma2FromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma2_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.Gemma2FromHF(ts, nlp.Gemma2Config{
		Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		QueryPreAttnScalar: 8, AttnLogitCap: 1.0, FinalLogitCap: 2.0,
	})
	if err != nil {
		t.Fatalf("Gemma2FromHF: %v", err)
	}
	if model.Config.HeadDim != 8 {
		t.Fatalf("head_dim not inferred: got %d want 8", model.Config.HeadDim)
	}
	golden, err := npy.LoadFile("testdata/gemma2_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.Forward(backend.NewContext(), []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got.Shape()[0] != golden.Shape()[0] || got.Shape()[1] != golden.Shape()[1] {
		t.Fatalf("shape %v != golden %v", got.Shape(), golden.Shape())
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("Gemma2 max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Gemma2 diverges from transformers: %.3e", maxAbs)
	}
}

// gemma2HFTensors builds a minimal well-formed Gemma2ForCausalLM tensor map — every
// name Gemma2FromHF requires, at a toy geometry — for tests that care about the config
// handling rather than the numerics.
func gemma2HFTensors(layers, vocab, dim, heads, kv, headDim, ffn int) map[string]*tensor.Tensor {
	w := func(shape ...int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape(shape))
		for i := range t.Numel() {
			t.SetF64(0.01*float64(i%7)-0.03, tensor.Unravel(i, t.Shape())...)
		}
		return t
	}
	ts := map[string]*tensor.Tensor{
		"model.embed_tokens.weight": w(vocab, dim),
		"model.norm.weight":         w(dim),
	}
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		ts[p+"self_attn.q_proj.weight"] = w(heads*headDim, dim) // HF stores [out, in]
		ts[p+"self_attn.k_proj.weight"] = w(kv*headDim, dim)
		ts[p+"self_attn.v_proj.weight"] = w(kv*headDim, dim)
		ts[p+"self_attn.o_proj.weight"] = w(dim, heads*headDim)
		ts[p+"input_layernorm.weight"] = w(dim)
		ts[p+"post_attention_layernorm.weight"] = w(dim)
		ts[p+"pre_feedforward_layernorm.weight"] = w(dim)
		ts[p+"post_feedforward_layernorm.weight"] = w(dim)
		ts[p+"mlp.gate_proj.weight"] = w(ffn, dim)
		ts[p+"mlp.up_proj.weight"] = w(ffn, dim)
		ts[p+"mlp.down_proj.weight"] = w(dim, ffn)
	}
	return ts
}

// §V16 tier-1 (loader symmetry): Gemma2FromHF must clamp Ctx to the sliding window
// exactly as Gemma2FromGGUF does (TestGemma2FromGGUFSlidingWindowClamp). GoAI's Gemma2
// implements FULL attention only, which equals the real alternating sliding-window
// graph only while seq ≤ window; without the clamp an HF-loaded Gemma2 accepts prompts
// past the window and every SWA layer silently attends the whole prefix — wrong logits,
// plausible text, no error. The two loaders must not disagree about the same model.
func TestGemma2FromHFClampsCtxToSlidingWindow(t *testing.T) {
	ts := gemma2HFTensors(1, 12, 8, 2, 1, 4, 16)
	// A caller asking for far more context than Gemma 2's 4096 window must be clamped.
	m, err := nlp.Gemma2FromHF(ts, nlp.Gemma2Config{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 8192})
	if err != nil {
		t.Fatal(err)
	}
	if m.Config.Ctx != 4096 {
		t.Errorf("Ctx not clamped to the sliding window: got %d want 4096", m.Config.Ctx)
	}
	// Below the window nothing is clamped — the clamp is a ceiling, not an assignment.
	m2, err := nlp.Gemma2FromHF(ts, nlp.Gemma2Config{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatal(err)
	}
	if m2.Config.Ctx != 32 {
		t.Errorf("Ctx below the window must be left alone: got %d want 32", m2.Config.Ctx)
	}
}
