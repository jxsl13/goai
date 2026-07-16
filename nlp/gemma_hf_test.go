package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

// TestGemmaFromHF is the forward-parity anchor for the Gemma converter (§V16):
// a real transformers GemmaForCausalLM's weights loaded through GemmaFromHF must
// reproduce that model's logits. The golden uses a DECOUPLED head_dim (12 with
// heads=2, hidden=16 → 24≠16) and GQA (kv=1), exercising Gemma's distinctive
// geometry. The only systematic (non-round-off) error source is GoAI's exact
// OpGELU vs Gemma's tanh-approx gelu_pytorch_tanh — negligible at this activation
// scale (~round-off here); the 2e-3 gate leaves headroom for larger-activation models.
func TestGemmaFromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gemma_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.GemmaFromHF(ts, nlp.GemmaConfig{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("GemmaFromHF: %v", err)
	}
	if model.Config.HeadDim != 12 {
		t.Fatalf("decoupled head_dim not inferred: got %d want 12", model.Config.HeadDim)
	}
	golden, err := npy.LoadFile("testdata/gemma_hf_logits.npy")
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
	t.Logf("Gemma max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Gemma diverges from transformers: %.3e (exact-vs-tanh GELU residual too large?)", maxAbs)
	}
}
