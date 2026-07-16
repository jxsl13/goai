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
