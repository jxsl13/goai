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

// TestPhi3FromHF is the forward-parity anchor for the Phi-3 converter (§V16): a
// real transformers Phi3ForCausalLM — whose attention/FFN input projections are
// row-packed (qkv_proj, gate_up_proj) — must reproduce its logits after Phi3FromHF
// unpacks them and loads the result as a plain Llama. The golden uses GQA
// (heads=4, kv=2) so the k/v split offsets are exercised, not just an even thirds.
func TestPhi3FromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/phi3_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.Phi3FromHF(ts, nlp.LlamaConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("Phi3FromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/phi3_hf_logits.npy")
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
	t.Logf("Phi-3 max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Phi-3 diverges from transformers: %.3e", maxAbs)
	}
}
