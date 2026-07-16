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

// TestMixtralFromHF is the forward-parity anchor for the Mixtral converter (§V16):
// a real transformers MixtralForCausalLM — Llama attention + a top-2 sparse-MoE FFN
// with row-packed fused experts — must reproduce its logits after MixtralFromHF
// loads the router and unpacks each expert into nn.SparseMoE. The golden uses GQA
// (heads=4, kv=2) and 4 experts / top-2, so the routing and k/v splits are exercised.
func TestMixtralFromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/mixtral_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.MixtralFromHF(ts, nlp.MixtralConfig{
		Heads: 4, KVHeads: 2, TopK: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("MixtralFromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/mixtral_hf_logits.npy")
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
	t.Logf("Mixtral max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Mixtral diverges from transformers: %.3e", maxAbs)
	}
}
