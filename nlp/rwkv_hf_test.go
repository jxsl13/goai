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

// TestRWKVFromHF is the forward-parity anchor for the RWKV-4 converter (§V16) and
// GoAI's third recurrent / linear-attention family (after Mamba and Retention/
// GLA). A real transformers RwkvForCausalLM — token embedding → block-0 pre_ln →
// 2 × (ln1 → WKV time-mixing → residual → ln2 → squared-ReLU channel-mixing →
// residual) → ln_out → untied head — must reproduce its logits after RWKVFromHF
// wires the time_decay/time_first WKV parameters, the time_mix token-shift
// interpolators, the ln0/ln1/ln2 LayerNorms and the two mixing sublayers into
// nn.RWKVBlock. It exercises the sign convention w = exp(WLog) ⇔ −exp(time_decay),
// the [1,1,dim]→[dim] time_mix squeeze, and the block-0-only pre_ln placement.
func TestRWKVFromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/rwkv_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.RWKVFromHF(ts, nlp.RWKVConfig{Eps: 1e-5})
	if err != nil {
		t.Fatalf("RWKVFromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/rwkv_hf_logits.npy")
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
	t.Logf("RWKV max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("RWKV diverges from transformers: %.3e", maxAbs)
	}
}
