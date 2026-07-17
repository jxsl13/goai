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

// TestMamba2FromHF is the forward-parity anchor for the Mamba-2 converter (§V16)
// and GoAI's state-space-duality model. A real transformers Mamba2ForCausalLM —
// token embedding → 2 × (RMSNorm → SSD mixer → residual) → RMSNorm → tied head —
// must reproduce its logits after Mamba2FromHF wires the fused in_proj split
// [z|xBC|dt], the depthwise conv, the per-head scalar A = −exp(A_log), the
// Δ = softplus(dt+dt_bias) discretization, the grouped B/C, the D skip and the
// gated RMSNorm into the per-head [nn.SSDRecurrent] scan. The golden uses
// num_heads=4, head_dim=8, n_groups=2, state_size=8 so the head→group sharing
// (heads_per_group = 2) is genuinely exercised.
func TestMamba2FromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/mamba2_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.Mamba2FromHF(ts, nlp.Mamba2Config{N: 8, NGroups: 2, Eps: 1e-5})
	if err != nil {
		t.Fatalf("Mamba2FromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/mamba2_hf_logits.npy")
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
	t.Logf("Mamba2 max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Mamba2 diverges from transformers: %.3e", maxAbs)
	}
}
