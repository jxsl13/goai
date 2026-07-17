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

// TestMambaFromHF is the forward-parity anchor for the Mamba converter (§V16) and
// GoAI's FIRST non-transformer (state-space) model. A real transformers
// MambaForCausalLM — token embedding → 2 × (RMSNorm → selective-scan mixer →
// residual) → RMSNorm → tied head — must reproduce its logits after MambaFromHF
// wires the in_proj/x_proj splits, the dt_proj Δ-bias, A = −exp(A_log), the D skip
// and the SiLU(z) gate into nn.MambaBlock. The golden exercises dt_rank ≠
// ceil(d_model/16) (4 vs 1), so the split offsets are genuinely tested.
func TestMambaFromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/mamba_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		t.Fatalf("MambaFromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/mamba_hf_logits.npy")
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
	t.Logf("Mamba max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Mamba diverges from transformers: %.3e", maxAbs)
	}
}
