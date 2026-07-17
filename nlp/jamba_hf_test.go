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

// TestJambaFromHF is the forward-parity anchor for the Jamba converter (§V16) —
// the HYBRID capstone that interleaves every mixer/FFN family GoAI supports. The
// golden transformers JambaForCausalLM has 4 layers with attn_layer_period=2/
// offset=1 and expert_layer_period=2/offset=0, so layers 0,2 are Mamba mixers
// with sparse-MoE FFNs and layers 1,3 are NoPE GQA attention with dense SwiGLU
// FFNs — all four component families are exercised in one stack. Parity proves
// the key-driven layer typing, the NoPE attention (no rotary), Jamba's dt/b/c
// RMSNorms inside the Mamba mixer, and the un-renormalized top-k MoE routing.
func TestJambaFromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/jamba_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.JambaFromHF(ts, nlp.JambaConfig{Heads: 4, TopK: 2, Eps: 1e-6})
	if err != nil {
		t.Fatalf("JambaFromHF: %v", err)
	}

	// Sanity: the hybrid stack must contain every component family.
	var nAttn, nMamba, nMoE, nDense int
	for _, l := range model.Layers {
		if l.Attn != nil {
			nAttn++
		}
		if l.Mamba != nil {
			nMamba++
		}
		if l.MoE != nil {
			nMoE++
		}
		if l.Dense != nil {
			nDense++
		}
	}
	if nAttn == 0 || nMamba == 0 || nMoE == 0 || nDense == 0 {
		t.Fatalf("hybrid stack incomplete: attn=%d mamba=%d moe=%d dense=%d", nAttn, nMamba, nMoE, nDense)
	}
	if model.Config.KVHeads != 2 {
		t.Fatalf("inferred KVHeads = %d, want 2", model.Config.KVHeads)
	}

	golden, err := npy.LoadFile("testdata/jamba_hf_logits.npy")
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
	t.Logf("Jamba max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("Jamba diverges from transformers: %.3e", maxAbs)
	}
}
