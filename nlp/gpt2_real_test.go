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

// TestGPT2FromHFMatchesTransformers is the forward-parity anchor for GPT-2
// against a REAL transformers GPT2LMHeadModel (the existing test only compared to
// GoAI's own manual assembly). GPT-2 uses gelu_new (tanh-approx) where GoAI uses
// exact GELU, so a small residual at that level is expected.
func TestGPT2FromHFMatchesTransformers(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/gpt2_real.safetensors")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	model, err := nlp.GPT2FromHF(ts, 2)
	if err != nil {
		t.Fatalf("GPT2FromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/gpt2_real_logits.npy")
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
	t.Logf("GPT-2 logits max abs diff vs transformers: %.3e", maxAbs)
	if maxAbs > 5e-3 {
		t.Fatalf("GPT-2 diverges from transformers: %.3e", maxAbs)
	}
}
