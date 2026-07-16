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

// TestT5FromHFMatchesTransformers is the forward-parity anchor (§V16) for the T5
// encoder: a real transformers T5EncoderModel's weights loaded through T5FromHF
// must reproduce its last_hidden_state — exercising the relative-position bias,
// the unscaled attention, RMSNorm and the gated-GELU FFN.
func TestT5FromHFMatchesTransformers(t *testing.T) {
	cases := []struct {
		name, weights, hidden string
	}{
		{"v1.1-gated-gelu", "testdata/t5_hf.safetensors", "testdata/t5_hf_hidden.npy"},
		{"v1.0-relu", "testdata/t5v10_hf.safetensors", "testdata/t5v10_hf_hidden.npy"},
	}
	tokens := []int{3, 7, 1, 9, 4, 2, 8}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ts, _, err := safetensors.LoadFile(c.weights)
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			model, err := nlp.T5FromHF(ts, nlp.T5Config{Heads: 2, HeadDim: 8, Eps: 1e-6})
			if err != nil {
				t.Fatalf("T5FromHF: %v", err)
			}
			golden, err := npy.LoadFile(c.hidden)
			if err != nil {
				t.Fatal(err)
			}
			got, err := model.Forward(backend.NewContext(), tokens)
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
			t.Logf("%s: max abs hidden-state diff vs transformers: %.3e", c.name, maxAbs)
			if maxAbs > 2e-3 {
				t.Fatalf("%s diverges from transformers: %.3e", c.name, maxAbs)
			}
		})
	}
}
