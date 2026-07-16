package nlp

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/tensor"
)

// TestPartialRoPE anchors the partial-rotary helper (seq=5, heads=2, head_dim=8,
// rotary_dim=4): only the first 4 of each head's 8 channels are rotated (split-half),
// the rest pass through. The GPT-NeoX convention (split-half rotate_half, rotary_dim <
// head_dim) was verified against the transformers source; the reference is that convention
// recomputed in f64 (transformers' own GPTNeoXRotaryEmbedding computes cos/sin in f32, so
// a direct golden matches only to ~1e-7 — this f64 reference isolates the helper's own
// exactness). This is the primitive that unblocks the GPT-NeoX / StableLM / Phi families.
func TestPartialRoPE(t *testing.T) {
	qIn, err := npy.LoadFile("testdata/prope_q.npy")
	if err != nil {
		t.Skip("partial-rope reference not generated")
	}
	want, err := npy.LoadFile("testdata/prope_out.npy")
	if err != nil {
		t.Fatal(err)
	}
	got, err := partialRoPE(backend.NewContext(), qIn, 2, 4, backend.RoPEAttrs{Base: 10000})
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for i := range want.Shape()[0] {
		for j := range want.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - want.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("partial-rope max abs diff vs transformers GPT-NeoX: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("partial-rope diverges from transformers: %.3e", maxAbs)
	}

	// rotaryDim == headDim must reduce to a full OpRoPE.
	full, err := partialRoPE(backend.NewContext(), qIn, 2, 8, backend.RoPEAttrs{Base: 10000})
	if err != nil {
		t.Fatal(err)
	}
	ref, err := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{qIn}, backend.RoPEAttrs{Base: 10000, Heads: 2})
	if err != nil {
		t.Fatal(err)
	}
	var fd float64
	for i := range ref[0].Shape()[0] {
		for j := range ref[0].Shape()[1] {
			if d := math.Abs(full.AtF64(i, j) - ref[0].AtF64(i, j)); d > fd {
				fd = d
			}
		}
	}
	if fd != 0 {
		t.Fatalf("partialRoPE(rotaryDim==headDim) != full OpRoPE: %.3e", fd)
	}
}
