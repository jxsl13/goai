package safetensors

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestLoadHalfFloatReference is the §B67 independent-fixture check for the F16 and
// BF16 read paths — the dtypes real HuggingFace checkpoints ship in.
//
// Every other .safetensors fixture in the tree is F32/F64, so GoAI's verbatim-U16
// read of the half floats was covered only by its own writer→reader round-trip. A
// symmetric round-trip proves nothing about the foreign convention (§B67): if
// GoAI's stored layout disagreed with what the reference `safetensors` library
// actually writes — byte order, which 16 bits, F16-vs-BF16 tag — a round-trip
// would still pass while every real checkpoint mis-loaded.
//
// `testdata/ref_halffloat.safetensors` is produced by the OFFICIAL torch
// `safetensors.torch.save_file` (testdata/gen.py, `make golden`), exactly as HF
// does. The values are chosen exactly representable in BOTH F16 (10-bit mantissa)
// and BF16 (7-bit mantissa), so a correct verbatim-U16 read must reproduce them to
// the bit. That exactness is the §B68 discriminator: 1.5 is F16 0x3E00 / BF16
// 0x3FC0 — a byte-swap makes it a tiny denormal, and reading the wrong 16 bits or
// the wrong dtype makes it nonsense. Exact equality therefore genuinely tests the
// convention, not just "some float came back".
func TestLoadHalfFloatReference(t *testing.T) {
	ts, meta, err := LoadFile("testdata/ref_halffloat.safetensors")
	if err != nil {
		t.Fatalf("load half-float reference (run `make golden`): %v", err)
	}
	if meta["format"] != "goai-golden" {
		t.Errorf("metadata lost: %v", meta)
	}

	// vals reshaped [2,3], exactly representable in F16 and BF16.
	want := [2][3]float64{{1.5, -2.5, 3.5}, {-4.25, 0.75, -0.125}}

	for _, c := range []struct {
		name string
		dt   tensor.Dtype
	}{
		{"h16", tensor.F16},
		{"hbf16", tensor.BF16},
	} {
		x := ts[c.name]
		if x == nil {
			t.Fatalf("%s: missing from file", c.name)
		}
		if x.Dtype() != c.dt {
			t.Errorf("%s: dtype %v, want %v", c.name, x.Dtype(), c.dt)
		}
		if !x.Shape().Equal(tensor.Shape{2, 3}) {
			t.Fatalf("%s: shape %v, want (2,3)", c.name, x.Shape())
		}
		for i := range 2 {
			for j := range 3 {
				if got := x.AtF64(i, j); got != want[i][j] {
					t.Errorf("%s(%d,%d) = %v, want %v (verbatim-U16 read disagrees with the reference lib)",
						c.name, i, j, got, want[i][j])
				}
			}
		}
	}
}
