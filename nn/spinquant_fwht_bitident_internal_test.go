package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestFoldWeightFWHTBitIdentical proves the cache-tiled foldWeightFWHT F64 path is BIT-identical to the
// original per-column (strided) walk — the tiling only reorders memory access, so every f64 output word
// must match. Covers a non-B-multiple `out` to exercise the tile tail.
func TestFoldWeightFWHTBitIdentical(t *testing.T) {
	for _, d := range []struct{ dim, out int }{{8, 5}, {64, 48}, {256, 33}, {4096, 70}} {
		s, err := NewHadamardRotation(tensor.F64, d.dim, WithSpinSeed(3))
		if err != nil {
			t.Fatal(err)
		}
		w := tensor.New(tensor.F64, tensor.Shape{d.dim, d.out})
		ws := w.Storage().F64()
		for i := range ws {
			ws[i] = float64((i*7+3)%97) / 13.0
		}
		got := s.foldWeightFWHT(w).Storage().F64()

		// Reference: the ORIGINAL column-walk algorithm.
		ref := make([]float64, d.dim*d.out)
		buf := make([]float64, d.dim)
		for l := 0; l < d.out; l++ {
			for k := 0; k < d.dim; k++ {
				buf[k] = ws[k*d.out+l]
			}
			fwhtInPlace(buf)
			for j := 0; j < d.dim; j++ {
				ref[j*d.out+l] = s.inv * s.signs[j] * buf[j]
			}
		}
		for i := range ref {
			if math.Float64bits(got[i]) != math.Float64bits(ref[i]) {
				t.Fatalf("[%dx%d] out[%d]: got %v want %v (tiling not bit-identical)", d.dim, d.out, i, got[i], ref[i])
			}
		}
	}
}
