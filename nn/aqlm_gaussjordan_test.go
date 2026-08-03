package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestGaussJordanAQLMResidual is the oracle that knows nothing about how the elimination is
// organized: build A and B, solve, and check A·X against B. A row the elimination skips — which
// is exactly what a bad band split produces — leaves a residual no rearrangement can hide,
// while a test that compared against a second copy of the same loop would agree with the bug.
//
// Sized to straddle the work gate: the 24-wide system stays serial, the 192-wide one bands.
func TestGaussJordanAQLMResidual(t *testing.T) {
	for _, n := range []int{1, 2, 24, 192} {
		const rhs = 3
		stride := n + rhs
		rng := rand.New(rand.NewPCG(11, 22))
		a := make([]float64, n*n)
		b := make([]float64, n*rhs)
		aug := make([]float64, n*stride)
		for i := range n {
			for j := range n {
				v := rng.NormFloat64()
				if i == j {
					v += float64(n) // diagonally dominant: well conditioned, no pivot degeneracy
				}
				a[i*n+j] = v
				aug[i*stride+j] = v
			}
			for c := range rhs {
				v := rng.NormFloat64()
				b[i*rhs+c] = v
				aug[i*stride+n+c] = v
			}
		}
		gaussJordanAQLM(aug, make([]float64, stride), n, stride)
		for i := range n {
			for c := range rhs {
				var s float64
				for j := range n {
					s += a[i*n+j] * aug[j*stride+n+c]
				}
				if math.Abs(s-b[i*rhs+c]) > 1e-8*(1+math.Abs(b[i*rhs+c])) {
					t.Fatalf("n=%d: residual at (%d,%d): A·X = %v, want %v", n, i, c, s, b[i*rhs+c])
				}
			}
		}
	}
}

// TestEncodeAQLMIsBitIdentical freezes the encoder's codes and codebooks. Banding the
// elimination claims to change no value — every row's arithmetic is untouched and only which
// goroutine runs it moves — and AQLM is an approximation whose accuracy would absorb a real
// change without complaint, so the digest is the only gate that would notice.
func TestEncodeAQLMIsBitIdentical(t *testing.T) {
	cases := []struct {
		rows, cols, m, bits, g int
		want                   uint64
	}{
		{16, 32, 2, 4, 8, 13893831496142817533},
		{24, 48, 3, 5, 8, 11964055707955120178}, // M=3: the ICM sweep subtracts two codebooks per step
		{32, 64, 2, 6, 16, 14319829737708910524},
	}
	for _, c := range cases {
		w := tensor.New(tensor.F64, tensor.Shape{c.rows, c.cols})
		ws := w.Storage().F64()
		for i := range ws {
			ws[i] = math.Sin(float64(i*7+3)) * 0.75
		}
		q, err := EncodeAQLM(w, WithAQLMCodebooks(c.m), WithAQLMBits(c.bits),
			WithAQLMGroupSize(c.g))
		if err != nil {
			t.Fatal(err)
		}
		h := uint64(14695981039346656037)
		mix := func(u uint64) {
			for s := 0; s < 64; s += 8 {
				h = (h ^ (u>>s)&0xff) * 1099511628211
			}
		}
		for _, code := range q.Codes {
			mix(uint64(code))
		}
		for _, cb := range q.Codebooks {
			for _, v := range cb {
				mix(math.Float64bits(v))
			}
		}
		if h != c.want {
			t.Fatalf("%dx%d M=%d B=%d g=%d digest %d, want %d",
				c.rows, c.cols, c.m, c.bits, c.g, h, c.want)
		}
	}
}
