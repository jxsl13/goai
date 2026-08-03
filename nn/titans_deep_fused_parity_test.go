package nn_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// The fused DEEP-memory inference path (ctx.Recorder==nil) must match the dispatch path
// (recorder set). It is BIT-IDENTICAL on the default build (the fused sigmoid replicates the
// backend's stable scalar form); the amd64 SIMD build's vectorized sigmoid differs by ~1ulp, so
// this asserts a tight tolerance that both builds satisfy while also confirming near-bit-exactness.
func TestTitansDeepFusedMatchesDispatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 41))
	randMat := func(rows, cols int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		s := x.Storage().F64()
		for i := range s {
			s[i] = rng.NormFloat64()
		}
		return x
	}
	sigCol := func(seq int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{seq, 1})
		s := x.Storage().F64()
		for i := range s {
			s[i] = 1 / (1 + math.Exp(-rng.NormFloat64()))
		}
		return x
	}

	var maxAbs, maxBits float64
	// EVERY dim AND hid HERE WAS A MULTIPLE OF FOUR, so when the inner loops were jammed four
	// units per pass, neither jam's by-one tail ran at all. dim=10 leaves a tail of two and
	// hid=13 a tail of one; dim=14 and hid=7 leave three.
	for _, tc := range []struct{ seq, dim, hid int }{{1, 8, 12}, {5, 16, 8}, {33, 24, 40},
		{96, 40, 64}, {7, 10, 13}, {9, 14, 7}} {
		m, err := nn.NewNeuralMemory(tensor.F64, tc.dim, tc.hid, 3) // deep (default)
		if err != nil {
			t.Fatal(err)
		}
		q, k, v := randMat(tc.seq, tc.dim), randMat(tc.seq, tc.dim), randMat(tc.seq, tc.dim)
		eta, theta, alpha := sigCol(tc.seq), sigCol(tc.seq), sigCol(tc.seq)

		disp, err := m.Scan(autograd.NewTape().Context(), q, k, v, eta, theta, alpha)
		if err != nil {
			t.Fatal(err)
		}
		fused, err := m.Scan(backend.NewContext(), q, k, v, eta, theta, alpha)
		if err != nil {
			t.Fatal(err)
		}
		for i := range disp.Numel() {
			c := tensor.Unravel(i, disp.Shape())
			a, b := disp.AtF64(c...), fused.AtF64(c...)
			d := math.Abs(a - b)
			if d > maxAbs {
				maxAbs = d
			}
			if math.Float64bits(a) != math.Float64bits(b) {
				maxBits++
			}
			// A TOLERANCE, DELIBERATELY. Raising this to a bit-exact assertion was tried and
			// is wrong: the two paths already disagree by one ulp on this host at seq=1 dim=8
			// hid=12, before any jam, so the relationship between them is agreement and not
			// identity. Bit-identity of the fused path against ITSELF is what
			// TestTitansScanIsBitIdentical freezes.
			if rel := d / (1 + math.Abs(a)); rel > 1e-9 {
				t.Fatalf("seq=%d dim=%d hid=%d idx=%d: dispatch %v vs fused %v (rel %g > 1e-9)", tc.seq, tc.dim, tc.hid, i, a, b, rel)
			}
		}
	}
	t.Logf("deep fused vs dispatch: maxAbs=%g, elems differing in bits=%.0f (0 == bit-exact, default build)", maxAbs, maxBits)
}
