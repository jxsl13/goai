package linalg

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// qrGoldenHash is an FNV-1a hash over every element of Q and R for a fixed input,
// captured before applyReflector's column loop was parallelized.
//
// IT EXISTS BECAUSE NOTHING ELSE COVERED THAT FUNCTION. A one-ulp perturbation of
// applyReflector's update turned NO test in linalg or nn red — including
// nn.TestOrthogonalBitIdenticalToSlowPath, which looks like exactly the right guard and
// structurally cannot be: it compares the fast path against the slow path, and BOTH call
// applyReflector, so perturbing shared code moves both arms identically and the
// comparison stays equal. A differential gate covers only what DIFFERS between its arms.
//
// The reconstruction test (TestQRReconstruct) checks Q·R ≈ A within a tolerance, which a
// one-ulp shift passes comfortably.
const qrGoldenHash = 0x2e7f379031a7dc78

// TestQRBitIdenticalToGolden holds Householder QR bit-for-bit at tolerance 0. The
// reflector application is parallel over columns, which must not move a single bit — each
// column's dot and update are independent and their accumulation order is untouched.
func TestQRBitIdenticalToGolden(t *testing.T) {
	const m, n = 24, 16
	a := tensor.New(tensor.F64, tensor.Shape{m, n})
	x := uint64(0x2545F4914F6CDD1D)
	for i := range m {
		for j := range n {
			x ^= x << 13
			x ^= x >> 7
			x ^= x << 17
			a.SetF64(float64(int64(x))/(1<<62), i, j)
		}
	}
	q, r, err := QR(a)
	if err != nil {
		t.Fatal(err)
	}
	var h uint64 = 1469598103934665603
	for i := range m {
		for j := range n {
			h = (h ^ math.Float64bits(q.AtF64(i, j))) * 1099511628211
		}
	}
	for i := range n {
		for j := range n {
			h = (h ^ math.Float64bits(r.AtF64(i, j))) * 1099511628211
		}
	}
	if h != qrGoldenHash {
		t.Fatalf("QR hash %#x, want %#x — Householder QR is no longer bit-identical to its "+
			"frozen reference", h, qrGoldenHash)
	}
}
