package linalg_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

func randRect(rng *rand.Rand, m, n int) *tensor.Tensor {
	d := make([]float64, m*n)
	for i := range d {
		d[i] = rng.NormFloat64()
	}
	return tensor.FromFloat64(tensor.Shape{m, n}, d)
}

// §V16 tier-1: Householder QR reconstructs A = Q·R, Q has orthonormal columns (QᵀQ = I_n), and R is
// upper-triangular (§R116).
func TestQRReconstruct(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 22))
	for _, d := range [][2]int{{1, 1}, {3, 3}, {5, 2}, {6, 4}, {8, 8}, {10, 3}} {
		m, n := d[0], d[1]
		a := randRect(rng, m, n)
		q, r, err := linalg.QR(a)
		if err != nil {
			t.Fatalf("%dx%d: %v", m, n, err)
		}
		if !q.Shape().Equal(tensor.Shape{m, n}) || !r.Shape().Equal(tensor.Shape{n, n}) {
			t.Fatalf("%dx%d shapes Q%v R%v", m, n, q.Shape(), r.Shape())
		}
		// R upper-triangular
		for i := range n {
			for j := range i {
				if math.Abs(r.AtF64(i, j)) > 1e-12 {
					t.Errorf("%dx%d R[%d,%d] = %g (not upper)", m, n, i, j, r.AtF64(i, j))
				}
			}
		}
		// Q·R == A
		qr := matmul(q, r)
		for i := range m {
			for j := range n {
				if math.Abs(qr.AtF64(i, j)-a.AtF64(i, j)) > 1e-9 {
					t.Errorf("%dx%d (Q·R−A)[%d,%d] = %g", m, n, i, j, qr.AtF64(i, j)-a.AtF64(i, j))
				}
			}
		}
		// QᵀQ == I_n (orthonormal columns)
		for i := range n {
			for j := range n {
				var s float64
				for p := range m {
					s += q.AtF64(p, i) * q.AtF64(p, j)
				}
				want := 0.0
				if i == j {
					want = 1
				}
				if math.Abs(s-want) > 1e-9 {
					t.Errorf("%dx%d (QᵀQ)[%d,%d] = %g, want %g", m, n, i, j, s, want)
				}
			}
		}
	}
}

// §V16 tier-1: for a square system, Lstsq gives the exact solution A·x = b.
func TestLstsqSquare(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 9))
	const n = 5
	a := randMatrix(rng, n) // well-conditioned square
	bd := make([]float64, n)
	for i := range bd {
		bd[i] = rng.NormFloat64()
	}
	b := tensor.FromFloat64(tensor.Shape{n}, bd)
	x, err := linalg.Lstsq(a, b)
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		var s float64
		for j := range n {
			s += a.AtF64(i, j) * x.AtF64(j)
		}
		if math.Abs(s-b.AtF64(i)) > 1e-9 {
			t.Errorf("square residual[%d] = %g", i, s-b.AtF64(i))
		}
	}
}

// §V16 tier-1: for an overdetermined system (m>n), the least-squares solution satisfies the normal
// equations AᵀA·x = Aᵀ·b (the residual is orthogonal to the column space).
func TestLstsqOverdetermined(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 13))
	const m, n = 12, 4
	a := randRect(rng, m, n)
	bd := make([]float64, m)
	for i := range bd {
		bd[i] = rng.NormFloat64()
	}
	b := tensor.FromFloat64(tensor.Shape{m}, bd)
	x, err := linalg.Lstsq(a, b)
	if err != nil {
		t.Fatal(err)
	}
	// AᵀA·x  vs  Aᵀ·b
	for i := range n {
		var lhs, rhs float64
		for j := range n {
			var ata float64
			for p := range m {
				ata += a.AtF64(p, i) * a.AtF64(p, j)
			}
			lhs += ata * x.AtF64(j)
		}
		for p := range m {
			rhs += a.AtF64(p, i) * b.AtF64(p)
		}
		if math.Abs(lhs-rhs) > 1e-8 {
			t.Errorf("normal eq %d: AᵀAx=%g vs Aᵀb=%g", i, lhs, rhs)
		}
	}
}

// §V16 tier-1: exact line fit — data on a straight line is recovered by least squares with ~0 residual.
func TestLstsqLineFit(t *testing.T) {
	// y = 2x + 1 at x = 0..4; design [x,1]
	xs := []float64{0, 1, 2, 3, 4}
	ad := make([]float64, len(xs)*2)
	bd := make([]float64, len(xs))
	for i, x := range xs {
		ad[i*2], ad[i*2+1] = x, 1
		bd[i] = 2*x + 1
	}
	a := tensor.FromFloat64(tensor.Shape{len(xs), 2}, ad)
	b := tensor.FromFloat64(tensor.Shape{len(xs)}, bd)
	coef, _ := linalg.Lstsq(a, b)
	if math.Abs(coef.AtF64(0)-2) > 1e-9 || math.Abs(coef.AtF64(1)-1) > 1e-9 {
		t.Errorf("line fit slope=%g intercept=%g, want 2/1", coef.AtF64(0), coef.AtF64(1))
	}
}

func TestQRErrors(t *testing.T) {
	if _, _, err := linalg.QR(tensor.New(tensor.F64, tensor.Shape{2, 3})); err == nil {
		t.Error("wide matrix (m<n) QR must error")
	}
	if _, err := linalg.Lstsq(tensor.New(tensor.F64, tensor.Shape{2, 3}), tensor.New(tensor.F64, tensor.Shape{2})); err == nil {
		t.Error("wide Lstsq must error")
	}
}

func ExampleQR() {
	// A = [[0,1],[1,1],[1,0]] (3×2); check Q·R reconstructs A[0].
	a := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{0, 1, 1, 1, 1, 0})
	q, r, _ := linalg.QR(a)
	rec := q.AtF64(1, 0)*r.AtF64(0, 0) + q.AtF64(1, 1)*r.AtF64(1, 0)
	fmt.Printf("%.0f\n", rec) // A[1,0] = 1
	// Output: 1
}

func ExampleLstsq() {
	// least-squares line fit of y=2x+1 through (0,1),(1,3),(2,5)
	a := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{0, 1, 1, 1, 2, 1})
	b := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 3, 5})
	c, _ := linalg.Lstsq(a, b)
	fmt.Printf("slope=%.0f intercept=%.0f\n", c.AtF64(0), c.AtF64(1))
	// Output: slope=2 intercept=1
}
