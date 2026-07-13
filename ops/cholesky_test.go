package ops_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: the factor L matches numpy.linalg.cholesky (LOWER, A=L·Lᵀ) exactly
// on fixed SPD matrices whose golden L was produced by numpy 2.5.1 in .venv.
func TestCholeskyVsNumpy(t *testing.T) {
	cases := []struct {
		name string
		n    int
		a    []float64
		l    []float64 // numpy.linalg.cholesky(A), row-major, lower (upper = 0)
	}{
		{"3x3", 3,
			[]float64{4, 12, -16, 12, 37, -43, -16, -43, 98},
			[]float64{2, 0, 0, 6, 1, 0, -8, 5, 3}},
		{"4x4", 4,
			[]float64{
				4.957555796462167, -1.506881695982532, -0.6380945028884456, -0.8890558190293714,
				-1.506881695982532, 6.98988257027522, 1.346849996002904, 1.8048632832193072,
				-0.6380945028884456, 1.346849996002904, 4.994569928934063, 0.7592623904683085,
				-0.8890558190293714, 1.8048632832193072, 0.7592623904683085, 5.361185147456223},
			[]float64{
				2.2265569376196437, 0, 0, 0,
				-0.6767766278609078, 2.5557495898965605, 0, 0,
				-0.2865835102203209, 0.45109934827461506, 2.170011336051054, 0,
				-0.39929624255638335, 0.6004613772535432, 0.17233224983796402, 2.1935121126242136}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := tensor.FromFloat64(tensor.Shape{c.n, c.n}, c.a)
			l, err := ops.Cholesky(a)
			if err != nil {
				t.Fatalf("Cholesky: %v", err)
			}
			for i := range c.n {
				for j := range c.n {
					want := c.l[i*c.n+j]
					if got := l.AtF64(i, j); math.Abs(got-want) > 1e-12 {
						t.Errorf("L[%d,%d] = %.17g, want %.17g", i, j, got, want)
					}
				}
			}
		})
	}
}

// round-trip: L·Lᵀ reconstructs A (the defining property).
func TestCholeskyRoundTrip(t *testing.T) {
	n := 5
	// A = MᵀM + n·I is SPD.
	raw := []float64{
		0.5, -1.2, 0.3, 0.9, -0.4,
		1.1, 0.2, -0.7, 0.6, 1.3,
		-0.8, 0.4, 1.5, -0.2, 0.1,
		0.7, -0.9, 0.2, 1.0, -0.6,
		0.3, 1.4, -0.5, 0.8, 0.2,
	}
	a := make([]float64, n*n)
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += raw[k*n+i] * raw[k*n+j]
			}
			if i == j {
				s += float64(n)
			}
			a[i*n+j] = s
		}
	}
	at := tensor.FromFloat64(tensor.Shape{n, n}, a)
	l, err := ops.Cholesky(at)
	if err != nil {
		t.Fatalf("Cholesky: %v", err)
	}
	for i := range n {
		for j := range n {
			var recon float64 // (L·Lᵀ)_ij = Σ_k L[i,k]·L[j,k]
			for k := range n {
				recon += l.AtF64(i, k) * l.AtF64(j, k)
			}
			if math.Abs(recon-a[i*n+j]) > 1e-10 {
				t.Errorf("(LLᵀ)[%d,%d] = %.12g, want %.12g", i, j, recon, a[i*n+j])
			}
			if j > i && l.AtF64(i, j) != 0 { // strict upper is exact zero
				t.Errorf("L[%d,%d] = %g, want 0 (upper triangle)", i, j, l.AtF64(i, j))
			}
		}
	}
}

func TestCholeskyErrors(t *testing.T) {
	// non-square
	if _, err := ops.Cholesky(tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 0, 0, 0, 1, 0})); err == nil {
		t.Error("non-square: want error")
	}
	// symmetric but indefinite (eigenvalues ±): pivot goes non-positive
	if _, err := ops.Cholesky(tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 2, 1})); err == nil {
		t.Error("indefinite: want error")
	}
	// negative diagonal at pivot 0
	if _, err := ops.Cholesky(tensor.FromFloat64(tensor.Shape{1, 1}, []float64{-1})); err == nil {
		t.Error("negative pivot: want error")
	}
}

// F32 input factors in f64 and narrows (§V10): reconstruction holds at f32 tol.
func TestCholeskyF32(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{2, 1, 1, 2})
	af := tensor.New(tensor.F32, tensor.Shape{2, 2})
	for i := range 2 {
		for j := range 2 {
			af.SetF64(a.AtF64(i, j), i, j)
		}
	}
	l, err := ops.Cholesky(af)
	if err != nil {
		t.Fatalf("Cholesky f32: %v", err)
	}
	if l.Dtype() != tensor.F32 {
		t.Fatalf("dtype = %v, want f32", l.Dtype())
	}
	for i := range 2 {
		for j := range 2 {
			var recon float64
			for k := range 2 {
				recon += l.AtF64(i, k) * l.AtF64(j, k)
			}
			if math.Abs(recon-a.AtF64(i, j)) > 1e-6 {
				t.Errorf("(LLᵀ)[%d,%d] = %g, want %g", i, j, recon, a.AtF64(i, j))
			}
		}
	}
}

func ExampleCholesky() {
	// A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] is symmetric positive-definite.
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{4, 12, -16, 12, 37, -43, -16, -43, 98})
	l, _ := ops.Cholesky(a)
	for i := range 3 {
		fmt.Printf("%g %g %g\n", l.AtF64(i, 0), l.AtF64(i, 1), l.AtF64(i, 2))
	}
	// Output:
	// 2 0 0
	// 6 1 0
	// -8 5 3
}
