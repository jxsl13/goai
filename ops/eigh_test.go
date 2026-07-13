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

// §V16 tier-1: eigenvalues match numpy.linalg.eigh (ascending) exactly, eigenvectors
// up to per-column sign, and V·diag(w)·Vᵀ reconstructs A with VᵀV=I. Goldens from
// numpy 2.5.1 in .venv.
func TestEighVsNumpy(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{2, 1, 0, 1, 3, 1, 0, 1, 4})
	w, v, err := ops.Eigh(a)
	if err != nil {
		t.Fatalf("Eigh: %v", err)
	}
	wantW := []float64{1.2679491924311228, 3.0, 4.7320508075688785}
	wantV := []float64{
		0.788675134594813, -0.5773502691896258, 0.21132486540518713,
		-0.5773502691896257, -0.577350269189626, 0.5773502691896257,
		0.2113248654051871, 0.5773502691896258, 0.788675134594813}
	for k := range 3 {
		if math.Abs(w.AtF64(k)-wantW[k]) > 1e-10 {
			t.Errorf("w[%d] = %.15g, want %.15g", k, w.AtF64(k), wantW[k])
		}
		// sign-align eigenvector column k by its first entry, then compare.
		s := 1.0
		if v.AtF64(0, k)*wantV[k] < 0 {
			s = -1
		}
		for i := range 3 {
			if got := s * v.AtF64(i, k); math.Abs(got-wantV[i*3+k]) > 1e-10 {
				t.Errorf("V[%d,%d] = %.15g, want %.15g", i, k, got, wantV[i*3+k])
			}
		}
	}
}

// defining properties: A·V[:,k] = w[k]·V[:,k], VᵀV = I, V·diag(w)·Vᵀ = A.
func TestEighProperties(t *testing.T) {
	n := 4
	// symmetric A = M + Mᵀ + diag boost
	raw := []float64{
		3, 0.5, -0.2, 0.7, 0.5, 2, 0.9, -0.4,
		-0.2, 0.9, 4, 0.1, 0.7, -0.4, 0.1, 5,
	}
	a := tensor.FromFloat64(tensor.Shape{n, n}, raw)
	w, v, err := ops.Eigh(a)
	if err != nil {
		t.Fatalf("Eigh: %v", err)
	}
	// ascending
	for k := 1; k < n; k++ {
		if w.AtF64(k) < w.AtF64(k-1)-1e-12 {
			t.Errorf("eigenvalues not ascending: w[%d]=%g < w[%d]=%g", k, w.AtF64(k), k-1, w.AtF64(k-1))
		}
	}
	// VᵀV = I
	for i := range n {
		for j := range n {
			var s float64
			for r := range n {
				s += v.AtF64(r, i) * v.AtF64(r, j)
			}
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(s-want) > 1e-9 {
				t.Errorf("(VᵀV)[%d,%d] = %g, want %g", i, j, s, want)
			}
		}
	}
	// V·diag(w)·Vᵀ = A
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += v.AtF64(i, k) * w.AtF64(k) * v.AtF64(j, k)
			}
			if math.Abs(s-a.AtF64(i, j)) > 1e-9 {
				t.Errorf("recon[%d,%d] = %.10g, want %.10g", i, j, s, a.AtF64(i, j))
			}
		}
	}
}

func TestEighErrors(t *testing.T) {
	if _, _, err := ops.Eigh(tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 0, 0, 0, 1, 0})); err == nil {
		t.Error("non-square: want error")
	}
}

func ExampleEigh() {
	// diag(3,1,2) has eigenvalues {1,2,3} returned ascending.
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{3, 0, 0, 0, 1, 0, 0, 0, 2})
	w, _, _ := ops.Eigh(a)
	fmt.Printf("%.0f %.0f %.0f\n", w.AtF64(0), w.AtF64(1), w.AtF64(2))
	// Output:
	// 1 2 3
}
