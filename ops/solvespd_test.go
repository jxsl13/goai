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

// §V16 tier-1: X = A⁻¹·B matches numpy.linalg.solve (SPD A) for vector and matrix
// right-hand sides; goldens from numpy 2.5.1 in .venv.
func TestSolveSPDVsNumpy(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{4, 12, -16, 12, 37, -43, -16, -43, 98})

	// vector rhs b = [1,2,3]
	bv := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	xv, err := ops.SolveSPD(a, bv)
	if err != nil {
		t.Fatalf("SolveSPD vec: %v", err)
	}
	wantV := []float64{28.583333333333268, -7.666666666666649, 1.3333333333333306}
	for i := range 3 {
		if math.Abs(xv.AtF64(i)-wantV[i]) > 1e-9 {
			t.Errorf("x[%d] = %.15g, want %.15g", i, xv.AtF64(i), wantV[i])
		}
	}

	// matrix rhs B = [[1,4],[2,5],[3,6]]
	bm := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 4, 2, 5, 3, 6})
	xm, err := ops.SolveSPD(a, bm)
	if err != nil {
		t.Fatalf("SolveSPD mat: %v", err)
	}
	wantM := []float64{28.583333333333268, 142.333333333333, -7.666666666666649, -38.66666666666658, 1.3333333333333306, 6.33333333333332}
	for i := range 3 {
		for c := range 2 {
			if got, w := xm.AtF64(i, c), wantM[i*2+c]; math.Abs(got-w) > 1e-9 {
				t.Errorf("X[%d,%d] = %.15g, want %.15g", i, c, got, w)
			}
		}
	}
}

// residual: A·X reconstructs B (the defining property).
func TestSolveSPDResidual(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{4, 12, -16, 12, 37, -43, -16, -43, 98})
	b := tensor.FromFloat64(tensor.Shape{3}, []float64{5, -2, 7})
	x, err := ops.SolveSPD(a, b)
	if err != nil {
		t.Fatalf("SolveSPD: %v", err)
	}
	for i := range 3 {
		var ax float64 // (A·x)_i
		for j := range 3 {
			ax += a.AtF64(i, j) * x.AtF64(j)
		}
		if math.Abs(ax-b.AtF64(i)) > 1e-9 {
			t.Errorf("(A·x)[%d] = %.12g, want %.12g", i, ax, b.AtF64(i))
		}
	}
}

func TestSolveSPDErrors(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{4, 12, -16, 12, 37, -43, -16, -43, 98})
	// rhs wrong leading dim
	if _, err := ops.SolveSPD(a, tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})); err == nil {
		t.Error("bad rhs length: want error")
	}
	// non-SPD A
	bad := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 2, 1})
	if _, err := ops.SolveSPD(bad, tensor.FromFloat64(tensor.Shape{2}, []float64{1, 1})); err == nil {
		t.Error("indefinite A: want error")
	}
}

func ExampleSolveSPD() {
	// 2·I₂ · x = [2,6]  ⇒  x = [1,3].
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{2, 0, 0, 2})
	b := tensor.FromFloat64(tensor.Shape{2}, []float64{2, 6})
	x, _ := ops.SolveSPD(a, b)
	fmt.Printf("%.4g %.4g\n", x.AtF64(0), x.AtF64(1))
	// Output:
	// 1 3
}
