package autograd_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// logDetOf runs the forward op and returns the scalar log-determinant.
func logDetOf(a [][]float64) (float64, error) {
	out, err := backend.Execute(backend.NewContext(), backend.OpLogDet, []*tensor.Tensor{matTensor(a)}, nil)
	if err != nil {
		return 0, err
	}
	return out[0].AtF64(), nil
}

// §V2: the Jacobi-formula gradient Ā = A⁻¹ matches central finite differences. A is
// symmetric, so perturbations are symmetric and ⟨Ā, dA⟩ = d logdet holds with the
// full Frobenius inner product; the tape gradient must equal the numeric directional
// derivative per symmetric entry-pair. (spd/matTensor/deepCopy live in
// cholesky_test.go in this package.)
func TestLogDetGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 17))
	const h = 1e-6
	for _, n := range []int{1, 2, 3, 4, 5} {
		a := spd(rng, n)
		at := matTensor(a)
		tape := autograd.NewTape()
		ld, err := backend.Execute(tape.Context(), backend.OpLogDet, []*tensor.Tensor{at}, nil)
		if err != nil {
			t.Fatalf("n=%d forward: %v", n, err)
		}
		if err := tape.Backward(ld[0]); err != nil {
			t.Fatalf("n=%d backward: %v", n, err)
		}
		abar := tape.Grad(at)
		if abar == nil {
			t.Fatalf("n=%d: nil gradient", n)
		}
		for i := range n {
			for j := 0; j <= i; j++ {
				ap := deepCopy(a)
				am := deepCopy(a)
				ap[i][j] += h
				am[i][j] -= h
				if i != j {
					ap[j][i] += h
					am[j][i] -= h
				}
				lp, err := logDetOf(ap)
				if err != nil {
					t.Fatalf("n=%d loss+: %v", n, err)
				}
				lm, err := logDetOf(am)
				if err != nil {
					t.Fatalf("n=%d loss-: %v", n, err)
				}
				num := (lp - lm) / (2 * h)
				var ana float64
				if i == j {
					ana = abar.AtF64(i, i)
				} else {
					ana = abar.AtF64(i, j) + abar.AtF64(j, i)
				}
				if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
					t.Errorf("n=%d grad[%d,%d]: numeric %.8g vs analytic %.8g", n, i, j, num, ana)
				}
			}
		}
	}
}

// closed form: the gradient equals A⁻¹ exactly (Jacobi's formula for symmetric A).
func TestLogDetGradIsInverse(t *testing.T) {
	// A = [[4,12,-16],[12,37,-43],[-16,-43,98]]; A⁻¹[0,0] = 49.36111… (numpy).
	a := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{4, 12, -16, 12, 37, -43, -16, -43, 98})
	tape := autograd.NewTape()
	ld, _ := backend.Execute(tape.Context(), backend.OpLogDet, []*tensor.Tensor{a}, nil)
	if err := tape.Backward(ld[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(a)
	if got := g.AtF64(0, 0); math.Abs(got-49.36111111111101) > 1e-8 {
		t.Errorf("Ā[0,0] = %.12g, want A⁻¹[0,0] = 49.36111111111101", got)
	}
	// Ā must be symmetric (A⁻¹ is).
	for i := range 3 {
		for j := range 3 {
			if math.Abs(g.AtF64(i, j)-g.AtF64(j, i)) > 1e-12 {
				t.Errorf("Ā not symmetric at [%d,%d]", i, j)
			}
		}
	}
}
