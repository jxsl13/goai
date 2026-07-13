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

// spd builds a random n×n symmetric positive-definite matrix M·Mᵀ + n·I.
func spd(rng *rand.Rand, n int) [][]float64 {
	m := make([][]float64, n)
	for i := range n {
		m[i] = make([]float64, n)
		for j := range n {
			m[i][j] = rng.NormFloat64()
		}
	}
	a := make([][]float64, n)
	for i := range n {
		a[i] = make([]float64, n)
		for j := range n {
			var s float64
			for k := range n {
				s += m[i][k] * m[j][k]
			}
			if i == j {
				s += float64(n)
			}
			a[i][j] = s
		}
	}
	return a
}

func matTensor(a [][]float64) *tensor.Tensor {
	n := len(a)
	t := tensor.New(tensor.F64, tensor.Shape{n, n})
	for i := range n {
		for j := range n {
			t.SetF64(a[i][j], i, j)
		}
	}
	return t
}

// cholLoss = Σ_ij W_ij·chol(A)_ij, the scalar whose A-gradient is the reverse-mode
// Cholesky (used both as the tape loss and the finite-difference target).
func cholLoss(a [][]float64, w [][]float64) (float64, error) {
	out, err := backend.Execute(backend.NewContext(), backend.OpCholesky, []*tensor.Tensor{matTensor(a)}, nil)
	if err != nil {
		return 0, err
	}
	l := out[0]
	n := len(a)
	var s float64
	for i := range n {
		for j := range n {
			s += w[i][j] * l.AtF64(i, j)
		}
	}
	return s, nil
}

// §V2: the Murray reverse-mode Cholesky gradient matches central finite differences.
// A is symmetric, so the perturbation dA is symmetric and ⟨Ā, dA⟩ = ⟨L̄, dL⟩ holds
// with the full Frobenius inner product on both sides (Murray 2016 eq. 9). The tape
// gradient Grad(A) must equal the numeric directional derivative for every symmetric
// direction — checked here per-entry-pair by finite-differencing symmetric bumps.
func TestCholeskyGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	const h = 1e-6
	for _, n := range []int{1, 2, 3, 4, 5} {
		a := spd(rng, n)
		// random lower-triangular output cotangent W (= L̄)
		w := make([][]float64, n)
		for i := range n {
			w[i] = make([]float64, n)
			for j := 0; j <= i; j++ {
				w[i][j] = rng.NormFloat64()
			}
		}
		// analytic gradient via the tape
		at := matTensor(a)
		wt := matTensor(w)
		tape := autograd.NewTape()
		l, err := backend.Execute(tape.Context(), backend.OpCholesky, []*tensor.Tensor{at}, nil)
		if err != nil {
			t.Fatalf("n=%d forward: %v", n, err)
		}
		p, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{l[0], wt}, nil)
		loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{p[0]}, nil)
		if err := tape.Backward(loss[0]); err != nil {
			t.Fatalf("n=%d backward: %v", n, err)
		}
		abar := tape.Grad(at)
		if abar == nil {
			t.Fatalf("n=%d: nil gradient", n)
		}
		// Ā must be symmetric.
		for i := range n {
			for j := range n {
				if math.Abs(abar.AtF64(i, j)-abar.AtF64(j, i)) > 1e-9 {
					t.Errorf("n=%d Ā not symmetric at [%d,%d]", n, i, j)
				}
			}
		}
		// Per symmetric direction dA = E_ij + E_ji (i≠j) or E_ii, the numeric
		// directional derivative equals ⟨Ā, dA⟩ = Ā_ij+Ā_ji (i≠j) or Ā_ii.
		for i := range n {
			for j := 0; j <= i; j++ {
				ap := deepCopy(a)
				am := deepCopy(a)
				ap[i][j] += h
				am[i][j] -= h
				if i != j { // keep the perturbation symmetric
					ap[j][i] += h
					am[j][i] -= h
				}
				lp, err := cholLoss(ap, w)
				if err != nil {
					t.Fatalf("n=%d loss+: %v", n, err)
				}
				lm, err := cholLoss(am, w)
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

func deepCopy(a [][]float64) [][]float64 {
	c := make([][]float64, len(a))
	for i := range a {
		c[i] = append([]float64(nil), a[i]...)
	}
	return c
}

// closed form for 1×1: L=√a, so dL/da = 1/(2√a) and Ā = ḡ/(2√a).
func TestCholeskyGrad1x1(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{1, 1}, []float64{9})
	tape := autograd.NewTape()
	l, _ := backend.Execute(tape.Context(), backend.OpCholesky, []*tensor.Tensor{a}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{l[0]}, nil) // W=1
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	// L=3, Ā = 1/(2·3) = 1/6
	if g := tape.Grad(a).AtF64(0, 0); math.Abs(g-1.0/6.0) > 1e-12 {
		t.Errorf("Ā = %.12g, want %.12g", g, 1.0/6.0)
	}
}
