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

// solveLoss = Σ_ic W_ic·(A⁻¹B)_ic, the scalar whose A- and B-gradients are checked.
func solveLoss(a, b, w [][]float64, k int) (float64, error) {
	out, err := backend.Execute(backend.NewContext(), backend.OpSolveSPD,
		[]*tensor.Tensor{matTensor(a), colMat(b, k)}, nil)
	if err != nil {
		return 0, err
	}
	x := out[0]
	var s float64
	for i := range a {
		for c := range k {
			s += w[i][c] * x.AtF64(i, c)
		}
	}
	return s, nil
}

// colMat builds an [n,k] tensor from a row-major [][]float64.
func colMat(b [][]float64, k int) *tensor.Tensor {
	n := len(b)
	t := tensor.New(tensor.F64, tensor.Shape{n, k})
	for i := range n {
		for c := range k {
			t.SetF64(b[i][c], i, c)
		}
	}
	return t
}

// §V2: the reverse-mode solve gradients match central finite differences — B̄
// per-entry (B is unconstrained), and Ā per symmetric entry-pair (A is symmetric,
// so ⟨Ā,dA⟩ uses the full Frobenius product with the ½-symmetrized convention).
func TestSolveSPDGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 4))
	const h = 1e-6
	for _, n := range []int{1, 2, 3, 4} {
		for _, k := range []int{1, 3} {
			a := spd(rng, n)
			b := make([][]float64, n)
			w := make([][]float64, n)
			for i := range n {
				b[i] = make([]float64, k)
				w[i] = make([]float64, k)
				for c := range k {
					b[i][c] = rng.NormFloat64()
					w[i][c] = rng.NormFloat64()
				}
			}
			at := matTensor(a)
			bt := colMat(b, k)
			wt := colMat(w, k)
			tape := autograd.NewTape()
			x, err := backend.Execute(tape.Context(), backend.OpSolveSPD, []*tensor.Tensor{at, bt}, nil)
			if err != nil {
				t.Fatalf("n=%d k=%d forward: %v", n, k, err)
			}
			p, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{x[0], wt}, nil)
			loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{p[0]}, nil)
			if err := tape.Backward(loss[0]); err != nil {
				t.Fatalf("n=%d k=%d backward: %v", n, k, err)
			}
			abar := tape.Grad(at)
			bbar := tape.Grad(bt)
			if abar == nil || bbar == nil {
				t.Fatalf("n=%d k=%d: nil gradient", n, k)
			}

			// B gradient — per entry.
			for i := range n {
				for c := range k {
					bp := deepCopy2(b)
					bm := deepCopy2(b)
					bp[i][c] += h
					bm[i][c] -= h
					lp, _ := solveLoss(a, bp, w, k)
					lm, _ := solveLoss(a, bm, w, k)
					num := (lp - lm) / (2 * h)
					if ana := bbar.AtF64(i, c); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
						t.Errorf("n=%d k=%d B̄[%d,%d]: numeric %.8g vs analytic %.8g", n, k, i, c, num, ana)
					}
				}
			}
			// A gradient — symmetric entry-pair.
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
					lp, _ := solveLoss(ap, b, w, k)
					lm, _ := solveLoss(am, b, w, k)
					num := (lp - lm) / (2 * h)
					var ana float64
					if i == j {
						ana = abar.AtF64(i, i)
					} else {
						ana = abar.AtF64(i, j) + abar.AtF64(j, i)
					}
					if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
						t.Errorf("n=%d k=%d Ā[%d,%d]: numeric %.8g vs analytic %.8g", n, k, i, j, num, ana)
					}
				}
			}
		}
	}
}

func deepCopy2(b [][]float64) [][]float64 {
	c := make([][]float64, len(b))
	for i := range b {
		c[i] = append([]float64(nil), b[i]...)
	}
	return c
}

// closed form for a vector rhs: with a diagonal A, X_i = b_i/A_ii, and the loss
// Σ X_i has ∂/∂b_i = 1/A_ii and ∂/∂A_ii = −b_i/A_ii².
func TestSolveSPDGradClosedForm(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{2, 0, 0, 4})
	b := tensor.FromFloat64(tensor.Shape{2}, []float64{6, 8})
	tape := autograd.NewTape()
	x, _ := backend.Execute(tape.Context(), backend.OpSolveSPD, []*tensor.Tensor{a, b}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{x[0]}, nil) // W=1
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	gb := tape.Grad(b)
	if math.Abs(gb.AtF64(0)-0.5) > 1e-12 || math.Abs(gb.AtF64(1)-0.25) > 1e-12 { // 1/2, 1/4
		t.Errorf("B̄ = [%g,%g], want [0.5,0.25]", gb.AtF64(0), gb.AtF64(1))
	}
	ga := tape.Grad(a)
	// ∂/∂A_00 = −b_0/A_00² = −6/4 = −1.5 ; ∂/∂A_11 = −8/16 = −0.5
	if math.Abs(ga.AtF64(0, 0)+1.5) > 1e-12 || math.Abs(ga.AtF64(1, 1)+0.5) > 1e-12 {
		t.Errorf("Ā diag = [%g,%g], want [-1.5,-0.5]", ga.AtF64(0, 0), ga.AtF64(1, 1))
	}
}
