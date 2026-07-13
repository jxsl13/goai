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

// symMat builds a random symmetric n×n matrix with well-separated eigenvalues
// (a diagonal spread + small symmetric off-diagonal) so F = 1/(w_j−w_i) stays finite.
func symMat(rng *rand.Rand, n int) [][]float64 {
	a := make([][]float64, n)
	for i := range n {
		a[i] = make([]float64, n)
	}
	for i := range n {
		a[i][i] = float64(3*i) + 1 + 0.3*rng.NormFloat64() // spread diagonal → distinct eigenvalues
		for j := i + 1; j < n; j++ {
			v := 0.3 * rng.NormFloat64()
			a[i][j] = v
			a[j][i] = v
		}
	}
	return a
}

// eighFwd returns (w, V) as slices, with each eigenvector column sign-aligned to
// ref (flip if its dot with ref's column is negative) so finite differences stay
// continuous across the decomposition.
func eighFwd(a [][]float64, ref [][]float64) (w []float64, v [][]float64, err error) {
	out, e := backend.Execute(backend.NewContext(), backend.OpEigh, []*tensor.Tensor{matTensor(a)}, nil)
	if e != nil {
		return nil, nil, e
	}
	n := len(a)
	w = make([]float64, n)
	v = make([][]float64, n)
	for i := range n {
		w[i] = out[0].AtF64(i)
		v[i] = make([]float64, n)
		for k := range n {
			v[i][k] = out[1].AtF64(i, k)
		}
	}
	if ref != nil {
		for k := range n {
			var dot float64
			for i := range n {
				dot += v[i][k] * ref[i][k]
			}
			if dot < 0 {
				for i := range n {
					v[i][k] = -v[i][k]
				}
			}
		}
	}
	return w, v, nil
}

// §V2: the Ionescu-2015 eigh gradient matches central finite differences, with both
// w̄ and V̄ active. A is symmetric (perturbed symmetrically), eigenvectors are
// sign-aligned to the base decomposition, and the check is per symmetric entry-pair.
func TestEighGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 6))
	const h = 1e-6
	for _, n := range []int{2, 3, 4} {
		a := symMat(rng, n)
		// base decomposition (reference eigenvector signs)
		_, vref, err := eighFwd(a, nil)
		if err != nil {
			t.Fatalf("n=%d base: %v", n, err)
		}
		wbar := make([]float64, n)
		wv := make([][]float64, n)
		for i := range n {
			wbar[i] = rng.NormFloat64()
			wv[i] = make([]float64, n)
			for j := range n {
				wv[i][j] = rng.NormFloat64()
			}
		}
		loss := func(a [][]float64) float64 {
			w, v, _ := eighFwd(a, vref)
			var s float64
			for k := range n {
				s += wbar[k] * w[k]
			}
			for i := range n {
				for k := range n {
					s += wv[i][k] * v[i][k]
				}
			}
			return s
		}

		// analytic gradient via the tape
		at := matTensor(a)
		tape := autograd.NewTape()
		out, err := backend.Execute(tape.Context(), backend.OpEigh, []*tensor.Tensor{at}, nil)
		if err != nil {
			t.Fatalf("n=%d forward: %v", n, err)
		}
		// loss = Σ w̄⊙w + Σ WV⊙V, built on the tape
		wbTensor := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			wbTensor.SetF64(wbar[i], i)
		}
		wvTensor := matTensor(wv)
		pw, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[0], wbTensor}, nil)
		sw, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{pw[0]}, nil)
		pv, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[1], wvTensor}, nil)
		sv, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{pv[0]}, nil)
		lossT, _ := backend.Execute(tape.Context(), backend.OpAdd, []*tensor.Tensor{sw[0], sv[0]}, nil)
		if err := tape.Backward(lossT[0]); err != nil {
			t.Fatalf("n=%d backward: %v", n, err)
		}
		abar := tape.Grad(at)
		if abar == nil {
			t.Fatalf("n=%d: nil gradient", n)
		}
		// symmetric-pair finite differences
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
				num := (loss(ap) - loss(am)) / (2 * h)
				var ana float64
				if i == j {
					ana = abar.AtF64(i, i)
				} else {
					ana = abar.AtF64(i, j) + abar.AtF64(j, i)
				}
				if math.Abs(num-ana) > 1e-3*math.Max(1, math.Abs(num)) {
					t.Errorf("n=%d Ā[%d,%d]: numeric %.8g vs analytic %.8g", n, i, j, num, ana)
				}
			}
		}
	}
}

// eigenvalue-only closed form: with V̄=0, Ā = V·diag(w̄)·Vᵀ. For a diagonal A the
// eigenvectors are ±e_k, so Ā is diagonal with Ā_kk = w̄ for the matched eigenvalue.
func TestEighGradEigenvaluesOnly(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{5, 0, 0, 2}) // eigenvalues 2 (k=0),5 (k=1)
	tape := autograd.NewTape()
	out, _ := backend.Execute(tape.Context(), backend.OpEigh, []*tensor.Tensor{a}, nil)
	// loss = Σ w  ⇒ w̄ = [1,1] ⇒ Ā = V·I·Vᵀ = I (diagonal ones).
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(a)
	for i := range 2 {
		for j := range 2 {
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(g.AtF64(i, j)-want) > 1e-9 {
				t.Errorf("Ā[%d,%d] = %g, want %g", i, j, g.AtF64(i, j), want)
			}
		}
	}
}
