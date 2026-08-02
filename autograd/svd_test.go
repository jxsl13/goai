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

// svdFwd runs OpSVD and returns U,s,V as slices. When refU is non-nil, each column k
// of U and V is sign-flipped together (by sign of U[:,k]·refU[:,k]) so the (U,V) gauge
// stays continuous across the decomposition — required for finite-difference validity.
func svdFwd(a [][]float64, refU [][]float64) (u [][]float64, s []float64, v [][]float64, err error) {
	m, n := len(a), len(a[0])
	out, e := backend.Execute(backend.NewContext(), backend.OpSVD, []*tensor.Tensor{rectTensor(a)}, nil)
	if e != nil {
		return nil, nil, nil, e
	}
	u = make([][]float64, m)
	for i := range m {
		u[i] = make([]float64, n)
		for k := range n {
			u[i][k] = out[0].AtF64(i, k)
		}
	}
	s = make([]float64, n)
	for k := range n {
		s[k] = out[1].AtF64(k)
	}
	v = make([][]float64, n)
	for i := range n {
		v[i] = make([]float64, n)
		for k := range n {
			v[i][k] = out[2].AtF64(i, k)
		}
	}
	if refU != nil {
		for k := range n {
			var dot float64
			for i := range m {
				dot += u[i][k] * refU[i][k]
			}
			if dot < 0 {
				for i := range m {
					u[i][k] = -u[i][k]
				}
				for i := range n {
					v[i][k] = -v[i][k] // flip V[:,k] together to keep A = UΣVᵀ
				}
			}
		}
	}
	return u, s, v, nil
}

// §V2: the Townsend-2016 SVD gradient matches central finite differences, for square
// (m=n) and tall (m>n) matrices, with Ū, s̄, V̄ all active. Also exercises the
// multi-output tape path (U,s,V all feed the loss).
func TestSVDGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(52, 9))
	const h = 1e-6
	// 40x32 is here for the row splits inside the rule: it clears their work gates (m*n^2 and n^3
	// against 1<<15) so the parallel path is checked against finite differences, which is an
	// INDEPENDENT oracle. The bit-parity test beside it compares the split against itself under
	// GOMAXPROCS(1) and cannot see a row that both arms skip — the mutations proved exactly that.
	shapes := [][2]int{{2, 2}, {3, 3}, {4, 2}, {5, 3}, {40, 32}}
	for _, sh := range shapes {
		m, n := sh[0], sh[1]
		a := randRect(rng, m, n)
		urefBase, _, _, err := svdFwd(a, nil)
		if err != nil {
			t.Fatalf("%dx%d base: %v", m, n, err)
		}
		ub := randRect(rng, m, n)
		vb := randRect(rng, n, n)
		sb := make([]float64, n)
		for i := range n {
			sb[i] = rng.NormFloat64()
		}
		loss := func(a [][]float64) float64 {
			u, s, v, _ := svdFwd(a, urefBase)
			var acc float64
			for i := range m {
				for k := range n {
					acc += ub[i][k] * u[i][k]
				}
			}
			for k := range n {
				acc += sb[k] * s[k]
			}
			for i := range n {
				for k := range n {
					acc += vb[i][k] * v[i][k]
				}
			}
			return acc
		}

		// analytic gradient via the tape
		at := rectTensor(a)
		ubt := rectTensor(ub)
		vbt := rectTensor(vb)
		sbt := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			sbt.SetF64(sb[i], i)
		}
		tape := autograd.NewTape()
		out, err := backend.Execute(tape.Context(), backend.OpSVD, []*tensor.Tensor{at}, nil)
		if err != nil {
			t.Fatalf("%dx%d forward: %v", m, n, err)
		}
		pu, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[0], ubt}, nil)
		su, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{pu[0]}, nil)
		ps, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[1], sbt}, nil)
		ss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{ps[0]}, nil)
		pv, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[2], vbt}, nil)
		sv, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{pv[0]}, nil)
		l1, _ := backend.Execute(tape.Context(), backend.OpAdd, []*tensor.Tensor{su[0], ss[0]}, nil)
		loss2, _ := backend.Execute(tape.Context(), backend.OpAdd, []*tensor.Tensor{l1[0], sv[0]}, nil)
		if err := tape.Backward(loss2[0]); err != nil {
			t.Fatalf("%dx%d backward: %v", m, n, err)
		}
		abar := tape.Grad(at)
		if abar == nil {
			t.Fatalf("%dx%d: nil gradient", m, n)
		}
		for i := range m {
			for j := range n {
				ap := deepCopy(a)
				am := deepCopy(a)
				ap[i][j] += h
				am[i][j] -= h
				num := (loss(ap) - loss(am)) / (2 * h)
				if ana := abar.AtF64(i, j); math.Abs(num-ana) > 1e-3*math.Max(1, math.Abs(num)) {
					t.Errorf("%dx%d Ā[%d,%d]: numeric %.8g vs analytic %.8g", m, n, i, j, num, ana)
				}
			}
		}
	}
}

// nuclear-norm gradient: ‖A‖_* = Σ s_i has gradient U·Vᵀ (a clean closed form that
// exercises the singular-value-only path, s̄=1, Ū=V̄=0).
func TestSVDNuclearNormGrad(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{3, 2, 2, 3, 1, 1})
	tape := autograd.NewTape()
	out, _ := backend.Execute(tape.Context(), backend.OpSVD, []*tensor.Tensor{a}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out[1]}, nil) // Σ s
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(a)
	// ∂‖A‖_*/∂A = U·Vᵀ
	u, v := out[0], out[2]
	for i := range 3 {
		for j := range 2 {
			var uv float64
			for k := range 2 {
				uv += u.AtF64(i, k) * v.AtF64(j, k)
			}
			if math.Abs(g.AtF64(i, j)-uv) > 1e-9 {
				t.Errorf("Ā[%d,%d] = %.10g, want (U·Vᵀ)[%d,%d] = %.10g", i, j, g.AtF64(i, j), i, j, uv)
			}
		}
	}
}
