package ref_test

import (
	"math"
	"sort"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// svdRows is the ORIGINAL row-of-slices one-sided Jacobi SVD, kept as the oracle for
// the flat rewrite. Written before the change (PROC-009): a one-ulp perturbation of
// the sweep passed the whole backend and autograd suites, so this kernel had no
// coverage at the level the rewrite touches — the fourth such kernel this session.
func svdRows(ad []float64, m, n int) (u []float64, s []float64, v []float64) {
	acol := make([][]float64, m)
	for i := range m {
		acol[i] = make([]float64, n)
		for j := range n {
			acol[i][j] = ad[i*n+j]
		}
	}
	vmat := make([][]float64, n)
	for i := range n {
		vmat[i] = make([]float64, n)
		vmat[i][i] = 1
	}
	const tol = 1e-14
	for range 100 {
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				var alpha, beta, gamma float64
				for k := range m {
					alpha += acol[k][i] * acol[k][i]
					beta += acol[k][j] * acol[k][j]
					gamma += acol[k][i] * acol[k][j]
				}
				if alpha == 0 || beta == 0 {
					continue
				}
				rel := math.Abs(gamma) / math.Sqrt(alpha*beta)
				if rel > off {
					off = rel
				}
				if rel <= tol {
					continue
				}
				zeta := (beta - alpha) / (2 * gamma)
				var t float64
				if zeta >= 0 {
					t = 1 / (zeta + math.Sqrt(1+zeta*zeta))
				} else {
					t = -1 / (-zeta + math.Sqrt(1+zeta*zeta))
				}
				c := 1 / math.Sqrt(1+t*t)
				sn := c * t
				for k := range m {
					ai, aj := acol[k][i], acol[k][j]
					acol[k][i] = c*ai - sn*aj
					acol[k][j] = sn*ai + c*aj
				}
				for k := range n {
					vi, vj := vmat[k][i], vmat[k][j]
					vmat[k][i] = c*vi - sn*vj
					vmat[k][j] = sn*vi + c*vj
				}
			}
		}
		if off <= tol {
			break
		}
	}
	sigma := make([]float64, n)
	for j := range n {
		var nrm float64
		for k := range m {
			nrm += acol[k][j] * acol[k][j]
		}
		sigma[j] = math.Sqrt(nrm)
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(x, y int) bool { return sigma[order[x]] > sigma[order[y]] })
	u = make([]float64, m*n)
	s = make([]float64, n)
	v = make([]float64, n*n)
	for jj, j := range order {
		s[jj] = sigma[j]
		if sigma[j] > 0 {
			for k := range m {
				u[k*n+jj] = acol[k][j] / sigma[j]
			}
		}
		for k := range n {
			v[k*n+jj] = vmat[k][j]
		}
	}
	return u, s, v
}

func TestSVDFlatBitIdenticalToRows(t *testing.T) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	for _, sz := range [][2]int{{1, 1}, {3, 2}, {5, 5}, {12, 4}, {16, 16}} {
		m, n := sz[0], sz[1]
		a := bench.RandF64(tensor.Shape{m, n}, uint64(m*100+n))
		ad := make([]float64, m*n)
		for i := range m {
			for j := range n {
				ad[i*n+j] = a.AtF64(i, j)
			}
		}
		out, err := backend.Execute(ctx, backend.OpSVD, []*tensor.Tensor{a}, nil)
		if err != nil {
			t.Fatal(err)
		}
		wu, ws, wv := svdRows(ad, m, n)
		for k := range m {
			for j := range n {
				if got := out[0].AtF64(k, j); math.Float64bits(got) != math.Float64bits(wu[k*n+j]) {
					t.Fatalf("%dx%d U[%d,%d]: got %v want %v", m, n, k, j, got, wu[k*n+j])
				}
			}
		}
		for j := range n {
			if got := out[1].AtF64(j); math.Float64bits(got) != math.Float64bits(ws[j]) {
				t.Fatalf("%dx%d S[%d]: got %v want %v", m, n, j, got, ws[j])
			}
			for k := range n {
				if got := out[2].AtF64(k, j); math.Float64bits(got) != math.Float64bits(wv[k*n+j]) {
					t.Fatalf("%dx%d V[%d,%d]: got %v want %v", m, n, k, j, got, wv[k*n+j])
				}
			}
		}
	}
}
