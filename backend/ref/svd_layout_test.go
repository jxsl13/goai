package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestSVDColumnMajorIsBitIdentical locks the column-major working buffers to the row-major
// arithmetic they replaced. The reference below is the original kernel's numerical core written out
// verbatim — same sweeps, same tolerance, same rotation formulas, same ordering — differing only in
// where a column lives.
//
// Exact bits on all three outputs. A Jacobi SVD is iterative, so any drift compounds across sweeps:
// a tolerance comparison here would hide a layout bug that changed which rotation fired.
func TestSVDColumnMajorIsBitIdentical(t *testing.T) {
	for _, sz := range [][2]int{{1, 1}, {4, 2}, {8, 8}, {17, 5}, {33, 16}} {
		m, n := sz[0], sz[1]
		a := tensor.New(tensor.F64, tensor.Shape{m, n})
		src := make([]float64, m*n)
		for i := range m {
			for j := range n {
				v := math.Sin(float64(i*11+j*5))*1.5 + 0.25*float64((i*j)%4)
				src[i*n+j] = v
				a.SetF64(v, i, j)
			}
		}
		got, err := backend.Execute(backend.NewContext(), backend.OpSVD, []*tensor.Tensor{a}, nil)
		if err != nil {
			t.Fatalf("%dx%d: %v", m, n, err)
		}
		wu, ws, wv := svdRowMajor(m, n, src)
		// The rank is passed explicitly rather than inferred from the column count: at 1x1 the
		// singular-value vector and the matrices all have one column, and inferring rank from that
		// indexes a rank-2 tensor with one coordinate.
		check := func(name string, tn *tensor.Tensor, want []float64, rows, cols int, vec bool) {
			t.Helper()
			for i := range rows {
				for j := range cols {
					var g float64
					if vec {
						g = tn.AtF64(i)
					} else {
						g = tn.AtF64(i, j)
					}
					if w := want[i*cols+j]; math.Float64bits(g) != math.Float64bits(w) {
						t.Fatalf("%dx%d %s[%d,%d]: column-major %v, row-major %v — not bit-identical",
							m, n, name, i, j, g, w)
					}
				}
			}
		}
		check("U", got[0], wu, m, n, false)
		check("S", got[1], ws, n, 1, true)
		check("V", got[2], wv, n, n, false)
	}
}

// svdRowMajor is the pre-relayout numerical core, kept verbatim.
func svdRowMajor(m, n int, src []float64) (u, s, v []float64) {
	acol := make([]float64, m*n)
	copy(acol, src)
	vmat := make([]float64, n*n)
	for i := range n {
		vmat[i*n+i] = 1
	}
	const tol = 1e-14
	for range 100 {
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				var alpha, beta, gamma float64
				for k := range m {
					alpha += acol[k*n+i] * acol[k*n+i]
					beta += acol[k*n+j] * acol[k*n+j]
					gamma += acol[k*n+i] * acol[k*n+j]
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
				var tt float64
				if zeta >= 0 {
					tt = 1 / (zeta + math.Sqrt(1+zeta*zeta))
				} else {
					tt = -1 / (-zeta + math.Sqrt(1+zeta*zeta))
				}
				c := 1 / math.Sqrt(1+tt*tt)
				sn := c * tt
				for k := range m {
					ai, aj := acol[k*n+i], acol[k*n+j]
					acol[k*n+i] = c*ai - sn*aj
					acol[k*n+j] = sn*ai + c*aj
				}
				for k := range n {
					vi, vj := vmat[k*n+i], vmat[k*n+j]
					vmat[k*n+i] = c*vi - sn*vj
					vmat[k*n+j] = sn*vi + c*vj
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
			nrm += acol[k*n+j] * acol[k*n+j]
		}
		sigma[j] = math.Sqrt(nrm)
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	// stable descending sort by sigma, matching the kernel
	for i := 1; i < n; i++ {
		for j := i; j > 0 && sigma[order[j]] > sigma[order[j-1]]; j-- {
			order[j], order[j-1] = order[j-1], order[j]
		}
	}
	u, s, v = make([]float64, m*n), make([]float64, n), make([]float64, n*n)
	for jj, j := range order {
		s[jj] = sigma[j]
		if sigma[j] > 0 {
			for k := range m {
				u[k*n+jj] = acol[k*n+j] / sigma[j]
			}
		}
		for k := range n {
			v[k*n+jj] = vmat[k*n+j]
		}
	}
	return u, s, v
}
