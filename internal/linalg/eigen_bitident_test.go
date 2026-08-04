package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// symEigRows is the ORIGINAL cyclic-Jacobi implementation, retained as a trusted
// (unconditionally-stable) NUMERIC reference oracle for the tridiag+QL SymEig. The two
// algorithms are NOT bit-identical, so the cross-check below compares eigenvalues to ~1e-9
// and validates reconstruction/orthonormality of the new solver directly, rather than
// bit-for-bit. Do not "optimize" this oracle — its value is being a second, independent
// implementation.
func symEigRows(a [][]float64) ([]float64, [][]float64) {
	n := len(a)
	m := make([][]float64, n)
	v := make([][]float64, n)
	for i := range n {
		m[i] = append([]float64(nil), a[i]...)
		v[i] = make([]float64, n)
		v[i][i] = 1
	}
	for range 100 {
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				off += m[i][j] * m[i][j]
			}
		}
		if off < 1e-28 {
			break
		}
		for p := range n {
			for q := p + 1; q < n; q++ {
				if math.Abs(m[p][q]) < 1e-300 {
					continue
				}
				theta := (m[q][q] - m[p][p]) / (2 * m[p][q])
				t := math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := range n {
					mkp, mkq := m[k][p], m[k][q]
					m[k][p] = c*mkp - s*mkq
					m[k][q] = s*mkp + c*mkq
				}
				for k := range n {
					mpk, mqk := m[p][k], m[q][k]
					m[p][k] = c*mpk - s*mqk
					m[q][k] = s*mpk + c*mqk
				}
				for k := range n {
					vkp, vkq := v[k][p], v[k][q]
					v[k][p] = c*vkp - s*vkq
					v[k][q] = s*vkp + c*vkq
				}
			}
		}
	}
	vals := make([]float64, n)
	for i := range n {
		vals[i] = m[i][i]
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	for i := range n {
		best := i
		for j := i + 1; j < n; j++ {
			if vals[order[j]] > vals[order[best]] {
				best = j
			}
		}
		order[i], order[best] = order[best], order[i]
	}
	sorted := make([]float64, n)
	vecs := make([][]float64, n)
	for k, oi := range order {
		sorted[k] = vals[oi]
		col := make([]float64, n)
		for r := range n {
			col[r] = v[r][oi]
		}
		vecs[k] = col
	}
	return sorted, vecs
}

// TestSymEigVsJacobiOracle cross-checks the tridiag+QL SymEig against the Jacobi oracle
// (eigenvalues) and validates the new solver's own reconstruction A≈VΛVᵀ and orthonormality
// VᵀV≈I. Eigenvectors are NOT compared elementwise across the two solvers (sign and
// degenerate-subspace ambiguity); reconstruction + orthonormality are the sign-free checks.
func TestSymEigVsJacobiOracle(t *testing.T) {
	for _, n := range []int{1, 2, 3, 8, 17, 64, 128} {
		rng := rand.New(rand.NewSource(int64(n)))
		a := make([][]float64, n)
		for i := range n {
			a[i] = make([]float64, n)
		}
		// symmetric, with diagonal dominance for a well-behaved SPD-ish spectrum
		for i := range n {
			for j := 0; j <= i; j++ {
				v := rng.NormFloat64()
				a[i][j], a[j][i] = v, v
			}
			a[i][i] += float64(n)
		}
		gotVals, gotVecs := SymEig(a)
		wantVals, _ := symEigRows(a)

		// scale for relative tolerances
		var nrm float64
		for k := range n {
			nrm = math.Max(nrm, math.Abs(gotVals[k]))
		}
		// (1) eigenvalues agree with the Jacobi oracle (both descending)
		for k := range n {
			if math.Abs(gotVals[k]-wantVals[k]) > 1e-9*math.Max(1, nrm) {
				t.Fatalf("n=%d val[%d]: qtl %v vs jacobi %v", n, k, gotVals[k], wantVals[k])
			}
		}
		// (2) orthonormality: VᵀV ≈ I  (vecs[k] is the k-th eigenvector, length n)
		for i := range n {
			for j := i; j < n; j++ {
				var dot float64
				for r := range n {
					dot += gotVecs[i][r] * gotVecs[j][r]
				}
				want := 0.0
				if i == j {
					want = 1.0
				}
				if math.Abs(dot-want) > 1e-10 {
					t.Fatalf("n=%d VᵀV[%d,%d]=%v want %v", n, i, j, dot, want)
				}
			}
		}
		// (3) reconstruction: A ≈ Σ_k λ_k v_k v_kᵀ
		for i := range n {
			for j := range n {
				var acc float64
				for k := range n {
					acc += gotVals[k] * gotVecs[k][i] * gotVecs[k][j]
				}
				if math.Abs(acc-a[i][j]) > 1e-10*math.Max(1, nrm) {
					t.Fatalf("n=%d recon[%d,%d]=%v want %v", n, i, j, acc, a[i][j])
				}
			}
		}
	}
}

// TestSymEigClusteredSpectrum exercises the pathology that made cyclic Jacobi blow up:
// tightly-clustered eigenvalues. tridiag+QL must still reconstruct A and stay orthonormal.
func TestSymEigClusteredSpectrum(t *testing.T) {
	const n = 64
	rng := rand.New(rand.NewSource(99))
	// random orthonormal Q via Gram-Schmidt on a random matrix
	q := make([][]float64, n)
	for i := range n {
		q[i] = make([]float64, n)
		for j := range n {
			q[i][j] = rng.NormFloat64()
		}
	}
	for i := range n { // modified Gram-Schmidt on rows
		for p := 0; p < i; p++ {
			var d float64
			for r := range n {
				d += q[i][r] * q[p][r]
			}
			for r := range n {
				q[i][r] -= d * q[p][r]
			}
		}
		var nn float64
		for r := range n {
			nn += q[i][r] * q[i][r]
		}
		nn = math.Sqrt(nn)
		for r := range n {
			q[i][r] /= nn
		}
	}
	// clustered eigenvalues: 1, 1, 1+1e-9, ... plus a couple of separated ones
	lam := make([]float64, n)
	for k := range n {
		lam[k] = 1.0 + float64(k)*1e-9
	}
	lam[0], lam[n-1] = 5.0, -3.0
	// A = Qᵀ diag(lam) Q
	a := make([][]float64, n)
	for i := range n {
		a[i] = make([]float64, n)
		for j := range n {
			var acc float64
			for k := range n {
				acc += q[k][i] * lam[k] * q[k][j]
			}
			a[i][j] = acc
		}
	}
	vals, vecs := SymEig(a)
	var nrm float64
	for k := range n {
		nrm = math.Max(nrm, math.Abs(vals[k]))
	}
	for i := range n {
		for j := range n {
			var acc float64
			for k := range n {
				acc += vals[k] * vecs[k][i] * vecs[k][j]
			}
			if math.Abs(acc-a[i][j]) > 1e-10*math.Max(1, nrm) {
				t.Fatalf("clustered recon[%d,%d]=%v want %v", i, j, acc, a[i][j])
			}
		}
	}
}
