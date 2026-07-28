package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Sinkhorn computes the entropically-regularized optimal-transport plan between two
// discrete distributions (Cuturi 2013, "Sinkhorn Distances: Lightspeed Computation of
// Optimal Transport", NeurIPS, arXiv:1306.0895). Given a cost matrix C [m,n], target row
// marginals r (length m) and column marginals c (length n) of equal total mass, and a
// regularization ε>0, it returns the plan
//
//	P* = argmin_P ⟨P,C⟩ − ε·H(P)   s.t.  P·1 = r,  Pᵀ·1 = c
//
// (H the entropy) via Sinkhorn-Knopp iteration on the Gibbs kernel K = exp(−C/ε): starting
// from u=v=1, it alternates u ← r ⊘ (K v) and v ← c ⊘ (Kᵀ u) for iters rounds, giving
// P = diag(u)·K·diag(v). P is nonnegative and its row/column sums converge to r/c as iters
// grows; ε→0 approaches the sharp (unregularized) transport plan, larger ε a smoother one.
// With uniform marginals this is exactly SwAV's equipartitioned cluster assignment (Caron
// et al. 2020, arXiv:2006.09882, ~3 iterations). A pure-f64 utility (SwAV applies it under
// stop-gradient to form the target assignment), not differentiable.
func Sinkhorn(cost *tensor.Tensor, r, c []float64, eps float64, iters int) (*tensor.Tensor, error) {
	if cost.Ndim() != 2 {
		return nil, fmt.Errorf("nn: Sinkhorn wants a rank-2 cost matrix, got %v", cost.Shape())
	}
	m, n := cost.Shape()[0], cost.Shape()[1]
	if len(r) != m || len(c) != n {
		return nil, fmt.Errorf("nn: Sinkhorn marginals must be r[%d] and c[%d], got r[%d] c[%d]", m, n, len(r), len(c))
	}
	if eps <= 0 {
		return nil, fmt.Errorf("nn: Sinkhorn eps must be > 0, got %g", eps)
	}
	if iters < 1 {
		return nil, fmt.Errorf("nn: Sinkhorn iters must be ≥ 1, got %d", iters)
	}
	var sr, sc float64
	for _, v := range r {
		sr += v
	}
	for _, v := range c {
		sc += v
	}
	if math.Abs(sr-sc) > 1e-9*math.Max(1, math.Abs(sr)) {
		return nil, fmt.Errorf("nn: Sinkhorn marginals must have equal total mass, got Σr=%g Σc=%g", sr, sc)
	}

	// K = exp(−(C − minC)/ε). Subtracting the minimum cost keeps the kernel in (0,1] to
	// avoid overflow; a constant shift of C leaves P unchanged (absorbed by u,v scaling).
	minC := math.Inf(1)
	for i := range m {
		for j := range n {
			if v := cost.AtF64(i, j); v < minC {
				minC = v
			}
		}
	}
	// Flat [m*n] rather than [][]float64: one allocation, no per-row pointer chase, and
	// contiguous rows so the axpy sweep below streams (PS4006).
	k := make([]float64, m*n)
	for i := range m {
		ki := k[i*n : i*n+n : i*n+n]
		for j := range n {
			ki[j] = math.Exp(-(cost.AtF64(i, j) - minC) / eps)
		}
	}

	u := make([]float64, m)
	v := make([]float64, n)
	for i := range u {
		u[i] = 1
	}
	for j := range v {
		v[j] = 1
	}
	// u = r ⊘ (K v) is register-blocked 4 ways over the OUTPUT index; v = c ⊘ (Kᵀ u) is
	// an axpy sweep. Both keep every accumulator walking its reduction axis in ascending
	// order, so the result is bit-identical to the textbook form — it is unrolling the
	// REDUCTION axis that would reassociate the adds and move the last bits.
	for range iters {
		i := 0
		for ; i+4 <= m; i += 4 { // four output rows per pass, one v[j] load feeding all four
			k0 := k[i*n : i*n+n : i*n+n]
			k1 := k[(i+1)*n : (i+1)*n+n : (i+1)*n+n]
			k2 := k[(i+2)*n : (i+2)*n+n : (i+2)*n+n]
			k3 := k[(i+3)*n : (i+3)*n+n : (i+3)*n+n]
			var a0, a1, a2, a3 float64
			for j := range n {
				vj := v[j]
				a0 += k0[j] * vj
				a1 += k1[j] * vj
				a2 += k2[j] * vj
				a3 += k3[j] * vj
			}
			for b, kv := range [4]float64{a0, a1, a2, a3} {
				if kv > 0 {
					u[i+b] = r[i+b] / kv
				}
			}
		}
		for ; i < m; i++ {
			ki := k[i*n : i*n+n : i*n+n]
			var kv float64
			for j := range n {
				kv += ki[j] * v[j]
			}
			if kv > 0 {
				u[i] = r[i] / kv
			}
		}
		// v = c ⊘ (Kᵀ u): four adjacent output columns per pass, so the cache line the
		// column walk pays for anyway serves all four accumulators.
		j := 0
		for ; j+4 <= n; j += 4 {
			var a0, a1, a2, a3 float64
			for i := range m {
				ui := u[i]
				ki := k[i*n+j : i*n+j+4 : i*n+j+4]
				a0 += ki[0] * ui
				a1 += ki[1] * ui
				a2 += ki[2] * ui
				a3 += ki[3] * ui
			}
			for b, ktu := range [4]float64{a0, a1, a2, a3} {
				if ktu > 0 {
					v[j+b] = c[j+b] / ktu
				}
			}
		}
		for ; j < n; j++ {
			var ktu float64
			for i := range m {
				ktu += k[i*n+j] * u[i]
			}
			if ktu > 0 {
				v[j] = c[j] / ktu
			}
		}
	}

	p := tensor.New(cost.Dtype(), tensor.Shape{m, n})
	for i := range m {
		for j := range n {
			p.SetF64(u[i]*k[i*n+j]*v[j], i, j)
		}
	}
	return p, nil
}
