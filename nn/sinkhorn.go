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
	k := make([][]float64, m)
	// K = exp(−(C−minC)/ε) is one independent transcendental per cell, and each row writes only
	// k[i], so fan the m rows across cores — bit-identical, every cell is a pure function of its
	// own C[i,j].
	parallelRows(m, n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			//perfscan:ignore PS2008,PS3064 resource-only per-row alloc; own desc says no speedup | resource-only [][]->flat allocs, no wallclock
			k[i] = make([]float64, n)
			//perfscan:ignore PS1001,PS4006 AtF64 inside exp-per-element parallel K-build; transcendental-dominated | row-hoist on exp-dominated parallel
			for j := range n {
				//perfscan:ignore PS3016 range/hoist on exp-per-element loop; exp dominates
				k[i][j] = math.Exp(-(cost.AtF64(i, j) - minC) / eps)
			}
		}
	})

	u := make([]float64, m)
	v := make([]float64, n)
	for i := range u {
		u[i] = 1
	}
	for j := range v {
		v[j] = 1
	}
	for range iters {
		// u = r ⊘ (K v): each u[i] is an independent dot of row k[i] with the (read-only) v, and
		// writes only u[i] → fan the rows across cores. The parallelRows barrier completes the full
		// u-update before the v-update reads it. Bit-identical: each u[i] sums over j in the same
		// ascending order.
		parallelRows(m, n, func(lo, hi int) {
			//perfscan:ignore PS6010 u-update dot row-contiguous, v cache-resident; register-block marginal
			for i := lo; i < hi; i++ {
				var kv float64
				//perfscan:ignore PS3010 same contiguous u-update dot; reassoc non-bit-identical, minor
				for j := range n {
					kv += k[i][j] * v[j]
				}
				if kv > 0 {
					u[i] = r[i] / kv
				}
			}
		})
		// v = c ⊘ (Kᵀ u): each v[j] folds i in ascending order from +0. The naive nest walked k
		// COLUMN-major (k[i][j] inner over i strides a full row per step, thrashing cache once k
		// exceeds L2). Interchange to row-outer: read each contiguous row k[i] once and scatter
		// u[i]·k[i][j] into a ktu accumulator over this worker's j-range. Bit-identical — each v[j]
		// still sums i ascending from +0 into its own accumulator; disjoint j-range writes, no
		// cross-worker reduction (§V-sinkhorn ascending-i order holds).
		parallelRows(n, m, func(lo, hi int) {
			ktu := make([]float64, hi-lo)
			for i := range m {
				ui := u[i]
				ki := k[i]
				for j := lo; j < hi; j++ {
					ktu[j-lo] += ki[j] * ui
				}
			}
			for j := lo; j < hi; j++ {
				if ktu[j-lo] > 0 {
					v[j] = c[j] / ktu[j-lo]
				}
			}
		})
	}

	p := tensor.New(cost.Dtype(), tensor.Shape{m, n})
	// P[i,j] = u[i]·K[i,j]·v[j]: each cell independent, each row writes only its own cells of p.
	parallelRows(m, n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			//perfscan:ignore PS1001 one-time parallel output store, small share; dtype-var dual path
			for j := range n {
				p.SetF64(u[i]*k[i][j]*v[j], i, j)
			}
		}
	})
	return p, nil
}
