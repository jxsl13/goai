package nn

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// TIESMerge merges N models fine-tuned from a shared pretrained base into a single
// model, resolving parameter interference (Yadav et al. 2023, arXiv:2306.01708,
// "TrIm, Elect Sign & Merge", §R123). base and every entry of models are parallel
// lists of parameter tensors with matching shapes/dtypes.
//
// For each parameter the algorithm forms task vectors τ_t = θ_t − θ_init, then:
//
//	TRIM:   per tensor, keep only the top-`density` fraction of entries by
//	        magnitude (the rest reset to 0) — drops the small, redundant changes.
//	ELECT:  for each coordinate the sign γ = sign(Σ_t τ̂_t) is elected (the sign
//	        whose trimmed values have the greater total magnitude wins).
//	MERGE:  average ONLY the trimmed values that agree with the elected sign
//	        (disjoint mean — divide by the count of agreeing entries, not N);
//	        coordinates with no agreeing entry stay 0.
//
// The result is θ_init + λ·τ_merged. density ∈ (0,1] (paper keeps k=20% → 0.2);
// lambda scales the merged task vector (paper λ≈1, tuned in [0.8,1.8]). Inputs
// are not mutated. Merging conflicting sign-updates this way beats a plain
// average of task vectors, which cancels opposing edits.
func TIESMerge(base []*tensor.Tensor, models [][]*tensor.Tensor, density, lambda float64) ([]*tensor.Tensor, error) {
	if len(models) == 0 {
		return nil, fmt.Errorf("nn: TIESMerge needs ≥1 model")
	}
	if density <= 0 || density > 1 {
		return nil, fmt.Errorf("nn: TIESMerge density must be in (0,1], got %g", density)
	}
	for t, m := range models {
		if len(m) != len(base) {
			return nil, fmt.Errorf("nn: TIESMerge model %d has %d params, base has %d", t, len(m), len(base))
		}
		for i := range base {
			if !m[i].Shape().Equal(base[i].Shape()) {
				return nil, fmt.Errorf("nn: TIESMerge model %d param %d shape %v != base %v", t, i, m[i].Shape(), base[i].Shape())
			}
			if m[i].Dtype() != base[i].Dtype() {
				return nil, fmt.Errorf("nn: TIESMerge model %d param %d dtype %v != base %v", t, i, m[i].Dtype(), base[i].Dtype())
			}
		}
	}

	out := make([]*tensor.Tensor, len(base))
	for i := range base {
		b := base[i]
		n := b.Numel()
		shape := b.Shape()

		// base + trimmed task vectors, flattened
		bflat := make([]float64, n)
		for p := range n {
			bflat[p] = b.AtF64(tensor.Unravel(p, shape)...)
		}
		keep := int(math.Ceil(density * float64(n)))
		keep = max(1, min(keep, n))
		trimmed := make([][]float64, len(models)) // [t][p] trimmed τ̂
		for t := range models {
			tau := make([]float64, n)
			mt := models[t][i]
			for p := range n {
				tau[p] = mt.AtF64(tensor.Unravel(p, shape)...) - bflat[p]
			}
			trimmed[t] = trimTopK(tau, keep)
		}

		// elect sign + disjoint mean per coordinate
		res := tensor.New(b.Dtype(), shape)
		for p := range n {
			var sum float64
			for t := range models {
				sum += trimmed[t][p]
			}
			var merged float64
			if sum != 0 { // γ = sign(sum); sum==0 → no elected sign → 0
				want := math.Signbit(sum) // true = negative
				var agg float64
				var cnt int
				for t := range models {
					v := trimmed[t][p]
					if v != 0 && math.Signbit(v) == want {
						agg += v
						cnt++
					}
				}
				if cnt > 0 {
					merged = agg / float64(cnt)
				}
			}
			res.SetF64(bflat[p]+lambda*merged, tensor.Unravel(p, shape)...)
		}
		out[i] = res
	}
	return out, nil
}

// trimTopK returns a copy of v with all but the `keep` largest-magnitude entries
// reset to 0 (ties broken by index order, keeping exactly `keep` nonzeros where
// the originals were nonzero).
func trimTopK(v []float64, keep int) []float64 {
	n := len(v)
	if keep >= n {
		return append([]float64(nil), v...)
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		return math.Abs(v[idx[a]]) > math.Abs(v[idx[b]])
	})
	res := make([]float64, n)
	for k := range keep {
		res[idx[k]] = v[idx[k]]
	}
	return res
}
