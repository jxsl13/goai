package nn

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

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
	// Each parameter's merged tensor is independent (out[i] depends only on base[i] and
	// models[·][i]; trimTopK is pure), so the per-tensor work — dominated by the top-k
	// magnitude sort — fans out across cores. The math is unchanged, so the result is
	// identical regardless of scheduling.
	do := func(i int) {
		b := base[i]
		n := b.Numel()
		shape := b.Shape()

		// base + trimmed task vectors, flattened. Typed contiguous fast path
		// (§base-perf; model-merge sibling of SLERP/DARE): read the dense backing slice
		// directly. F32 widens through float64 exactly as AtF64 does, so bflat is
		// bit-identical to the generic walk.
		bflat := make([]float64, n)
		if bf := flatF64(b); bf != nil {
			copy(bflat, bf)
		} else if bf := flatF32(b); bf != nil {
			for p := range bflat {
				bflat[p] = float64(bf[p])
			}
		} else {
			for p := range n {
				bflat[p] = b.AtF64(tensor.Unravel(p, shape)...)
			}
		}
		keep := int(math.Ceil(density * float64(n)))
		keep = max(1, min(keep, n))
		trimmed := make([][]float64, len(models)) // [t][p] trimmed τ̂
		for t := range models {
			tau := make([]float64, n)
			mt := models[t][i]
			if mf := flatF64(mt); mf != nil {
				for p := range tau {
					tau[p] = mf[p] - bflat[p]
				}
			} else if mf := flatF32(mt); mf != nil {
				for p := range tau {
					tau[p] = float64(mf[p]) - bflat[p]
				}
			} else {
				for p := range n {
					tau[p] = mt.AtF64(tensor.Unravel(p, shape)...) - bflat[p]
				}
			}
			trimmed[t] = trimTopK(tau, keep)
		}

		// elect sign + disjoint mean per coordinate
		res := tensor.New(b.Dtype(), shape)
		rf64, rf32 := flatF64(res), flatF32(res) // res is fresh + contiguous
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
			val := bflat[p] + lambda*merged
			switch {
			case rf64 != nil:
				rf64[p] = val
			case rf32 != nil:
				rf32[p] = float32(val)
			default:
				res.SetF64(val, tensor.Unravel(p, shape)...)
			}
		}
		out[i] = res
	}
	workers := min(runtime.GOMAXPROCS(0), len(base))
	if workers > 1 {
		var next atomic.Int64
		var wg sync.WaitGroup
		for range workers {
			wg.Go(func() {
				for {
					i := int(next.Add(1)) - 1 // work-steal (the top-k sorts dominate)
					if i >= len(base) {
						return
					}
					do(i)
				}
			})
		}
		wg.Wait()
	} else {
		for i := range base {
			do(i)
		}
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
	// Dominant cost of TIESMerge. `res` below sets only the top-`keep` positions and is
	// ORDER-INDEPENDENT (each idx[k] is a distinct destination), so only the top-keep SET
	// is needed, not its sorted order — a quickselect partitions idx[:keep] to that set in
	// average O(n) instead of the O(n log n) full sort. Rank by descending |v| with a strict
	// TOTAL order (on |v| ties break by ascending index): the top-keep set is then uniquely
	// determined, so the result is bit-identical to the previous full sort's idx[:keep].
	less := func(a, b int) bool {
		xa, xb := math.Abs(v[a]), math.Abs(v[b])
		if xa != xb {
			return xa > xb
		}
		return a < b
	}
	selectTopK(idx, keep, less)
	res := make([]float64, n)
	for k := range keep {
		res[idx[k]] = v[idx[k]]
	}
	return res
}

// selectTopK partitions a in place so that a[:k] holds the k elements that rank BEFORE the
// rest under less (a ranks before b ⟺ less(a,b)), in unspecified order — a Hoare quickselect,
// average O(len(a)). less MUST be a strict total order (no two elements compare equal), which
// makes the retained set unique and independent of pivot choice. Small ranges are finished by
// insertion sort. No-op when k ≤ 0 or k ≥ len(a).
func selectTopK(a []int, k int, less func(x, y int) bool) {
	if k <= 0 || k >= len(a) {
		return
	}
	lo, hi := 0, len(a)-1
	for lo < hi {
		if hi-lo < 12 { // insertion sort the small range, then the split point is settled
			for i := lo + 1; i <= hi; i++ {
				for j := i; j > lo && less(a[j], a[j-1]); j-- {
					a[j], a[j-1] = a[j-1], a[j]
				}
			}
			return
		}
		mid := lo + (hi-lo)/2
		// median-of-three (lo, mid, hi) so a[lo] ≤ a[mid] ≤ a[hi] under less; pivot = a[mid].
		// A median pivot bounds the inner scans (a[lo] ≤ pivot ≤ a[hi]) and avoids the O(n²)
		// blow-up on already-sorted / organ-pipe input.
		if less(a[mid], a[lo]) {
			a[lo], a[mid] = a[mid], a[lo]
		}
		if less(a[hi], a[lo]) {
			a[lo], a[hi] = a[hi], a[lo]
		}
		if less(a[hi], a[mid]) {
			a[mid], a[hi] = a[hi], a[mid]
		}
		pivot := a[mid]
		i, j := lo, hi
		for {
			for less(a[i], pivot) {
				i++
			}
			for less(pivot, a[j]) {
				j--
			}
			if i >= j {
				break
			}
			a[i], a[j] = a[j], a[i]
			i++
			j--
		}
		// [lo, j] all rank ≤ pivot, [j+1, hi] all rank ≥ pivot. Recurse into the side holding k.
		if k <= j {
			hi = j
		} else {
			lo = j + 1
		}
	}
}
