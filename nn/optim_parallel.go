package nn

import "github.com/jxsl13/goai/internal/parallel"

// parStepMinElems is the per-parameter element count above which an optimizer's element-wise
// update is fanned across cores. Below it the parameter fits cache and a single core already
// saturates its memory bandwidth, so the goroutine/WaitGroup fan-out would only add overhead;
// the threshold sits above L2 (2 MB of f64 state / 1 MB f32) so ONLY genuinely DRAM-resident
// parameters — the embedding, lm_head and FFN matrices that dominate an LLM's numel — take the
// parallel path, and small models stay serial (zero regression).
//
// The motivation: forward and backward already fan every GEMM across all cores (parallelWork),
// so a single-threaded optimizer step is amplified to a large fraction of the PARALLEL step wall
// time — parallelizing it is the bit-exact analog of PyTorch's fused/foreach optimizer.
var parStepMinElems = 1 << 18

// parStep runs body over the index range [0,n): fanned into disjoint contiguous chunks across the
// worker pool when n is large enough to pay, else inline on the caller. body MUST write only
// per-element (index-i) state — moments and the parameter at index i — so the partition is
// BIT-IDENTICAL to the serial loop: every element computes exactly what it would alone, and no
// value is accumulated across chunks (the chunk ORDER is unspecified).
func parStep(n int, body func(lo, hi int)) {
	if n >= parStepMinElems {
		parallel.Rows(n, body)
		return
	}
	body(0, n)
}

// parSumSqF64 returns Σ gf[i]² — fanned across cores (deterministic per-chunk partials) when gf is
// large enough to pay, else a serial left-to-right sum. The parallel path REASSOCIATES the sum, so
// it is not bit-identical to the serial one; callers must tolerate ~1 ULP (ClipGradNorm's global
// norm is tolerance-checked, and numpy/torch reduce the same way).
func parSumSqF64(gf []float64) float64 {
	if len(gf) < parStepMinElems {
		var s float64
		for _, v := range gf {
			s += v * v
		}
		return s
	}
	return parallel.SumF64(len(gf), func(lo, hi int) float64 {
		var s float64
		for i := lo; i < hi; i++ {
			s += gf[i] * gf[i]
		}
		return s
	})
}

// parCountNonzeroF64 returns the count of nonzero entries of u, fanned across cores (deterministic
// integer partials that sum BIT-EXACTLY in float64 — counts are integers < 2^53, no reassociation
// error). CautiousAdamW uses it for the kept-coordinate count: after the masked build, u[i]!=0 iff the
// coord passed the cautious sign test (ui·g>0 ⇒ ui!=0; masked ⇒ ui=0), so this equals the serial count.
func parCountNonzeroF64(u []float64) float64 {
	if len(u) < parStepMinElems {
		var k float64
		for _, x := range u {
			if x != 0 {
				k++
			}
		}
		return k
	}
	return parallel.SumF64(len(u), func(lo, hi int) float64 {
		var k float64
		for i := lo; i < hi; i++ {
			if u[i] != 0 {
				k++
			}
		}
		return k
	})
}

// parStepRows fans a ROW-chunked body across cores when the TOTAL element count n (not the row count)
// clears parStepMinElems — for 2D optimizer passes whose per-row work writes a disjoint output row, so
// the matrix parallelizes on its element count exactly as the flat parStep does and stays BIT-IDENTICAL
// to the serial loop. Below the threshold, or with a single row, it runs inline on the caller.
func parStepRows(rows, n int, body func(lo, hi int)) {
	if n >= parStepMinElems && rows > 1 {
		parallel.Rows(rows, body)
		return
	}
	body(0, rows)
}

// parSumF64x2 runs a fused parallel map-reduce over [0,n): chunk writes its per-index state — disjoint
// index-i writes, so BIT-IDENTICAL to the serial loop — and returns two partial sums, whose totals are
// reassociated across cores (tol ~1 ULP, like parSumSq) when n clears parStepMinElems, else a serial
// single-chunk pass. Prodigy uses it to update m/v/s AND accumulate the d-estimate's ⟨g,x₀−x⟩ and ‖s‖₁
// in ONE memory pass (splitting into parStep+two reductions would double the DRAM traffic on a
// bandwidth-bound kernel). The per-index writes stay bit-exact; only the two summed scalars relax.
func parSumF64x2(n int, chunk func(lo, hi int) (float64, float64)) (float64, float64) {
	if n < parStepMinElems {
		return chunk(0, n)
	}
	return parallel.SumF64x2(n, chunk)
}

// parSumSqF32 is parSumSqF64 for an f32 grad, accumulating in float64 exactly as the serial path does.
func parSumSqF32(gf []float32) float64 {
	if len(gf) < parStepMinElems {
		var s float64
		for _, gv := range gf {
			v := float64(gv)
			s += v * v
		}
		return s
	}
	return parallel.SumF64(len(gf), func(lo, hi int) float64 {
		var s float64
		for i := lo; i < hi; i++ {
			v := float64(gf[i])
			s += v * v
		}
		return s
	})
}
