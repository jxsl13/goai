package nlp

// SnapKV prompt KV-cache compression (Li, Huang, Yang, Yang, Zhang, Cai, Feng, Xu, Chen et
// al. 2024, "SnapKV: LLM Knows What You are Looking for Before Generation", NeurIPS 2024,
// arXiv:2404.14469). Unlike the decode-time eviction policies here — StreamingLLM (recent +
// sink) and H2O (accumulated attention) — SnapKV compresses the prompt KV cache once at
// PREFILL. Its insight: the last few prompt tokens (the "observation window") already know
// which earlier positions matter, so their attention picks the important prompt positions
// to keep. Per attention head it (1) sums the observation window's attention over each key,
// (2) MAX-POOLS the scores along position to keep contiguous important spans intact, and
// (3) keeps the top scoring prefix positions plus the whole observation window.

// SnapKVPool applies SnapKV's 1D max-pooling (kernel odd, stride 1, "same" length, edges
// clamped) to importance scores. Pooling spreads a peak to its neighbours so that top-k
// selection retains a cluster around each important position rather than isolated tokens —
// the ablation-critical step of SnapKV. kernel ≤ 1 returns the scores unchanged.
func SnapKVPool(scores []float64, kernel int) []float64 {
	n := len(scores)
	if kernel <= 1 || n == 0 {
		return append([]float64(nil), scores...)
	}
	half := kernel / 2
	out := make([]float64, n)
	for i := range scores {
		lo, hi := i-half, i+half
		if lo < 0 {
			lo = 0
		}
		if hi >= n {
			hi = n - 1
		}
		m := scores[lo]
		for j := lo + 1; j <= hi; j++ {
			if scores[j] > m {
				m = scores[j]
			}
		}
		out[i] = m
	}
	return out
}

// SnapKVKeep returns the prompt token indices SnapKV retains, given obsAttn — the attention
// from the observation-window queries (the last len(obsAttn) prompt tokens) to all
// len(obsAttn[0]) prompt keys, per one head. budget is the prompt KV budget
// (max_capacity_prompt); kernel is the pooling kernel (default 7). It keeps the whole
// observation window plus the top (budget − windowSize) earlier positions by pooled
// importance, returned ascending (temporal order). If the prompt already fits the budget
// all indices are kept.
func SnapKVKeep(obsAttn [][]float64, budget, kernel int) []int {
	win := len(obsAttn)
	if win == 0 {
		return nil
	}
	seq := len(obsAttn[0])
	if budget <= 0 || seq <= budget {
		all := make([]int, seq)
		for i := range all {
			all[i] = i
		}
		return all
	}
	// (1) aggregate over the PREFIX keys only (the window is always kept separately, so its
	// own attention is excluded — matching the code's `[...,:-window_size]` key slice):
	// importance(j) = Σ over observation-window queries of attention to prefix key j.
	winStart := seq - win
	agg := aggregatePrefixAttn(obsAttn, winStart)
	// (2) pool to cluster contiguous important spans.
	pooled := SnapKVPool(agg, kernel)

	// (3) always keep the observation window (the last `win` positions).
	keep := make([]bool, seq)
	for i := winStart; i < seq; i++ {
		keep[i] = true
	}
	// top (budget − win) prefix positions by pooled score (index asc on ties).
	if kPrefix := budget - win; kPrefix > 0 {
		cand := make([]int, winStart)
		for i := range cand {
			cand[i] = i
		}
		// Only the top-kPrefix SET is needed (keep-mask, emitted ascending below), not a
		// full ordering — quickselect the set instead of a full radix sort over all prefix
		// positions, resolving the equal-score boundary band by ascending index to match
		// the stable radix's kept set bit-for-bit.
		keepTopKDescInto(cand, pooled, kPrefix, keep)
	}
	out := make([]int, 0, budget)
	for i := range seq {
		if keep[i] {
			out = append(out, i)
		}
	}
	return out
}

// aggregatePrefixAttn sums the first winStart entries of every observation-window row into one
// importance vector: agg[j] = Σ_h obsAttn[h][j].
//
// FOUR ROWS PER PASS. Written one row at a time, every element of agg was loaded and stored
// once per head — at seq 8192 and 8 heads that is 57k loads and 57k stores of agg to perform
// 57k adds, so half the memory operations existed only to carry a partial sum between rows.
// Four rows per pass keeps agg[j] in a register across the group and cuts those load/store
// pairs to a quarter, while all four input rows are still read contiguously (PS1007 remedy b).
// Measured on BenchmarkSnapKVKeep (seq 8192, 8 heads, both arm orders, 24 samples):
// 137.5us -> 109.9us, -20.1%. Two rows per pass gave -17.6%, so the third and fourth row are
// worth their registers here.
//
// BIT-IDENTICAL, and by construction rather than by tolerance: the adds stay separate
// statements in ascending head order, so every agg[j] accumulates exactly the terms it did
// before, in the same order, with no reassociation and no fused multiply-add anywhere (these
// are pure adds). A head count that is not a multiple of 4 finishes in the tail loop, which is
// the original body verbatim — so counts 1, 2 and 3 run precisely the old code.
func aggregatePrefixAttn(obsAttn [][]float64, winStart int) []float64 {
	agg := make([]float64, winStart)
	h := 0
	for ; h+4 <= len(obsAttn); h += 4 {
		r0, r1, r2, r3 := obsAttn[h], obsAttn[h+1], obsAttn[h+2], obsAttn[h+3]
		for j := range winStart {
			agg[j] += r0[j]
			agg[j] += r1[j]
			agg[j] += r2[j]
			agg[j] += r3[j]
		}
	}
	for ; h < len(obsAttn); h++ {
		row := obsAttn[h]
		for j := range winStart {
			agg[j] += row[j]
		}
	}
	return agg
}
