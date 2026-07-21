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
	agg := make([]float64, winStart)
	for _, row := range obsAttn {
		for j := range winStart {
			agg[j] += row[j]
		}
	}
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
		// pooled scores are non-negative (max-pooled attention), so the stable
		// radix orders them identically to the closure sort and preserves the
		// ascending-index tie-break — closure-free / O(n) on the candidate set.
		sortIdxDescByProb(cand, pooled)
		if kPrefix > len(cand) {
			kPrefix = len(cand)
		}
		for _, i := range cand[:kPrefix] {
			keep[i] = true
		}
	}
	out := make([]int, 0, budget)
	for i := range seq {
		if keep[i] {
			out = append(out, i)
		}
	}
	return out
}
