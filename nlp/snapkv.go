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
// Four rows per pass cuts the LOADS of agg to a quarter, while all four input rows are still read
// contiguously (PS1007 remedy b). It did NOT originally cut the stores — that claim stood here for
// a while and was wrong; see the note in the loop below for what it took and what it bought.
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
	// THE UNROLL ABOVE ONLY HALF-WORKED, and the disassembly said so: the four loads of agg[j]
	// did collapse to one, but there were still FOUR stores per iteration, not one. `for j :=
	// range winStart` ranges an INTEGER, which proves nothing about r0..r3, so each row read kept
	// a bounds check — and a store cannot be sunk past a branch that may panic, so agg[j] had to
	// be written back before every following check. Clamping the rows to agg's length and ranging
	// over agg discharges all four checks, which lets the partial sum finally stay in a register:
	// four stores become one. The four live row lengths it also frees were competing for
	// registers with the loop's own state.
	//
	// Clamping cannot change behavior: r0[j] already panicked when a row was shorter than agg, so
	// the failure domain is identical — only the panic's site moves one step earlier.
	for ; h+4 <= len(obsAttn); h += 4 {
		r0, r1 := obsAttn[h][:len(agg)], obsAttn[h+1][:len(agg)]
		r2, r3 := obsAttn[h+2][:len(agg)], obsAttn[h+3][:len(agg)]
		for j := range agg {
			// The partial sum is carried in a LOCAL, which is what actually collapses the four
			// stores to one. Discharging the bounds checks alone did not: with four separate
			// `agg[j] +=` statements the compiler must make each write observable, because it
			// cannot prove agg does not overlap r0..r3 — aliasing, not the checks, was forcing
			// the stores. That proof is available to us and not to the compiler: agg is a fresh
			// make in this function and the rows come from the caller, so they cannot overlap.
			//
			// Bit-identical: the accumulation order is unchanged, still ((((agg+r0)+r1)+r2)+r3)
			// in ascending head order over the same operands. It relies on the no-alias fact
			// above — with overlap, reading all four rows before the single store would differ
			// from interleaving them, which is exactly why the compiler refused.
			//
			// AND IT BOUGHT NOTHING MEASURABLE, which is worth stating so the next reader does
			// not spend on this axis again. The disassembly is unambiguous — four bounds checks
			// to zero, four stores to one — and BenchmarkSnapKVKeep did not move (p=0.630, n=12,
			// against an untouched control that was also flat). At seq 8192 and 8 heads each row
			// is 64 KB, so a four-row group streams 256 KB past a 64 KB agg: the loop is memory
			// bandwidth bound and the instructions removed were never the constraint
			// (MEMORY-BOUND-HIDES-CHECK-REMOVAL-TOO-001). The change is kept because it is
			// strictly fewer instructions and because the comment above it previously claimed a
			// store reduction the code did not actually perform — not because it is faster.
			v := agg[j]
			v += r0[j]
			v += r1[j]
			v += r2[j]
			v += r3[j]
			agg[j] = v
		}
	}
	for ; h < len(obsAttn); h++ {
		row := obsAttn[h][:len(agg)]
		for j := range agg {
			agg[j] += row[j]
		}
	}
	return agg
}
