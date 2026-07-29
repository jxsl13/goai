package nn

import (
	"math"
	"sort"
)

// Expert Choice routing (Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Le & Laudon 2022,
// "Mixture-of-Experts with Expert Choice Routing", arXiv:2202.09368, NeurIPS'22).
// Where token-choice routing (TopKGating) has each TOKEN pick its top-k experts —
// which can overload popular experts and needs an auxiliary load-balancing loss —
// Expert Choice inverts it: each EXPERT picks its top-k tokens. Every expert then
// processes EXACTLY k tokens, so the load is PERFECTLY balanced with no auxiliary
// loss, while a token may be selected by zero, one, or several experts (heterogeneous
// per-token compute).

// ExpertChoiceCapacity returns the per-expert capacity k = round(n·c/e) for n tokens,
// e experts, and capacity factor c (the average number of experts each token is routed
// to, e.g. 1 or 2). Clamped to [1, n].
func ExpertChoiceCapacity(nTokens, nExperts int, capacityFactor float64) int {
	k := int(math.Round(float64(nTokens) * capacityFactor / float64(nExperts)))
	return min(max(k, 1), nTokens)
}

// ExpertAffinity converts router logits[token][expert] to the affinity scores
// S = softmax over experts (each token's row sums to 1), the S = softmax(X·Wg) of the
// paper. It does not mutate logits.
func ExpertAffinity(logits [][]float64) [][]float64 {
	s := make([][]float64, len(logits))
	for t, row := range logits {
		m := math.Inf(-1)
		for _, v := range row {
			if v > m {
				m = v
			}
		}
		// Hoist the destination row out of both loops (PS4006): s[t] is a separate
		// allocation, so re-indexing it per element is a pointer load each time.
		// Bit-identical — same values, same order, only the address computation moves.
		dst := make([]float64, len(row))
		s[t] = dst
		var sum float64
		for j, v := range row {
			dst[j] = math.Exp(v - m)
			sum += dst[j]
		}
		for j := range dst {
			dst[j] /= sum
		}
	}
	return s
}

// ExpertChoiceRoute performs Expert Choice routing: each expert selects its top
// `capacity` tokens by the affinity scores. scores[token][expert] is the affinity of a
// token for an expert (e.g. from ExpertAffinity). It returns, per expert, the selected
// token indices (highest affinity first) and the corresponding gate weights (the
// affinities G). Each expert selects EXACTLY min(capacity, nTokens) tokens — the
// perfect-load-balance guarantee. Ties resolve to the lower token index.
func ExpertChoiceRoute(scores [][]float64, capacity int) (tokens [][]int, gates [][]float64) {
	n := len(scores)
	if n == 0 {
		return nil, nil
	}
	e := len(scores[0])
	if capacity > n {
		capacity = n
	}
	tokens = make([][]int, e)
	gates = make([][]float64, e)
	// One id-indexed key column, refilled per expert. The comparator would otherwise do
	// scores[idx[a]][ex] — a row-pointer load plus an index — on every one of the
	// e·O(n log n) comparisons, to read a value that depends only on the token (PS3005).
	// Filling it once per expert is O(n) and leaves the comparator a single load.
	key := make([]float64, n)
	for ex := range e {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
			key[i] = scores[i][ex]
		}
		// sort token indices by this expert's affinity, descending (stable → ties by index).
		// key[id] == scores[id][ex], so this is the SAME PREDICATE as the direct form and
		// SliceStable returns the identical permutation — the tie-by-index guarantee in
		// this function's contract is preserved exactly, not merely "probably unchanged".
		// total-order comparator (key desc, then token index asc) so the faster unstable
		// sort.Slice reproduces the stable permutation exactly (indices are unique) — the
		// tie-by-lowest-index contract is preserved.
		sort.Slice(idx, func(a, b int) bool {
			if ka, kb := key[idx[a]], key[idx[b]]; ka != kb {
				return ka > kb
			}
			return idx[a] < idx[b]
		})
		tokens[ex] = append([]int(nil), idx[:capacity]...)
		gr := make([]float64, capacity)
		gates[ex] = gr
		for i, t := range tokens[ex] {
			gr[i] = scores[t][ex]
		}
	}
	return tokens, gates
}

// ExpertChoiceCombine assembles each token's output y[token] = Σ over experts that
// selected it of gate · expertOut, from the routing (tokens, gates) and the per-expert
// per-slot outputs expertOut[expert][slot][:dim] (slot i corresponds to token
// tokens[expert][i]). nTokens is the total token count; dim the feature width. Tokens
// selected by no expert stay zero.
func ExpertChoiceCombine(tokens [][]int, gates [][]float64, expertOut [][][]float64, nTokens, dim int) [][]float64 {
	y := make([][]float64, nTokens)
	for t := range y {
		y[t] = make([]float64, dim)
	}
	for ex := range tokens {
		gx, ox := gates[ex], expertOut[ex]
		for i, t := range tokens[ex] {
			g := gx[i]
			// Both operands are separate row allocations; re-indexing them per element
			// is two pointer loads per d (PS4006). Re-slicing also gives the compiler
			// one bounds check for the run instead of one per element.
			yt, oi := y[t][:dim], ox[i][:dim]
			for d := range yt {
				yt[d] += g * oi[d]
			}
		}
	}
	return y
}
