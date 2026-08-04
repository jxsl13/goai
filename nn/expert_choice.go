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
		//perfscan:ignore PS2008 resource-only per-row make; no speedup
		dst := make([]float64, len(row))
		s[t] = dst
		var sum float64
		//perfscan:ignore PS3010 softmax sum over e experts, low trip; non-bit-identical
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
	// Experts are independent: expert ex reads only its own affinity column scores[·][ex]
	// (the rest of scores is read-only shared) and writes only tokens[ex]/gates[ex], so the
	// per-expert loop fans out over GOMAXPROCS bit-identically to the serial loop. Each worker
	// owns a private key column, selection heap, and comparator/sift closures (they capture
	// that worker's key/heap by reference); gated on e·2n (each expert makes ≥2 O(n) passes —
	// the key fill and the selection scan) so a small routing stays serial.
	parallelRows(e, 2*n, func(elo, ehi int) {
		// One id-indexed key column, refilled per expert. The comparator would otherwise do
		// scores[idx[a]][ex] — a row-pointer load plus an index — on every one of the
		// O(n log n) comparisons, to read a value that depends only on the token (PS3005).
		// Filling it once per expert is O(n) and leaves the comparator a single load.
		//perfscan:ignore PS6008 per-worker key scratch, dispatch once/call, sanctioned
		key := make([]float64, n)
		//perfscan:ignore PS6008 per-worker heap scratch, sanctioned
		heap := make([]int, 0, capacity) // size-`capacity` selection heap of token ids, reused per expert
		// worse(a,b): token id a ranks AFTER b under the total order (key desc, then id asc).
		worse := func(a, b int) bool {
			if key[a] != key[b] {
				return key[a] < key[b]
			}
			return a > b
		}
		siftDown := func(i int) {
			for {
				lo := i
				if l := 2*i + 1; l < len(heap) && worse(heap[l], heap[lo]) {
					lo = l
				}
				if r := 2*i + 2; r < len(heap) && worse(heap[r], heap[lo]) {
					lo = r
				}
				if lo == i {
					return
				}
				heap[i], heap[lo] = heap[lo], heap[i]
				i = lo
			}
		}
		for ex := elo; ex < ehi; ex++ {
			for i := 0; i < n; i++ {
				//perfscan:ignore PS3016 memory-bound column gather; ~2%, memory-bound hides bce removal
				key[i] = scores[i][ex]
			}
			// Select the top-`capacity` token ids into a min-heap whose ROOT is the WORST kept,
			// so a better token evicts it in O(log capacity) — O(n log capacity) instead of
			// sorting ALL n (and its reflect.Swapper swaps + per-expert O(n) idx alloc). The
			// comparator (key desc, then id asc) is a strict total order (ids are unique), so the
			// retained set and its sorted order are uniquely determined: bit-identical to the
			// previous full sort's idx[:capacity], preserving the tie-by-lowest-index contract (PS3006).
			heap = heap[:0]
			for i := 0; i < n; i++ {
				if len(heap) < capacity {
					heap = append(heap, i)
					for j := len(heap) - 1; j > 0; { // sift up
						p := (j - 1) / 2
						if !worse(heap[j], heap[p]) {
							break
						}
						heap[j], heap[p] = heap[p], heap[j]
						j = p
					}
				} else if capacity > 0 && worse(heap[0], i) { // root is worst kept; i is better
					heap[0] = i
					siftDown(0)
				}
			}
			//perfscan:ignore PS2008 resource-only per-expert make, no wallclock
			sel := make([]int, len(heap))
			copy(sel, heap)
			//perfscan:ignore PS3002,PS6009 sel capacity-sized short slice, radix loses; tiny share | short capacity-sized sort per expert, routing small
			sort.Slice(sel, func(a, b int) bool { return worse(sel[b], sel[a]) })
			tokens[ex] = sel
			//perfscan:ignore PS2008 resource-only per-expert make
			gr := make([]float64, capacity)
			gates[ex] = gr
			for i, t := range tokens[ex] {
				//perfscan:ignore PS1010 gather scores[t][ex] by arbitrary token, not interchangeable; capacity-sized
				gr[i] = scores[t][ex]
			}
		}
	})
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
		//perfscan:ignore PS2008,PS3064 resource-only per-row make | resource-only [][]->flat allocs
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
