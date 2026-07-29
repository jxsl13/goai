package nlp

import (
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// KV-cache eviction for long-context inference (§T72). Two bounded-cache policies
// keep decoding within a fixed KV budget instead of growing without limit:
// StreamingLLM attention sinks (Xiao et al. 2023, §R73) and H2O heavy-hitters
// (Zhang et al. 2023, §R73). Both select which cached token rows to retain; the
// selected rows are gathered from the append-only KVCache with GatherRows.

// StreamingKeep returns the token indices a StreamingLLM cache retains for a
// cache of n tokens: the first `sink` "attention sink" tokens (kept permanently)
// plus the most recent `recent` tokens, evicting the middle (Xiao et al. 2023).
// The sinks matter because softmax must sum to 1, so a model dumps excess
// attention onto the always-visible initial tokens regardless of content; keeping
// ~4 of them restores window-attention quality once the window rolls past the
// start. If sink+recent ≥ n nothing is evicted (all indices returned). Indices are
// ascending, so gathering them preserves temporal order (and positions are then
// reassigned by position-within-cache, per the paper).
func StreamingKeep(n, sink, recent int) []int {
	if sink < 0 {
		sink = 0
	}
	if recent < 0 {
		recent = 0
	}
	if sink+recent >= n { // nothing to evict
		keep := make([]int, n)
		for i := range keep {
			keep[i] = i
		}
		return keep
	}
	keep := make([]int, 0, sink+recent)
	for i := range sink {
		keep = append(keep, i)
	}
	for i := n - recent; i < n; i++ {
		keep = append(keep, i)
	}
	return keep
}

// H2OScores returns the H2O accumulated-attention score of each key: the column
// sum of the post-softmax attention matrix, score(j) = Σ_i A[i,j] over all query
// rows i (Zhang et al. 2023). attn is [queries][keys]; the result is length-keys.
// A key that received a lot of attention across queries is a "heavy hitter".
func H2OScores(attn [][]float64) []float64 {
	if len(attn) == 0 {
		return nil
	}
	scores := make([]float64, len(attn[0]))
	for _, row := range attn {
		for j, a := range row {
			scores[j] += a
		}
	}
	return scores
}

// H2OKeep returns the token indices an H2O cache of budget size retains given the
// per-token accumulated-attention scores: the union of the most recent `recent`
// tokens and the top heavy-hitters (highest scores) among the rest, up to budget
// (Zhang et al. 2023, Alg. 1). When the token count exceeds budget the lowest-
// scoring non-recent tokens are evicted. Returns ascending indices (temporal
// order preserved). recent is capped at budget; if n ≤ budget all are kept.
func H2OKeep(scores []float64, recent, budget int) []int {
	n := len(scores)
	if budget <= 0 || n <= budget {
		keep := make([]int, n)
		for i := range keep {
			keep[i] = i
		}
		return keep
	}
	if recent > budget {
		recent = budget
	}
	if recent < 0 {
		recent = 0
	}
	// Dense keep-mask over the [0,n) index domain (mirrors SnapKVKeep): mark the
	// kept tokens, then emit them ascending with a single linear scan. Replaces a
	// map[int]bool (budget hashed inserts + bucket alloc) plus sort.Ints — the
	// [0,n) domain makes a []bool strictly cheaper, and the scan yields the same
	// strictly-ascending kept set the map+sort did → bit-identical output.
	keep := make([]bool, n)
	for i := n - recent; i < n; i++ { // most-recent window (always kept)
		keep[i] = true
	}
	// heavy hitters among the non-recent candidates, by score desc (index asc tie)
	cand := make([]int, 0, n-recent)
	for i := range n - recent {
		cand = append(cand, i)
	}
	// Only the heavy-largest SET is needed (it becomes a keep-mask, then an
	// ascending emit) — not a full ordering — so partition with quickselect
	// instead of a full radix sort. quickselectIdxDesc leaves the heavy
	// highest-scoring indices in cand[:heavy] (order within, and membership of the
	// equal-score boundary band, arbitrary); the boundary tie band is then resolved
	// by ascending index below to reproduce the stable radix's exact kept set.
	// Only the heavy-largest SET is needed (it becomes a keep-mask emitted in ascending
	// order), not a full ordering — partition with quickselect instead of a full radix sort.
	keepTopKDescInto(cand, scores, budget-recent, keep)

	out := make([]int, 0, budget)
	for i := 0; i < n; i++ {
		if keep[i] {
			out = append(out, i)
		}
	}
	return out
}

// GatherRows selects rows idx (in order) from t[n,d] → [len(idx),d]. Used to apply
// an eviction policy's kept-index list to a cached K or V tensor (inference-only,
// no gradient path).
func GatherRows(t *tensor.Tensor, idx []int) *tensor.Tensor {
	d := t.Shape()[1]
	out := tensor.New(t.Dtype(), tensor.Shape{len(idx), d})
	for r, src := range idx {
		for j := range d {
			out.SetF64(t.AtF64(src, j), r, j)
		}
	}
	return out
}

// EvictStreaming bounds every block's K/V cache to the StreamingLLM sink+recent
// window in place (Xiao et al. 2023, §R73). Uniform across layers — the retained
// token positions are the same for every block — so a single index list applies
// to all. A no-op while the cache still fits in sink+recent.
//
// ZERO VALUE / boundary behavior: a window of zero or less — EvictStreaming(0, 0),
// or any negative argument that clamps to it — is a NO-OP, not an instruction to
// empty the cache. This is a deliberate clamp to the documented meaning above.
// Before it, the two-zero call fell through to a keep-list of no indices and
// DESTROYED every cached key and value: Go's zero value for a bounding window
// landed on "discard everything the model has seen", and a caller who computed
// recent = budget − used and got −1 back hit the same total wipe. A window that
// retains nothing is never a meaningful eviction policy, so it is read as "no
// bound requested" rather than executed.
//
// The clamp only ever ADDS retention, so every window that already bounded the
// cache (sink+recent ≥ 1) evicts exactly as before. To intentionally drop cached
// context, rebuild the cache rather than asking this policy for an empty window.
//
// POSITIONS ARE NOT RENUMBERED, and callers need not adjust anything. Eviction frees
// rows; it does not rewind the stream. The tokens that survive keep the positions they
// were encoded at, the next token keeps the position it would have had anyway, and
// [KVCache.NextPos] goes on reporting it — so a decode loop that was passing the true
// position before an eviction is still correct after one, with no bookkeeping of its
// own. What changes is only that NextPos and [KVCache.Len] stop being equal, and a
// caller who reaches for Len as the position now gets an ERROR out of
// [GPT.DecodeStep] rather than logits computed at a position the stream already used.
//
// Renumbering is not an option here even though the StreamingLLM paper calls for it
// (positions reassigned by position-within-cache): these rows are cached AFTER the
// position embedding has been folded in, so re-labelling them would not re-encode them.
// llama.cpp renumbers precisely because it can fix the rows to match, re-rotating
// cached keys with a K-shift pass; it has no such repair for learned-absolute-position
// models and, tellingly, leaves context shifting off by default for them. Bounded-cache
// decoding through this cache therefore saves memory but not positional room, and still
// ends at the model's Ctx. [Llama.StreamStep] is the path that streams without that
// limit, by caching keys BEFORE the rotation so it can re-apply it every step.
func (c *KVCache) EvictStreaming(sink, recent int) {
	if sink < 0 {
		sink = 0
	}
	if recent < 0 {
		recent = 0
	}
	n := c.Len()
	if n == 0 || sink+recent <= 0 || sink+recent >= n {
		return
	}
	keep := StreamingKeep(n, sink, recent)
	for l := range c.K {
		if c.K[l] != nil {
			c.K[l] = GatherRows(c.K[l], keep)
			c.V[l] = GatherRows(c.V[l], keep)
		}
	}
	// The evicted rows' positions stay consumed, so NextPos keeps counting the stream
	// rather than the surviving rows.
	c.pos.drop(n - len(keep))
}

// keepTopKDescInto marks keep[ci]=true for the k candidates in cand with the highest
// scores, ties broken by ASCENDING index — reproducing a stable descending-by-score
// sort's kept SET without a full sort. When the caller only needs the top-k SET (a
// keep-mask later emitted in ascending order), the full O(n)-pass radix / O(n log n) sort
// is wasted: a quickselect partition puts the top-k in cand[:k] (arbitrary order, and the
// equal-score boundary band is arbitrary), then the boundary tie band is resolved by
// ascending index to reproduce the stable sort's exact kept set. Shipped: H2O
// heavy-hitters, SnapKV prompt compression.
func keepTopKDescInto(cand []int, scores []float64, k int, keep []bool) {
	switch {
	case k <= 0:
		return
	case k >= len(cand):
		for _, ci := range cand {
			keep[ci] = true
		}
		return
	}
	quickselectIdxDesc(cand, scores, k)
	// bscore = the k-th largest score = smallest among the k selected.
	bscore := scores[cand[0]]
	for i := 1; i < k; i++ {
		if v := scores[cand[i]]; v < bscore {
			bscore = v
		}
	}
	// Every candidate scoring strictly above the boundary is unambiguously kept.
	need := k
	for i := 0; i < k; i++ {
		if ci := cand[i]; scores[ci] > bscore {
			keep[ci] = true
			need--
		}
	}
	// Fill remaining slots from the score==bscore band, lowest index first — exactly the
	// ascending-index tie-break the stable radix applied.
	if need > 0 {
		tie := make([]int, 0, need)
		for _, ci := range cand {
			if scores[ci] == bscore {
				tie = append(tie, ci)
			}
		}
		sort.Ints(tie)
		for i := 0; i < need && i < len(tie); i++ {
			keep[tie[i]] = true
		}
	}
}
