package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// prefixTreeMask assembles the full [P+N, P+N] additive attention mask for tree
// verification: prefix rows are ordinary causal, tree rows see the entire prefix
// plus their MedusaTreeMask block (§T508).
func prefixTreeMask(prefixLen int, treeMask *tensor.Tensor) *tensor.Tensor {
	n := treeMask.Shape()[0]
	seq := prefixLen + n
	neg := math.Inf(-1)
	m := tensor.New(tensor.F64, tensor.Shape{seq, seq})
	for i := range seq {
		//perfscan:ignore PS1005 per-round mask build dwarfed by verification forward; ref path
		for j := range seq {
			switch {
			case i < prefixLen: // prefix row: ordinary causal
				if j > i {
					m.SetF64(neg, i, j)
				}
			case j < prefixLen: // tree row sees the entire prefix
			default: // tree row × tree column: the tree structure
				m.SetF64(treeMask.AtF64(i-prefixLen, j-prefixLen), i, j)
			}
		}
	}
	return m
}

// treeEmbed builds the embedded input for tree verification: row i is
// TokEmb[toks[i]] + PosEmb[pos[i]] — the sum GPT.Embed produces, but with tree
// positions (prefix 0..P−1, node t at P+depth[t]) instead of 0..seq−1.
func treeEmbed(g *GPT, toks, pos []int) *tensor.Tensor {
	dim := g.TokEmb.Shape()[1]
	x := tensor.New(g.TokEmb.Dtype(), tensor.Shape{len(toks), dim})
	for i, tok := range toks {
		//perfscan:ignore PS1001 tiny per-round tree embed, dominated by forward
		for d := range dim {
			x.SetF64(g.TokEmb.AtF64(tok, d)+g.PosEmb.AtF64(pos[i], d), i, d)
		}
	}
	return x
}

// topKIdx returns the indices of the k largest entries of row, best first.
func topKIdx(row []float64, k int) []int {
	if k > len(row) {
		k = len(row)
	}
	idx := make([]int, 0, k)
	for range k {
		best := -1
		for j, v := range row {
			taken := false
			for _, t := range idx {
				if t == j {
					taken = true
					break
				}
			}
			if !taken && (best == -1 || v > row[best]) {
				best = j
			}
		}
		idx = append(idx, best)
	}
	return idx
}

// MedusaGenerateTree is MedusaGenerate with the paper's CANDIDATE TREE (§2.3):
// instead of one chain of argmax proposals, head k contributes its top-topK
// tokens and the Cartesian product forms a tree under the base's greedy token
// (always emitted). ONE verification forward scores every root-to-node path
// simultaneously — tree rows attend only to the prefix and their ancestors
// (MedusaTreeMask through the MHA.Mask seam, §T508) with positions P+depth —
// and the deepest fully TypicalAcceptance-accepted path is emitted (leftmost,
// i.e. highest-ranked candidates, on ties). topK=1 collapses onto
// MedusaGenerate's chain exactly. Deeper levels are dropped when the tree
// would not fit the context window or the remaining budget. Stats count
// path POSITIONS offered/accepted (comparable to MedusaGenerate), not tree
// nodes. Analysis-scale like MedusaGenerate: O(seq) full forwards per round,
// with the masked attention served by the reference kernel (§T461 fallback).
func MedusaGenerateTree(model *GPT, heads *MedusaHeads, prompt []int, maxNew int, epsilon, delta float64, topK int) ([]int, SpecStats, error) {
	var stats SpecStats
	if len(prompt) == 0 {
		return nil, stats, fmt.Errorf("nlp: MedusaGenerateTree needs a non-empty prompt")
	}
	if heads == nil || len(heads.W) == 0 {
		return nil, stats, fmt.Errorf("nlp: MedusaGenerateTree needs ≥1 Medusa head")
	}
	if topK < 1 {
		return nil, stats, fmt.Errorf("nlp: MedusaGenerateTree needs topK ≥ 1, got %d", topK)
	}
	if epsilon <= 0 {
		epsilon = MedusaEpsilon
	}
	if delta <= 0 {
		delta = MedusaDelta
	}
	ctx := backend.NewContext()
	out := append([]int(nil), prompt...)
	for len(out)-len(prompt) < maxNew && len(out) < model.Config.Ctx {
		hidden, err := model.ForwardHidden(ctx, out)
		if err != nil {
			return nil, stats, err
		}
		last := hidden.Shape()[0] - 1
		//perfscan:ignore PS6017 resource-only variadic exec1 (1 alloc), spec-decode ref path
		lastRow, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: last, End: last + 1}, hidden)
		if err != nil {
			return nil, stats, err
		}
		//perfscan:ignore PS6017 resource-only variadic exec1, spec-decode ref path
		mature, err := exec1(ctx, backend.OpMatMul, nil, lastRow, model.Head)
		if err != nil {
			return nil, stats, err
		}
		greedy := argmax(rowAt(mature, 0))
		hl, err := heads.Logits(ctx, lastRow)
		if err != nil {
			return nil, stats, err
		}

		// depth budget: emitted tokens ≤ room, tree must fit the context window.
		room := min(maxNew-(len(out)-len(prompt)), model.Config.Ctx-len(out))
		levels := min(len(hl), room-1)
		parents, tokens := []int{-1}, []int{greedy}
		levelNodes := []int{0}
		for h := 0; h < levels; h++ {
			cands := topKIdx(rowAt(hl[h], 0), topK)
			if width := len(levelNodes) * len(cands); len(parents)+width > model.Config.Ctx-len(out) {
				break // deeper levels dropped: the tree would not fit the window
			}
			var next []int
			for _, p := range levelNodes {
				for _, tok := range cands {
					parents = append(parents, p)
					tokens = append(tokens, tok)
					next = append(next, len(parents)-1)
				}
			}
			levelNodes = next
		}

		treeMask, depth := MedusaTreeMask(parents)
		full := prefixTreeMask(len(out), treeMask)
		toks := append(append([]int{}, out...), tokens...)
		pos := make([]int, len(toks))
		for i := range out {
			pos[i] = i
		}
		for i, d := range depth {
			pos[len(out)+i] = len(out) + d
		}
		for _, b := range model.Blocks {
			b.Attn.Mask = full
		}
		vlog, err := model.ForwardFromEmbed(ctx, treeEmbed(model, toks, pos))
		for _, b := range model.Blocks {
			b.Attn.Mask = nil
		}
		if err != nil {
			return nil, stats, err
		}

		// acceptance walk: a node is accepted iff its whole ancestor path is and the
		// base's distribution at its PARENT row admits its token; deepest leftmost wins.
		accepted := make([]bool, len(parents))
		accepted[0] = true // the greedy token, always emitted
		best := 0
		for i := 1; i < len(parents); i++ {
			p := parents[i]
			if !accepted[p] {
				continue
			}
			probs := softmaxProb(rowAt(vlog, len(out)+p))
			if TypicalAcceptance(probs, tokens[i], epsilon, delta) {
				accepted[i] = true
				if depth[i] > depth[best] {
					best = i
				}
			}
		}
		var path []int
		for n := best; n != -1; n = parents[n] {
			path = append([]int{tokens[n]}, path...)
		}
		maxDepth := 0
		for _, d := range depth {
			maxDepth = max(maxDepth, d)
		}
		stats.Proposed += maxDepth
		stats.Accepted += len(path) - 1
		out = append(out, path...)
	}
	return out, stats, nil
}
