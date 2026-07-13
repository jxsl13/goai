package nlp

import (
	"math"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// Sequence packing for efficient LLM training (Krell, Kosec, Perez & Fitzgibbon 2021,
// "Efficient Sequence Packing without Cross-contamination", arXiv:2107.02027; the standard
// packed-pretraining recipe). Padding every short document to the context length L wastes
// compute on pad tokens; instead several documents are concatenated into one length-L pack.
// To keep this numerically equivalent to unpacked training, two corrections avoid
// "cross-contamination" between the co-packed documents: a BLOCK-DIAGONAL attention mask (a
// token attends only within its own document) and POSITION IDS that reset per document.

// PackSequences bin-packs seqs into blocks of length ≤ maxLen using first-fit-decreasing,
// returning the packed token blocks and, for each block, the local document index of every
// position (0 for the first document in the block, 1 for the next, …). Sequences longer than
// maxLen are truncated to maxLen; empty sequences are skipped.
func PackSequences(seqs [][]int, maxLen int) (blocks, docIDs [][]int) {
	if maxLen <= 0 {
		return nil, nil
	}
	order := make([]int, len(seqs))
	for i := range order {
		order[i] = i
	}
	// first-fit-DECREASING: place longer sequences first for tighter packing.
	sort.SliceStable(order, func(a, b int) bool { return len(seqs[order[a]]) > len(seqs[order[b]]) })

	var rem []int // remaining capacity per block
	for _, idx := range order {
		s := seqs[idx]
		if len(s) > maxLen {
			s = s[:maxLen] // truncate over-long documents
		}
		if len(s) == 0 {
			continue
		}
		placed := -1
		for b := range rem {
			if rem[b] >= len(s) {
				placed = b
				break
			}
		}
		if placed == -1 {
			blocks = append(blocks, nil)
			docIDs = append(docIDs, nil)
			rem = append(rem, maxLen)
			placed = len(blocks) - 1
		}
		did := 0
		if n := len(docIDs[placed]); n > 0 {
			did = docIDs[placed][n-1] + 1
		}
		for _, t := range s {
			blocks[placed] = append(blocks[placed], t)
			docIDs[placed] = append(docIDs[placed], did)
		}
		rem[placed] -= len(s)
	}
	return blocks, docIDs
}

// DocumentCausalMask builds the [L,L] additive attention mask for a packed block given the
// per-position document ids: entry [i,j] is 0 when token j is in the same document as token
// i AND j ≤ i (intra-document causal), and −∞ otherwise. This is block-diagonal — each
// document is an independent causal block — so no token attends across a document boundary.
// A block of a single document reproduces the ordinary causal mask.
func DocumentCausalMask(docIDs []int) *tensor.Tensor {
	n := len(docIDs)
	m := tensor.New(tensor.F64, tensor.Shape{n, n})
	neg := math.Inf(-1)
	for i := range n {
		for j := range n {
			if docIDs[i] == docIDs[j] && j <= i {
				m.SetF64(0, i, j)
			} else {
				m.SetF64(neg, i, j)
			}
		}
	}
	return m
}

// DocumentPositions returns the position ids for a packed block: they restart at 0 at each
// document boundary, so every document sees positions as if decoded alone (e.g. document
// ids [0,0,0,1,1] → [0,1,2,0,1]). Documents are assumed contiguous within the block, as
// produced by PackSequences.
func DocumentPositions(docIDs []int) []int {
	pos := make([]int, len(docIDs))
	for i := range docIDs {
		if i > 0 && docIDs[i] == docIDs[i-1] {
			pos[i] = pos[i-1] + 1
		}
	}
	return pos
}
