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
	// The mask is a fresh contiguous [n,n], so SetF64(v,i,j) is exactly md[i*n+j]=v —
	// write the flat storage directly and skip the per-element Unravel/dispatch.
	md := m.Storage().F64()
	neg := math.Inf(-1)
	// Rows belonging to the same document have IDENTICAL same-document masks — the predicate
	// di == docIDs[j] depends on the row only through di — so the whole fill collapses to two
	// memmoves per row once one template per distinct document id exists. That replaces n²
	// branchy stores, each carrying a data-dependent docIDs[j] load, with n² bytes of
	// straight-line copy.
	//
	// Guarded, because the template set costs D·n to build and D·n·8 bytes to hold: with a
	// distinct id per position it would be n² of extra work for nothing. Above the guard the
	// original loop runs unchanged.
	if tmplIdx, tmpl, ok := docMaskTemplates(docIDs, n, neg); ok {
		negRow := tmpl[len(tmpl)-1] // the all-neg row, appended by docMaskTemplates
		for i := range n {
			row := md[i*n : i*n+n]
			t := tmpl[tmplIdx[i]]
			copy(row[:i+1], t[:i+1])
			copy(row[i+1:], negRow[i+1:])
		}
		return m
	}
	for i := range n {
		di := docIDs[i]
		row := md[i*n : i*n+n]
		for j := range n {
			if di == docIDs[j] && j <= i {
				row[j] = 0
			} else {
				row[j] = neg
			}
		}
	}
	return m
}

// docMaskTemplateLimit bounds how many distinct documents the templated fill will build for.
// Beyond it the template set stops being cheaper than the direct loop it replaces: building
// it is D·n work and D·n·8 bytes, so at D approaching n it costs a second n² pass plus an
// n²-sized allocation. 256 keeps the templates under 256·n·8 bytes while covering every
// realistic packed block, which holds tens of documents.
const docMaskTemplateLimit = 256

// docMaskTemplates returns, for each position, the index of its document's mask template,
// plus the templates themselves with an all-neg row appended last. It reports false when the
// document count exceeds the limit, leaving the caller's direct loop in charge.
//
// Template k has entry j equal to 0 exactly when docIDs[j] == id_k, and neg otherwise — the
// same predicate the direct loop evaluates, so a row copied from it and then neg-filled past
// the diagonal is bit-identical. The zero written here and the zero the direct loop writes
// are both +0.0.
func docMaskTemplates(docIDs []int, n int, neg float64) ([]int, [][]float64, bool) {
	slot := make(map[int]int, 16)
	tmplIdx := make([]int, n)
	for i, id := range docIDs {
		k, seen := slot[id]
		if !seen {
			if len(slot) >= docMaskTemplateLimit {
				return nil, nil, false
			}
			k = len(slot)
			slot[id] = k
		}
		tmplIdx[i] = k
	}
	d := len(slot)
	buf := make([]float64, (d+1)*n)
	tmpl := make([][]float64, d+1)
	for k := range d + 1 {
		tmpl[k] = buf[k*n : (k+1)*n : (k+1)*n]
	}
	for k := range d + 1 {
		row := tmpl[k]
		for j := range n {
			row[j] = neg
		}
	}
	for j, id := range docIDs {
		tmpl[slot[id]][j] = 0
	}
	return tmplIdx, tmpl, true
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
