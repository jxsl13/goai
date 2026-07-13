package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

func attendsPack(m *tensor.Tensor, i, j int) bool { return m.AtF64(i, j) == 0 }
func blockedPack(m *tensor.Tensor, i, j int) bool { return math.IsInf(m.AtF64(i, j), -1) }

// §V2 (packing): blocks respect the capacity, preserve all tokens, and pack tightly
// (first-fit-decreasing).
func TestPackSequences(t *testing.T) {
	seqs := [][]int{{1, 1, 1}, {2, 2}, {3, 3, 3, 3}, {4}} // lens 3,2,4,1; maxLen 5
	blocks, docIDs := nlp.PackSequences(seqs, 5)
	total := 0
	for b := range blocks {
		if len(blocks[b]) > 5 {
			t.Errorf("block %d length %d > maxLen 5", b, len(blocks[b]))
		}
		if len(blocks[b]) != len(docIDs[b]) {
			t.Errorf("block %d: %d tokens but %d docIDs", b, len(blocks[b]), len(docIDs[b]))
		}
		total += len(blocks[b])
	}
	if total != 3+2+4+1 {
		t.Errorf("packed %d tokens, want %d (all preserved)", total, 3+2+4+1)
	}
	// FFD packs 4+1 and 3+2 into two full blocks of 5.
	if len(blocks) != 2 {
		t.Errorf("got %d blocks, want 2 (tight FFD packing)", len(blocks))
	}
}

// §V16 tier-1 (arXiv:2107.02027, block-diagonal causal mask): a single document reduces to
// the ordinary causal mask; co-packed documents never attend across the boundary.
func TestDocumentCausalMask(t *testing.T) {
	// single document: plain causal.
	single := nlp.DocumentCausalMask([]int{0, 0, 0})
	for i := range 3 {
		for j := range 3 {
			if j <= i && !attendsPack(single, i, j) {
				t.Errorf("single-doc: %d should attend %d", i, j)
			}
			if j > i && !blockedPack(single, i, j) {
				t.Errorf("single-doc: %d should not attend future %d", i, j)
			}
		}
	}
	// two documents [0,0,1,1]: doc 1 (positions 2,3) must not attend doc 0 (0,1), and vice versa.
	two := nlp.DocumentCausalMask([]int{0, 0, 1, 1})
	if !blockedPack(two, 2, 1) || !blockedPack(two, 3, 0) {
		t.Error("cross-document attention not blocked")
	}
	if !attendsPack(two, 3, 2) || !attendsPack(two, 1, 0) { // within-doc causal still allowed
		t.Error("within-document causal attention wrongly blocked")
	}
	if !blockedPack(two, 2, 3) { // within doc 1, still causal (2 cannot see future 3)
		t.Error("within-document future attention not blocked")
	}
}

// §V16 tier-1 (position reset): positions restart at 0 per document.
func TestDocumentPositions(t *testing.T) {
	got := nlp.DocumentPositions([]int{0, 0, 0, 1, 1})
	want := []int{0, 1, 2, 0, 1}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("positions=%v want %v", got, want)
		}
	}
}

// §V2 (truncation): a document longer than maxLen is truncated to maxLen.
func TestPackTruncatesOverLong(t *testing.T) {
	blocks, _ := nlp.PackSequences([][]int{{1, 2, 3, 4, 5, 6, 7}}, 4)
	if len(blocks) != 1 || len(blocks[0]) != 4 {
		t.Errorf("over-long seq: got blocks %v, want one block of 4", blocks)
	}
}

// §V2 (end-to-end): the mask and positions built from a pack's docIDs are mutually
// consistent — position 0 exactly at each masked document boundary.
func TestPackingIntegration(t *testing.T) {
	_, docIDs := nlp.PackSequences([][]int{{1, 1}, {2, 2, 2}}, 5)
	d := docIDs[0]
	pos := nlp.DocumentPositions(d)
	mask := nlp.DocumentCausalMask(d)
	for i := 1; i < len(d); i++ {
		boundary := d[i] != d[i-1]
		if boundary != (pos[i] == 0) {
			t.Errorf("pos reset at %d = %v but boundary = %v", i, pos[i] == 0, boundary)
		}
		if boundary && !blockedPack(mask, i, i-1) {
			t.Errorf("boundary at %d should mask attention to previous document", i)
		}
	}
}

func ExampleDocumentPositions() {
	// Two documents (lengths 3 and 2) packed together: the mask keeps them separate and the
	// position ids restart per document.
	_, docIDs := nlp.PackSequences([][]int{{10, 11, 12}, {20, 21}}, 5)
	fmt.Println(docIDs[0], nlp.DocumentPositions(docIDs[0]))
	// Output:
	// [0 0 0 1 1] [0 1 2 0 1]
}
