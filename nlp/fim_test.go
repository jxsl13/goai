package nlp_test

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

var fimSent = nlp.FIMSentinels{Pre: 1000, Suf: 1001, Mid: 1002}

func countTok(s []int, tok int) int {
	n := 0
	for _, v := range s {
		if v == tok {
			n++
		}
	}
	return n
}

// §V16 tier-1 (arXiv:2207.14255, PSM layout): <PRE> prefix <SUF> suffix <MID> middle — the
// sequence starts with <PRE>, has exactly one of each sentinel, and <SUF> precedes <MID>.
func TestFIMTransformPSM(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2207))
	fim := nlp.FIMTransform([]int{10, 11, 12, 13, 14}, fimSent, false, rng)
	if fim[0] != fimSent.Pre {
		t.Errorf("PSM does not start with <PRE>: %v", fim)
	}
	if countTok(fim, fimSent.Pre) != 1 || countTok(fim, fimSent.Suf) != 1 || countTok(fim, fimSent.Mid) != 1 {
		t.Errorf("PSM sentinel counts wrong: %v", fim)
	}
	if slices.Index(fim, fimSent.Suf) >= slices.Index(fim, fimSent.Mid) {
		t.Errorf("PSM <SUF> must precede <MID>: %v", fim)
	}
}

// §V16 tier-1 (SPM layout): <PRE> <SUF> suffix <MID> prefix middle — the <PRE> and <SUF>
// sentinels are adjacent at the front.
func TestFIMTransformSPM(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2208))
	fim := nlp.FIMTransform([]int{10, 11, 12, 13, 14}, fimSent, true, rng)
	if fim[0] != fimSent.Pre || fim[1] != fimSent.Suf {
		t.Errorf("SPM must start with <PRE><SUF> adjacent: %v", fim)
	}
}

// §V16 tier-1 (lossless reversibility): FIMReconstruct inverts FIMTransform for both modes.
func TestFIMRoundTrip(t *testing.T) {
	doc := []int{5, 6, 7, 8, 9, 10}
	for _, spm := range []bool{false, true} {
		rng := rand.New(rand.NewPCG(3, 42))
		for i := 0; i < 20; i++ { // exercise many random splits
			fim := nlp.FIMTransform(doc, fimSent, spm, rng)
			got, ok := nlp.FIMReconstruct(fim, fimSent, spm)
			if !ok || !slices.Equal(got, doc) {
				t.Fatalf("spm=%v round-trip: fim=%v ⇒ %v (ok=%v), want %v", spm, fim, got, ok, doc)
			}
		}
	}
}

// §V2 (edges): empty and single-token documents round-trip.
func TestFIMEdges(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 9))
	for _, doc := range [][]int{{}, {7}} {
		for _, spm := range []bool{false, true} {
			got, ok := nlp.FIMReconstruct(nlp.FIMTransform(doc, fimSent, spm, rng), fimSent, spm)
			if !ok || !slices.Equal(got, doc) {
				t.Errorf("spm=%v doc=%v: got %v ok=%v", spm, doc, got, ok)
			}
		}
	}
}

// §V2: a malformed (non-FIM) sequence is rejected.
func TestFIMReconstructRejectsMalformed(t *testing.T) {
	if _, ok := nlp.FIMReconstruct([]int{1, 2, 3}, fimSent, false); ok {
		t.Error("sequence without sentinels should not reconstruct")
	}
	if _, ok := nlp.FIMReconstruct([]int{fimSent.Pre, 5, 6}, fimSent, false); ok {
		t.Error("PSM without <SUF>/<MID> should not reconstruct")
	}
}

// §V15 fuzz: for any token stream, seed and mode, the FIM reorder round-trips losslessly.
func FuzzFIMRoundTrip(f *testing.F) {
	f.Add([]byte("hello"), uint64(1), false)
	f.Add([]byte{}, uint64(0), true)
	f.Add([]byte{7, 200, 3}, uint64(99), true)
	f.Fuzz(func(t *testing.T, data []byte, seed uint64, spm bool) {
		tokens := make([]int, len(data)) // bytes 0..255, disjoint from the 1000+ sentinels
		for i, b := range data {
			tokens[i] = int(b)
		}
		rng := rand.New(rand.NewPCG(seed, 0xF1A))
		got, ok := nlp.FIMReconstruct(nlp.FIMTransform(tokens, fimSent, spm, rng), fimSent, spm)
		if !ok {
			t.Fatalf("reconstruct failed for %v (spm=%v)", tokens, spm)
		}
		if !slices.Equal(got, tokens) {
			t.Fatalf("round-trip mismatch (spm=%v): %v != %v", spm, got, tokens)
		}
	})
}

func ExampleFIMTransform() {
	// Reorder a document into PSM fill-in-the-middle layout, then recover the original.
	s := nlp.FIMSentinels{Pre: 1000, Suf: 1001, Mid: 1002}
	doc := []int{1, 2, 3, 4}
	rng := rand.New(rand.NewPCG(5, 5))
	fim := nlp.FIMTransform(doc, s, false, rng)
	back, _ := nlp.FIMReconstruct(fim, s, false)
	fmt.Println(slices.Equal(back, doc))
	// Output:
	// true
}
