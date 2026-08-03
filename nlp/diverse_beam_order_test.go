package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// TestDiverseBeamSelectionOrderIsFrozen pins the beams DiverseBeamSearch returns — their tokens,
// their order and their scores — across replacing the full candidate sort with a bounded top-k
// selection.
//
// The order is the part the other tests do not cover, and that is why this exists. The suite
// checks diversity, the reduction to plain beam search, and multi-step behavior, but a mutation
// that leaves the selected set in HEAP order rather than sorted order passes all of them: the
// final result is re-sorted by score before it is returned, so an intra-group permutation only
// shows up through which beams survive the NEXT step. This digest covers a run long enough for
// that to matter.
//
// The scorer is deliberately full of near-ties. The selection's comparator is a strict total
// order over (augmented score, parent, token), and ties are exactly where a heap and a sort can
// disagree if the comparator is not respected on both paths.
func TestDiverseBeamSelectionOrderIsFrozen(t *testing.T) {
	const wantDigest uint64 = 5606564999642185812
	const vocab, steps = 24, 6
	var score nlp.NextLogits = func(toks []int) []float64 {
		out := make([]float64, vocab)
		last := 0
		if len(toks) > 0 {
			last = toks[len(toks)-1]
		}
		for t := range out {
			// Coarse quantization so many candidates land on identical scores.
			out[t] = math.Trunc(math.Sin(float64(last*7+t))*8) / 8
		}
		return out
	}
	beams, err := nlp.DiverseBeamSearch(score, []int{0}, 6, 3, steps, -1, 0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	for _, b := range beams {
		for _, tok := range b.Tokens {
			u := uint64(tok)
			for s := 0; s < 64; s += 8 {
				h = (h ^ (u>>s)&0xff) * 1099511628211
			}
		}
		u := math.Float64bits(b.Score)
		for s := 0; s < 64; s += 8 {
			h = (h ^ (u>>s)&0xff) * 1099511628211
		}
	}
	if h != wantDigest {
		t.Fatalf("beam digest %d, want %d", h, wantDigest)
	}
}
