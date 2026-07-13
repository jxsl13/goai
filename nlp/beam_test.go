package nlp_test

import (
	"math"
	"sort"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// markov builds a NextLogits whose next-token logits depend only on the last
// token, via the row M[last] — a tiny model we can enumerate exhaustively.
func markov(M [][]float64) nlp.NextLogits {
	return func(prefix []int) []float64 { return M[prefix[len(prefix)-1]] }
}

func logSoftmax(logits []float64) []float64 {
	m := math.Inf(-1)
	for _, v := range logits {
		if v > m {
			m = v
		}
	}
	var s float64
	for _, v := range logits {
		s += math.Exp(v - m)
	}
	lse := m + math.Log(s)
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = v - lse
	}
	return out
}

// §V16 exact case: beam width 1 is greedy decoding — always the argmax token.
func TestBeamOneEqualsGreedy(t *testing.T) {
	M := [][]float64{{0.0, 1.0, 0.5}, {2.0, 0.0, 1.0}, {0.5, 1.5, 0.0}}
	next := markov(M)
	start := []int{0}

	beams := nlp.BeamSearch(next, start, 1, 4, -1, 0)
	if len(beams) != 1 {
		t.Fatalf("beam-1 returned %d hyps, want 1", len(beams))
	}

	// manual greedy
	greedy := []int{0}
	for range 4 {
		ls := logSoftmax(next(greedy))
		best := 0
		for i := 1; i < len(ls); i++ {
			if ls[i] > ls[best] {
				best = i
			}
		}
		greedy = append(greedy, best)
	}
	if len(beams[0].Tokens) != len(greedy) {
		t.Fatalf("length %d vs greedy %d", len(beams[0].Tokens), len(greedy))
	}
	for i := range greedy {
		if beams[0].Tokens[i] != greedy[i] {
			t.Errorf("token %d: beam %d vs greedy %d", i, beams[0].Tokens[i], greedy[i])
		}
	}
}

// §V16 exact case: with an unbounded beam (never pruning) beam search returns the
// exact top sequences by total log-probability — verified against exhaustive
// enumeration of all Vᴸ sequences on the tiny Markov model.
func TestBeamUnboundedEqualsExhaustive(t *testing.T) {
	M := [][]float64{{0.0, 1.0, 0.5}, {2.0, 0.0, 1.0}, {0.5, 1.5, 0.0}}
	next := markov(M)
	start := []int{0}
	const V, L = 3, 3

	// exhaustive: score every length-L continuation by Σ log-softmax
	type seq struct {
		toks  []int
		score float64
	}
	var all []seq
	var rec func(prefix []int, score float64, depth int)
	rec = func(prefix []int, score float64, depth int) {
		if depth == L {
			all = append(all, seq{append([]int(nil), prefix...), score})
			return
		}
		ls := logSoftmax(next(prefix))
		for tok := range V {
			rec(append(prefix, tok), score+ls[tok], depth+1)
		}
	}
	rec(start, 0, 0)

	// exhaustive scores, sorted, for the monotonicity check
	wantScores := make([]float64, len(all))
	byKey := map[string]float64{}
	key := func(toks []int) string {
		b := make([]byte, len(toks))
		for i, v := range toks {
			b[i] = byte('0' + v)
		}
		return string(b)
	}
	for i, s := range all {
		wantScores[i] = s.score
		byKey[key(s.toks)] = s.score
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(wantScores)))

	beams := nlp.BeamSearch(next, start, V*V*V, L, -1, 0) // width ≥ Vᴸ ⇒ no pruning
	if len(beams) != len(all) {
		t.Fatalf("beam returned %d, exhaustive %d", len(beams), len(all))
	}
	// (a) every beam sequence exists with the exact same total log-prob (set match,
	// tie-order agnostic); (b) beam scores are sorted descending and equal the
	// exhaustive score at each rank.
	for i, b := range beams {
		want, ok := byKey[key(b.Tokens)]
		if !ok {
			t.Errorf("beam produced unknown sequence %v", b.Tokens)
			continue
		}
		if math.Abs(b.Score-want) > 1e-12 {
			t.Errorf("sequence %v: beam score %.12g vs exhaustive %.12g", b.Tokens, b.Score, want)
		}
		if math.Abs(b.Score-wantScores[i]) > 1e-12 {
			t.Errorf("rank %d: beam score %.12g vs exhaustive-sorted %.12g", i, b.Score, wantScores[i])
		}
		if i > 0 && beams[i-1].Score < b.Score-1e-15 {
			t.Errorf("beam not sorted descending at rank %d", i)
		}
	}
}

// Beam search finds a higher-total-log-prob sequence than greedy when the locally
// best first token leads to a poor continuation (the classic beam > greedy case).
func TestBeamBeatsGreedy(t *testing.T) {
	// after 0: token 0 slightly preferred; after 0 again: flat; after 1: token 0
	// almost certain → the [0,1,0] path beats greedy's [0,0,·].
	M := [][]float64{{2.0, 1.9}, {5.0, 0.0}}
	next := markov(M)
	start := []int{0}

	greedy := nlp.BeamSearch(next, start, 1, 2, -1, 0)
	beam2 := nlp.BeamSearch(next, start, 2, 2, -1, 0)
	if beam2[0].Score <= greedy[0].Score {
		t.Fatalf("beam-2 %.4f must beat greedy %.4f", beam2[0].Score, greedy[0].Score)
	}
	want := []int{0, 1, 0}
	for i, tok := range want {
		if beam2[0].Tokens[i] != tok {
			t.Errorf("beam-2 best token %d: %d, want %d", i, beam2[0].Tokens[i], tok)
		}
	}
}

// A hypothesis that emits EOS is completed before maxNew.
func TestBeamEOSCompletes(t *testing.T) {
	// after 0 → token 1; after 1 → EOS(2) is argmax. eos=2.
	M := [][]float64{{0.0, 3.0, 0.0}, {0.0, 0.0, 5.0}, {0.0, 0.0, 0.0}}
	next := markov(M)
	beams := nlp.BeamSearch(next, []int{0}, 1, 10, 2, 0)
	if len(beams) == 0 {
		t.Fatal("no completed hypothesis")
	}
	toks := beams[0].Tokens
	if toks[len(toks)-1] != 2 {
		t.Errorf("best hyp must end in eos=2, got %v", toks)
	}
	if len(toks) >= 10 {
		t.Errorf("hyp should complete on eos well before maxNew, len=%d", len(toks))
	}
}

// The length penalty divides the raw log-prob sum by ((5+n)/6)^alpha exactly.
func TestBeamLengthPenaltyFormula(t *testing.T) {
	M := [][]float64{{0.0, 1.0, 0.5}, {2.0, 0.0, 1.0}, {0.5, 1.5, 0.0}}
	next := markov(M)
	raw := nlp.BeamSearch(next, []int{0}, 1, 3, -1, 0.0)
	pen := nlp.BeamSearch(next, []int{0}, 1, 3, -1, 0.7)
	// same sequence (greedy), score scaled by 1/((5+3)/6)^0.7
	lp := math.Pow((5.0+3.0)/6.0, 0.7)
	if math.Abs(pen[0].Score-raw[0].Score/lp) > 1e-12 {
		t.Errorf("penalized %.12g != raw/lp %.12g", pen[0].Score, raw[0].Score/lp)
	}
}
