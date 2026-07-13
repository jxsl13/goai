package nlp_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

const sentBase = 10000

func countSentinels(s []int) int {
	n := 0
	for _, v := range s {
		if v >= sentBase {
			n++
		}
	}
	return n
}

// §V16 tier-1 (arXiv:1910.10683 §3.1.4): each dropped span is one sentinel in the input; the
// target carries one sentinel per span plus a trailing final sentinel.
func TestT5CorruptStructure(t *testing.T) {
	tokens := make([]int, 100)
	for i := range tokens {
		tokens[i] = i
	}
	rng := rand.New(rand.NewPCG(1, 5))
	input, target := nlp.T5Corrupt(tokens, 0.15, 3, sentBase, rng)
	ins, tgs := countSentinels(input), countSentinels(target)
	if tgs != ins+1 {
		t.Errorf("target has %d sentinels, want input's %d + 1 (final)", tgs, ins)
	}
	// masked token count (target non-sentinels) ≈ 15% of the document.
	masked := len(target) - tgs
	if frac := float64(masked) / float64(len(tokens)); math.Abs(frac-0.15) > 0.03 {
		t.Errorf("masked fraction %.3f, want ≈0.15", frac)
	}
	// input length + masked tokens == original (input replaces each span with one sentinel).
	if len(input)-ins+masked != len(tokens) {
		t.Errorf("input(%d) − sentinels(%d) + masked(%d) != n(%d)", len(input), ins, masked, len(tokens))
	}
}

// §V16 tier-1 (unique ordered sentinels): sentinels appear as base, base+1, … in both streams.
func TestT5CorruptSentinelsOrdered(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 6))
	tokens := make([]int, 60)
	for i := range tokens {
		tokens[i] = i
	}
	input, target := nlp.T5Corrupt(tokens, 0.15, 3, sentBase, rng)
	check := func(name string, s []int) {
		want := sentBase
		for _, v := range s {
			if v >= sentBase {
				if v != want {
					t.Errorf("%s sentinel out of order: %d, want %d", name, v, want)
				}
				want++
			}
		}
	}
	check("input", input)
	check("target", target)
}

// §V16 tier-1 (reversibility): reconstruction splices the target spans back into the input.
func TestT5RoundTrip(t *testing.T) {
	doc := make([]int, 40)
	for i := range doc {
		doc[i] = 100 + i
	}
	rng := rand.New(rand.NewPCG(3, 7))
	for i := 0; i < 30; i++ {
		input, target := nlp.T5Corrupt(doc, 0.15, 3, sentBase, rng)
		got, ok := nlp.T5Reconstruct(input, target, sentBase)
		if !ok || !slices.Equal(got, doc) {
			t.Fatalf("round-trip: input=%v target=%v ⇒ %v (ok=%v)", input, target, got, ok)
		}
	}
}

// §V2 (edges): empty, single-token, and fully-masked documents round-trip.
func TestT5Edges(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 8))
	for _, doc := range [][]int{{}, {7}, {1, 2, 3}} {
		in, tg := nlp.T5Corrupt(doc, 1.0, 3, sentBase, rng) // density 1 ⇒ everything masked
		got, ok := nlp.T5Reconstruct(in, tg, sentBase)
		if !ok || !slices.Equal(got, doc) {
			t.Errorf("doc=%v: got %v ok=%v", doc, got, ok)
		}
	}
}

// §V2: a malformed pair is rejected.
func TestT5RejectsMalformed(t *testing.T) {
	if _, ok := nlp.T5Reconstruct([]int{1, sentBase, 2}, []int{9, 9}, sentBase); ok {
		t.Error("input sentinel with no target span should not reconstruct")
	}
}

// §V15 fuzz: for any token stream, seed, density and mean span, corruption round-trips.
func FuzzT5RoundTrip(f *testing.F) {
	f.Add([]byte("hello world"), uint64(1), 15, 3)
	f.Add([]byte{}, uint64(0), 50, 2)
	f.Fuzz(func(t *testing.T, data []byte, seed uint64, dPct, span int) {
		density := math.Min(1, math.Max(0.01, float64((dPct%100+1))/100))
		meanSpan := math.Max(1, float64(span%8+1))
		tokens := make([]int, len(data)) // bytes 0..255, below sentBase
		for i, b := range data {
			tokens[i] = int(b)
		}
		rng := rand.New(rand.NewPCG(seed, 0x7175))
		in, tg := nlp.T5Corrupt(tokens, density, meanSpan, sentBase, rng)
		got, ok := nlp.T5Reconstruct(in, tg, sentBase)
		if !ok {
			t.Fatalf("reconstruct failed for %v", tokens)
		}
		if !slices.Equal(got, tokens) {
			t.Fatalf("round-trip mismatch: %v != %v", got, tokens)
		}
	})
}

func ExampleT5Corrupt() {
	// Corrupt a short document; reconstruction recovers it.
	doc := []int{10, 11, 12, 13, 14, 15}
	rng := rand.New(rand.NewPCG(5, 5))
	in, tg := nlp.T5Corrupt(doc, 0.34, 2, sentBase, rng)
	back, _ := nlp.T5Reconstruct(in, tg, sentBase)
	fmt.Println(slices.Equal(back, doc))
	// Output:
	// true
}
