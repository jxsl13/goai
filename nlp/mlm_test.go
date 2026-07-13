package nlp_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §V16 tier-1 (arXiv:1810.04805 §3.1): ~15% of positions are selected for prediction (labels set).
func TestMLMSelectionRate(t *testing.T) {
	tokens := make([]int, 20000)
	for i := range tokens {
		tokens[i] = 5
	}
	rng := rand.New(rand.NewPCG(1, 2))
	_, labels := nlp.MLMMask(tokens, 0.15, 0, 10000, rng)
	sel := 0
	for _, l := range labels {
		if l != nlp.MLMIgnoreLabel {
			sel++
		}
	}
	if frac := float64(sel) / float64(len(tokens)); math.Abs(frac-0.15) > 0.01 {
		t.Errorf("selection rate %.4f, want ≈0.15", frac)
	}
}

// §V16 tier-1 (80/10/10): of the selected positions, ~80% become [MASK], ~10% a random token, ~10%
// stay unchanged. Counted with maskID=0 (never drawn as a content token) and constant token 5.
func TestMLM801010(t *testing.T) {
	tokens := make([]int, 40000)
	for i := range tokens {
		tokens[i] = 5
	}
	rng := rand.New(rand.NewPCG(3, 4))
	const maskID = 0
	input, labels := nlp.MLMMask(tokens, 0.15, maskID, 10000, rng)
	var mask, unchanged, random, sel int
	for i := range input {
		if labels[i] == nlp.MLMIgnoreLabel {
			continue
		}
		sel++
		switch {
		case input[i] == maskID:
			mask++
		case input[i] == 5: // original token
			unchanged++
		default:
			random++
		}
	}
	for _, c := range []struct {
		name string
		got  float64
		want float64
	}{{"mask", float64(mask) / float64(sel), 0.8}, {"random", float64(random) / float64(sel), 0.1}, {"unchanged", float64(unchanged) / float64(sel), 0.1}} {
		if math.Abs(c.got-c.want) > 0.02 {
			t.Errorf("%s fraction %.4f, want ≈%.2f", c.name, c.got, c.want)
		}
	}
}

// §V16 tier-1 (labels): a label is set exactly at the selected positions, holds the original token
// there, and non-selected positions keep the original input.
func TestMLMLabels(t *testing.T) {
	tokens := []int{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
	rng := rand.New(rand.NewPCG(5, 6))
	input, labels := nlp.MLMMask(tokens, 0.5, 99, 1000, rng)
	for i := range tokens {
		if labels[i] != nlp.MLMIgnoreLabel {
			if labels[i] != tokens[i] {
				t.Errorf("label[%d]=%d, want original %d", i, labels[i], tokens[i])
			}
		} else if input[i] != tokens[i] {
			t.Errorf("non-selected input[%d]=%d changed from %d", i, input[i], tokens[i])
		}
	}
}

// §V15 (round-trip): MLMReconstruct recovers the original from the corrupted input + labels.
func TestMLMRoundTrip(t *testing.T) {
	doc := make([]int, 200)
	for i := range doc {
		doc[i] = 100 + i
	}
	rng := rand.New(rand.NewPCG(7, 8))
	for trial := 0; trial < 50; trial++ {
		input, labels := nlp.MLMMask(doc, 0.15, 1, 10000, rng)
		got, ok := nlp.MLMReconstruct(input, labels)
		if !ok || !slices.Equal(got, doc) {
			t.Fatalf("round-trip: got=%v ok=%v", got, ok)
		}
	}
}

// §V2 (edges): empty and single-token documents round-trip.
func TestMLMEdges(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	for _, doc := range [][]int{{}, {7}, {1, 2, 3}} {
		in, lab := nlp.MLMMask(doc, 0.15, 0, 100, rng)
		got, ok := nlp.MLMReconstruct(in, lab)
		if !ok || !slices.Equal(got, doc) {
			t.Errorf("doc=%v: got %v ok=%v", doc, got, ok)
		}
	}
}

func TestMLMErrors(t *testing.T) {
	if _, ok := nlp.MLMReconstruct([]int{1, 2}, []int{1}); ok {
		t.Error("length mismatch should return ok=false")
	}
}

// §V15 fuzz: for any token stream, seed and mask rate, MLM masking round-trips exactly.
func FuzzMLMRoundTrip(f *testing.F) {
	f.Add([]byte("mask me maybe"), uint64(1), 15)
	f.Add([]byte{}, uint64(0), 50)
	f.Add([]byte{5, 5, 5, 5}, uint64(9), 100)
	f.Fuzz(func(t *testing.T, data []byte, seed uint64, ratePct int) {
		maskProb := math.Min(1, math.Max(0, float64(((ratePct%101)+101)%101)/100))
		tokens := make([]int, len(data)) // bytes 0..255, all non-negative
		for i, b := range data {
			tokens[i] = int(b) + 1 // +1 so 0 can serve as a distinct [MASK] id
		}
		rng := rand.New(rand.NewPCG(seed, 0x117))
		in, lab := nlp.MLMMask(tokens, maskProb, 0, 10000, rng)
		got, ok := nlp.MLMReconstruct(in, lab)
		if !ok {
			t.Fatalf("reconstruct failed for %v", tokens)
		}
		if !slices.Equal(got, tokens) {
			t.Fatalf("round-trip mismatch: %v != %v", got, tokens)
		}
	})
}

func ExampleMLMMask() {
	doc := []int{10, 11, 12, 13, 14, 15}
	rng := rand.New(rand.NewPCG(2, 3))
	input, labels := nlp.MLMMask(doc, 0.5, 0, 1000, rng)
	back, ok := nlp.MLMReconstruct(input, labels)
	fmt.Println(ok, slices.Equal(back, doc))
	// Output:
	// true true
}

// §V16 tier-1 (§R229 special-token exclusion): protected ids are never selected — their label
// stays ignore and their input is unchanged — across many maskings.
func TestMLMExcludeNeverMasksSpecial(t *testing.T) {
	// [CLS]=1, [SEP]=2, pad=0 at fixed positions; content tokens 10.. between.
	tokens := []int{1, 10, 11, 12, 2, 13, 14, 0, 0}
	special := []int{0, 1, 2}
	specialPos := map[int]bool{0: true, 4: true, 7: true, 8: true} // where 1/2/0 sit
	rng := rand.New(rand.NewPCG(1, 2))
	for trial := 0; trial < 500; trial++ {
		input, labels := nlp.MLMMaskExcluding(tokens, 0.9, 99, 1000, special, rng) // high rate to stress it
		for i := range tokens {
			if specialPos[i] {
				if labels[i] != nlp.MLMIgnoreLabel || input[i] != tokens[i] {
					t.Fatalf("trial %d: special position %d was masked (label=%d input=%d)", trial, i, labels[i], input[i])
				}
			}
		}
	}
}

// §V16 tier-1 (nil special ⇒ identical to MLMMask): the general function reduces to MLMMask exactly
// for the same seed (special positions consume no randomness).
func TestMLMExcludeReducesToMLMMask(t *testing.T) {
	tokens := make([]int, 500)
	for i := range tokens {
		tokens[i] = 100 + i%50
	}
	in1, lab1 := nlp.MLMMask(tokens, 0.15, 3, 1000, rand.New(rand.NewPCG(7, 8)))
	in2, lab2 := nlp.MLMMaskExcluding(tokens, 0.15, 3, 1000, nil, rand.New(rand.NewPCG(7, 8)))
	if !slices.Equal(in1, in2) || !slices.Equal(lab1, lab2) {
		t.Error("MLMMaskExcluding(nil) differs from MLMMask for the same seed")
	}
}

// §V15 (round-trip): MLMReconstruct recovers the original even with special exclusion.
func TestMLMExcludeRoundTrip(t *testing.T) {
	doc := []int{1, 10, 11, 12, 2, 13, 14, 15, 0}
	rng := rand.New(rand.NewPCG(3, 4))
	for trial := 0; trial < 50; trial++ {
		in, lab := nlp.MLMMaskExcluding(doc, 0.3, 99, 1000, []int{0, 1, 2}, rng)
		got, ok := nlp.MLMReconstruct(in, lab)
		if !ok || !slices.Equal(got, doc) {
			t.Fatalf("round-trip: got=%v ok=%v", got, ok)
		}
	}
}

// §V15 fuzz: masking with special exclusion round-trips and never masks a protected id.
func FuzzMLMExcludeRoundTrip(f *testing.F) {
	f.Add([]byte("protect me"), uint64(1), 15)
	f.Add([]byte{1, 2, 3, 0, 1}, uint64(9), 90)
	f.Fuzz(func(t *testing.T, data []byte, seed uint64, ratePct int) {
		maskProb := math.Min(1, math.Max(0, float64(((ratePct%101)+101)%101)/100))
		tokens := make([]int, len(data))
		for i, b := range data {
			tokens[i] = int(b) + 1
		}
		special := []int{1, 2, 3} // protect a few low ids
		specialSet := map[int]bool{1: true, 2: true, 3: true}
		rng := rand.New(rand.NewPCG(seed, 0x5e))
		in, lab := nlp.MLMMaskExcluding(tokens, maskProb, 0, 10000, special, rng)
		got, ok := nlp.MLMReconstruct(in, lab)
		if !ok || !slices.Equal(got, tokens) {
			t.Fatalf("round-trip mismatch: %v != %v (ok=%v)", got, tokens, ok)
		}
		for i, tok := range tokens {
			if specialSet[tok] && lab[i] != nlp.MLMIgnoreLabel {
				t.Fatalf("protected token %d at %d was masked", tok, i)
			}
		}
	})
}

func ExampleMLMMaskExcluding() {
	// [CLS]=1 and [SEP]=2 are protected: with a 100% rate, only the content token is masked.
	doc := []int{1, 10, 2}
	in, lab := nlp.MLMMaskExcluding(doc, 1.0, 99, 1000, []int{1, 2}, rand.New(rand.NewPCG(5, 5)))
	fmt.Println(lab[0] == nlp.MLMIgnoreLabel, lab[2] == nlp.MLMIgnoreLabel, in[1] == 99)
	// Output:
	// true true true
}
