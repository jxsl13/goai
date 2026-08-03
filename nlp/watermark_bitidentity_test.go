package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// TestWatermarkDetectIsBitIdentical freezes the detector's three outputs. Banding the token
// loop claims to change no value — each step reseeds from (Key, previous token) alone, restores
// its permutation, and contributes an integer to a count whose summation order cannot matter —
// and a z-score is exactly the kind of number a small drift would hide in.
//
// The lengths straddle the fan-out gate: 64 tokens run inline, 20000 band, and 4001 is not a
// multiple of the worker count so the last band is short.
func TestWatermarkDetectIsBitIdentical(t *testing.T) {
	cases := []struct {
		n          int
		wantZ      uint64
		wantGreen  int
		wantScored int
	}{
		{64, 13835632080035413903, 8, 63},
		{4001, 13815929971456765315, 997, 4000},
		{20000, 4609831100823785270, 5097, 19999},
	}
	w := &nlp.Watermark{Key: 0x9e3779b97f4a7c15, VocabSize: 32000, Gamma: 0.25, Delta: 2}
	for _, c := range cases {
		toks := make([]int, c.n)
		for i := range toks {
			toks[i] = (i*7919 + 13) % w.VocabSize
		}
		z, green, scored := w.Detect(toks)
		zb := math.Float64bits(z)
		if zb != c.wantZ || green != c.wantGreen || scored != c.wantScored {
			t.Fatalf("n=%d: z=%v green=%d scored=%d, want zbits=%d green=%d scored=%d",
				c.n, z, green, scored, c.wantZ, c.wantGreen, c.wantScored)
		}
	}
}
