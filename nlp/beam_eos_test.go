package nlp

import "testing"

// §B71 regression. BeamSearch used to test EVERY candidate for completion, not
// just the top-`width` frontier. Because the expansion covers the whole
// vocabulary, one eos-terminated candidate exists for every live beam at every
// step, so `done` reached `width` on the first step and the function returned a
// 2-token EOS hypothesis — for any model with a real eos id, i.e. in every
// realistic use. It looked healthy: the beams came back sorted, length-penalized
// and plausibly shaped, with no error.
//
// These tests pin BOTH directions. Pinning only "an improbable EOS is ignored"
// would pass if EOS were disabled outright, which is why the honoured case below
// carries equal weight.

// eosImprobable makes token 1 overwhelmingly likely and eos (3) astronomically
// unlikely: p(eos) ≈ e^-110.
func eosImprobable(prefix []int) []float64 { return []float64{0, 10, 0, -100} }

// eosLikely inverts it — eos is the argmax, so a correct search must stop.
func eosLikely(prefix []int) []float64 { return []float64{0, 0, 0, 10} }

func TestBeamSearchIgnoresImprobableEOS(t *testing.T) {
	const maxNew = 10
	for _, width := range []int{1, 3} {
		got := BeamSearch(eosImprobable, []int{0}, width, maxNew, 3, 0)
		if len(got) == 0 {
			t.Fatalf("width=%d: no beams", width)
		}
		best := got[0].Tokens
		if len(best) != maxNew+1 {
			t.Errorf("width=%d: best has %d tokens, want %d — an EOS at p≈e^-110 must not "+
				"outrank a token at p≈1 (got %v)", width, len(best), maxNew+1, best)
		}
		for i, tok := range best[1:] {
			if tok != 1 {
				t.Errorf("width=%d: best[%d]=%d, want the argmax token 1 (%v)", width, i+1, tok, best)
				break
			}
		}
	}
}

// The other direction: a likely EOS must still stop the search promptly. Without
// this, a "fix" that simply stopped completing on EOS would pass the test above.
func TestBeamSearchHonoursLikelyEOS(t *testing.T) {
	const maxNew = 10
	for _, width := range []int{1, 3} {
		got := BeamSearch(eosLikely, []int{0}, width, maxNew, 3, 0)
		if len(got) == 0 {
			t.Fatalf("width=%d: no beams", width)
		}
		best := got[0].Tokens
		if len(best) != 2 || best[len(best)-1] != 3 {
			t.Errorf("width=%d: best=%v, want the 2-token [start eos] — when eos IS the "+
				"argmax the search must complete immediately", width, best)
		}
	}
}

// Disabling eos (eos<0) must run to maxNew regardless of what the logits say.
func TestBeamSearchEOSDisabled(t *testing.T) {
	const maxNew = 10
	got := BeamSearch(eosLikely, []int{0}, 2, maxNew, -1, 0)
	if len(got) == 0 {
		t.Fatal("no beams")
	}
	if len(got[0].Tokens) != maxNew+1 {
		t.Errorf("best has %d tokens, want %d — eos<0 disables completion entirely (%v)",
			len(got[0].Tokens), maxNew+1, got[0].Tokens)
	}
}
