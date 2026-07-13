package nlp_test

import (
	"fmt"
	"math"
	"math/rand"
	"regexp"
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §V16 tier-1: from the start state of `[0-9]+`, digit tokens are allowed and non-digits masked;
// the start state is not accepting (needs ≥1 digit) but becomes accepting after one digit, and a
// multi-character token is allowed only if every character has a valid transition.
func TestRegexGuideMasking(t *testing.T) {
	vocab := []string{"0", "5", "a", "x", "12", "1a"}
	g, err := nlp.NewRegexGuide(`[0-9]+`, vocab)
	if err != nil {
		t.Fatal(err)
	}
	st := g.Start()
	if g.Accepting(st) {
		t.Error("start of [0-9]+ must not accept the empty string")
	}
	want := map[string]bool{"0": true, "5": true, "a": false, "x": false, "12": true, "1a": false}
	for tok, allow := range want {
		id := indexOf(vocab, tok)
		if g.Allowed(st, id) != allow {
			t.Errorf("Allowed(start, %q) = %v, want %v", tok, g.Allowed(st, id), allow)
		}
	}
	st2 := g.Advance(st, indexOf(vocab, "5"))
	if !g.Accepting(st2) {
		t.Error("after one digit, [0-9]+ must accept (a complete match)")
	}
	if !g.Allowed(st2, indexOf(vocab, "0")) {
		t.Error("more digits must stay allowed after the first")
	}
}

// §V16 tier-1: MaskLogits sets disallowed tokens to −Inf and permits EOS exactly at accepting
// states — the termination rule (generation may stop only in a final state).
func TestRegexGuideMaskLogitsAndTermination(t *testing.T) {
	vocab := []string{"a", "b", "c"}
	const eos = 3
	g, _ := nlp.NewRegexGuide(`ab`, vocab)
	st := g.Start()
	logits := []float64{0, 0, 0, 0} // a,b,c,eos
	eosOK := g.MaskLogits(st, logits, eos)
	if eosOK {
		t.Error("EOS must be disallowed before a complete match")
	}
	if !math.IsInf(logits[eos], -1) {
		t.Error("EOS logit must be masked when not accepting")
	}
	if logits[0] != 0 || !math.IsInf(logits[1], -1) || !math.IsInf(logits[2], -1) {
		t.Errorf("from start of `ab` only 'a' is allowed, got %v", logits)
	}
	st = g.Advance(st, 0) // 'a'
	st = g.Advance(st, 1) // 'b' → full match
	logits = []float64{0, 0, 0, 0}
	if eosOK := g.MaskLogits(st, logits, eos); !eosOK {
		t.Error("EOS must be allowed at the accepting state")
	}
	if math.IsInf(logits[eos], -1) {
		t.Error("EOS logit must survive at an accepting state")
	}
	// nothing more can be appended to a complete `ab` match
	if !math.IsInf(logits[0], -1) || !math.IsInf(logits[1], -1) || !math.IsInf(logits[2], -1) {
		t.Errorf("no token should extend a complete `ab` match, got %v", logits)
	}
}

// §V16 tier-1 (the paper's guarantee, Willard & Louf 2023): every string produced under the guide is
// in the regular language. For a battery of patterns and random vocabularies, run many guided
// generations (randomly picking among allowed tokens, stopping only at accepting states) and assert
// that every output ending in an accepting state fully matches the regex.
func TestRegexGuideAlwaysMatchesFuzz(t *testing.T) {
	patterns := []string{
		`[0-9]+`,
		`[0-9]+\.[0-9]+`,
		`(yes|no|maybe)`,
		`a(b|c)*d`,
		`[a-c]{2,4}`,
		`(ab)+`,
		`x*`, // accepts empty
		`-?[0-9]+`,
	}
	vocabs := [][]string{
		{"0", "1", "2", "5", "9", "12", "007", ".", "-", "a", "b", "c", "d", "x", "y", "yes", "no", "maybe", "ab", "cd"},
	}
	rng := rand.New(rand.NewSource(0x6c1))
	for _, pat := range patterns {
		full := regexp.MustCompile(`^(?:` + pat + `)$`)
		g, err := nlp.NewRegexGuide(pat, vocabs[0])
		if err != nil {
			t.Fatalf("compile %q: %v", pat, err)
		}
		for trial := 0; trial < 300; trial++ {
			st := g.Start()
			var sb strings.Builder
			for step := 0; step < 24; step++ {
				var allowed []int
				for id := range vocabs[0] {
					if g.Allowed(st, id) {
						allowed = append(allowed, id)
					}
				}
				// stop opportunistically at an accepting state, or when nothing can extend
				if g.Accepting(st) && (len(allowed) == 0 || rng.Intn(3) == 0) {
					break
				}
				if len(allowed) == 0 {
					break // stuck in a non-accepting dead-end for this vocab
				}
				id := allowed[rng.Intn(len(allowed))]
				sb.WriteString(vocabs[0][id])
				st = g.Advance(st, id)
			}
			if g.Accepting(st) { // only assert on completed matches
				out := sb.String()
				if !full.MatchString(out) {
					t.Fatalf("pattern %q produced %q which does not match", pat, out)
				}
			}
		}
	}
}

// §V16 tier-1: interned states are stable — advancing along equivalent character paths reaches the
// same state id, so the memoized index is well-defined.
func TestRegexGuideStateInterning(t *testing.T) {
	vocab := []string{"a", "ab", "b"}
	g, _ := nlp.NewRegexGuide(`abab`, vocab)
	viaAB := g.Advance(g.Start(), 1)              // "ab"
	viaA := g.Advance(g.Advance(g.Start(), 0), 2) // "a" then "b"
	if viaAB != viaA {
		t.Errorf("equivalent paths must intern to the same state: %d vs %d", viaAB, viaA)
	}
}

// A RegexGuide constrains generation to a regular language: from the start of `[0-9]+`, digit tokens
// are allowed and others masked, and the state becomes accepting (generation may stop) once at least
// one digit has been emitted.
func ExampleRegexGuide() {
	vocab := []string{"3", "x", "."}
	g, _ := nlp.NewRegexGuide(`[0-9]+`, vocab)
	st := g.Start()
	fmt.Println(g.Allowed(st, 0), g.Allowed(st, 1), g.Accepting(st)) // "3" ok, "x" no, not yet accepting
	st = g.Advance(st, 0)                                            // consume "3"
	fmt.Println(g.Accepting(st))                                     // a complete match now
	// Output:
	// true false false
	// true
}

func indexOf(v []string, s string) int {
	for i, x := range v {
		if x == s {
			return i
		}
	}
	return -1
}
