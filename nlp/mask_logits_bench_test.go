package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// MaskLogits had no benchmark, which is why its cost was unranked: the guide benchmarks next door
// measure ε-closure and interning WARMUP and never call it (that file says so at its head).
//
// What this measures is the steady state of constrained generation: the memo is warm, so every token
// is a table hit, and the whole cost is the per-token call plus its re-validation. That is the regime
// a real decode spends nearly all its time in — the first visit to a (state, token) pair pays the
// walk once, every later one does not.
//
// Vocabulary size is the variable that matters, since the loop is O(vocab) per generated token, and a
// 128-entry fixture would understate a real 32k-128k vocabulary by two orders of magnitude.
func maskVocab(n int) []string {
	base := []string{
		"{", "}", "[", "]", ":", ",", "\"", " ", "\n", "true", "false", "null",
		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", ".", "e",
		"a", "b", "c", "d", "ab", "abc", "abcd", "aa", "bb", "name", "value",
	}
	out := make([]string, 0, n)
	for len(out) < n {
		for _, s := range base {
			if len(out) >= n {
				break
			}
			out = append(out, s)
		}
	}
	return out
}

func BenchmarkRegexGuideMaskLogits(b *testing.B) {
	for _, v := range []int{4096, 32768} {
		vocab := maskVocab(v)
		g, err := nlp.NewRegexGuide(`\{("[a-z]+"\s*:\s*[0-9]+\s*,?\s*)*\}`, vocab)
		if err != nil {
			b.Fatal(err)
		}
		st := g.Start()
		logits := make([]float64, v)
		// Warm the memo for this state so the loop measures table hits, not first-visit walks.
		g.MaskLogits(st, logits, -1)
		b.Run(maskName(v), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				for i := range logits {
					logits[i] = 0
				}
				g.MaskLogits(st, logits, 1)
			}
			// Guard against a vacuous fixture in BOTH directions: if the state allowed everything
			// the loop would never take its masking branch, and if it allowed nothing the regex
			// would be degenerate. Token 0 is "{", which this start state legitimately allows, so
			// counting is the right check rather than probing one index.
			masked := 0
			for _, v := range logits {
				if math.IsInf(v, -1) {
					masked++
				}
			}
			if masked == 0 || masked == len(logits) {
				b.Fatalf("fixture masks %d of %d tokens; it must mask some but not all", masked, len(logits))
			}
		})
	}
}

func maskName(n int) string {
	if n == 0 {
		return "v0"
	}
	var d [8]byte
	i := len(d)
	for n > 0 {
		i--
		d[i] = byte('0' + n%10)
		n /= 10
	}
	return "v" + string(d[i:])
}

// The GrammarGuide twin of the benchmark above. Same regime — memo warm, every token a table hit —
// and the same reason it needs its own numbers rather than inheriting the regex ones: the two
// MaskLogits bodies are structurally identical but sit on different Advance implementations, so what
// share of the loop is the hoistable per-call overhead is a question only measurement answers.
func BenchmarkGrammarGuideMaskLogits(b *testing.B) {
	const gr = `root ::= obj
obj ::= "{" pairs "}"
pairs ::= pair | pair "," pairs | ""
pair ::= str ":" val
str ::= "\"" [a-z]+ "\""
val ::= str | num | "true" | "false" | "null"
num ::= [0-9]+`
	for _, v := range []int{4096, 32768} {
		vocab := maskVocab(v)
		g, err := nlp.NewGrammarGuide(gr, vocab)
		if err != nil {
			b.Fatal(err)
		}
		st := g.Start()
		logits := make([]float64, v)
		g.MaskLogits(st, logits, -1) // warm the memo for this state
		b.Run(maskName(v), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				for i := range logits {
					logits[i] = 0
				}
				g.MaskLogits(st, logits, 1)
			}
			masked := 0
			for _, x := range logits {
				if math.IsInf(x, -1) {
					masked++
				}
			}
			if masked == 0 || masked == len(logits) {
				b.Fatalf("fixture masks %d of %d tokens; it must mask some but not all", masked, len(logits))
			}
		})
	}
}
