package nlp_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// Warmup benchmarks for the constrained-decoding guides, which had none.
//
// THE COST BEING MEASURED IS DELIBERATELY THE WARMUP, NOT THE DECODE LOOP, and the distinction
// is load-bearing. Both guides memoize per (state, token) in a tokNext table, so the expensive
// part — the ε-closure for RegexGuide, the stack expansion and interning for GrammarGuide — runs
// only on the FIRST visit to each pair. A benchmark that decoded many tokens through a small
// automaton would measure table lookups and report nothing about that work.
//
// The warmup is not a synthetic concern: it is exactly what the first request against a new
// pattern or schema pays, and in serving that is per-schema, not once per process. So these
// explore the reachable state graph from scratch on every iteration, which is the shape of that
// cost, and the guide is rebuilt each time so no memoization survives between iterations.
//
// Note what this means for triaging PS6009 here: the sorts inside these paths run once per
// (state, token) pair rather than once per generated token, so they are bounded by the automaton
// size. That is a smaller claim than "hot per token" and these benchmarks are the honest way to
// size it.

func guideVocab() []string {
	// Short, JSON-flavored pieces: enough distinct first characters to fan the automaton out,
	// and multi-rune tokens so the per-rune consume loop actually iterates.
	base := []string{
		"{", "}", "[", "]", ":", ",", "\"", " ", "\n", "true", "false", "null",
		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", ".", "e",
		"a", "b", "c", "d", "ab", "abc", "abcd", "aa", "bb", "name", "value",
	}
	out := make([]string, 0, 128)
	for len(out) < 128 {
		for _, s := range base {
			if len(out) >= 128 {
				break
			}
			out = append(out, s)
		}
	}
	return out
}

// exploreRegex walks the reachable state graph breadth-first, forcing the ε-closure and the
// resting-set sort for every (state, token) pair it reaches, up to a state cap so the benchmark
// stays bounded regardless of pattern.
func exploreRegex(b *testing.B, g *nlp.RegexGuide, vocabSize, maxStates int) {
	b.Helper()
	seen := map[int]bool{g.Start(): true}
	queue := []int{g.Start()}
	for len(queue) > 0 && len(seen) < maxStates {
		st := queue[0]
		queue = queue[1:]
		for tok := 0; tok < vocabSize; tok++ {
			next := g.Advance(st, tok)
			if next >= 0 && !seen[next] {
				seen[next] = true
				queue = append(queue, next)
			}
		}
	}
}

func exploreGrammar(b *testing.B, g *nlp.GrammarGuide, vocabSize, maxStates int) {
	b.Helper()
	seen := map[int]bool{g.Start(): true}
	queue := []int{g.Start()}
	for len(queue) > 0 && len(seen) < maxStates {
		st := queue[0]
		queue = queue[1:]
		for tok := 0; tok < vocabSize; tok++ {
			next := g.Advance(st, tok)
			if next >= 0 && !seen[next] {
				seen[next] = true
				queue = append(queue, next)
			}
		}
	}
}

// BenchmarkRegexGuideWarmup exercises RegexGuide.closure — the ε-closure whose resting set is
// sorted before interning — once per (state, token) pair over a freshly built guide.
func BenchmarkRegexGuideWarmup(b *testing.B) {
	vocab := guideVocab()
	patterns := []string{
		`[a-z]+`,
		`"[a-z]{1,8}"\s*:\s*("[a-z]*"|[0-9]+|true|false|null)`,
		`\{("[a-z]+"\s*:\s*[0-9]+\s*,?\s*)*\}`,
	}
	for i, p := range patterns {
		b.Run(fmt.Sprint(i), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				g, err := nlp.NewRegexGuide(p, vocab)
				if err != nil {
					b.Fatal(err)
				}
				exploreRegex(b, g, len(vocab), 64)
			}
		})
	}
}

// BenchmarkGrammarGuideWarmup exercises GrammarGuide.internStacks — where the stack set is
// key-sorted before interning — once per (state, token) pair over a freshly built guide.
func BenchmarkGrammarGuideWarmup(b *testing.B) {
	vocab := guideVocab()
	grammars := []string{
		`root ::= "(" root ")" | ""`,
		`root ::= obj
obj ::= "{" pairs "}"
pairs ::= pair | pair "," pairs | ""
pair ::= str ":" val
str ::= "\"" [a-z]+ "\""
val ::= str | num | "true" | "false" | "null"
num ::= [0-9]+`,
	}
	for i, gr := range grammars {
		b.Run(fmt.Sprint(i), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				g, err := nlp.NewGrammarGuide(gr, vocab)
				if err != nil {
					b.Fatal(err)
				}
				exploreGrammar(b, g, len(vocab), 64)
			}
		})
	}
}
