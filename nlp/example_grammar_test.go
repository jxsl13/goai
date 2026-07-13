package nlp_test

import (
	"fmt"
	"math"
	"strings"

	"github.com/jxsl13/goai/nlp"
)

// A balanced-parentheses grammar — nesting no regex can express — walked token by token:
// Start gives the initial state, Allowed/Advance drive it, Accepting gates termination.
func ExampleGrammarGuide() {
	vocab := []string{"(", ")", "x"}
	g, err := nlp.NewGrammarGuide(`root ::= "(" root ")" | ""`, vocab)
	if err != nil {
		panic(err)
	}

	st := g.Start()
	fmt.Println(g.Accepting(st))                    // the empty string is in the language
	fmt.Println(g.Allowed(st, 0), g.Allowed(st, 2)) // "(" may start, "x" may not
	st = g.Advance(st, 0)                           // consume "("
	fmt.Println(g.Accepting(st))                    // "(" alone is not balanced
	st = g.Advance(st, 1)                           // consume ")"
	fmt.Println(g.Accepting(st))
	// Output:
	// true
	// true false
	// false
	// true
}

// NewGrammarGuide rejects grammars the automaton cannot run — here left recursion.
func ExampleNewGrammarGuide() {
	_, err := nlp.NewGrammarGuide(`root ::= root "a" | "a"`, []string{"a"})
	fmt.Println(err != nil)
	// Output: true
}

// MaskLogits disables every off-grammar token in place; EOS stays masked until the
// grammar accepts.
func ExampleGrammarGuide_MaskLogits() {
	vocab := []string{"t", "x"}
	g, err := nlp.NewGrammarGuide(`root ::= "t"`, vocab)
	if err != nil {
		panic(err)
	}

	logits := []float64{0, 0}
	eosAllowed := g.MaskLogits(g.Start(), logits, -1 /*no EOS token*/)
	fmt.Println(eosAllowed, logits[0], math.IsInf(logits[1], -1))
	// Output: false 0 true
}

// Sampler plugs the grammar into any generation loop: greedy alone would emit "))" here,
// the guided wrapper is forced onto the only balanced completion.
func ExampleGrammarGuide_Sampler() {
	vocab := []string{"(", ")"}
	g, err := nlp.NewGrammarGuide(`root ::= "(" root ")" | ""`, vocab)
	if err != nil {
		panic(err)
	}

	s := g.Sampler(nlp.Greedy(), -1 /*no EOS token*/) // stateful: fresh Sampler per sequence
	biased := []float64{0, 1}                         // ")" always looks better to greedy
	fmt.Println(s.Sample(biased), s.Sample(biased))
	// Output: 0 1
}

// A JSON Schema compiles to a grammar that forces conforming output: build the guide from
// the generated GBNF and generation can only ever produce documents with an integer "id".
func ExampleJSONSchemaToGrammar() {
	gbnf, err := nlp.JSONSchemaToGrammar([]byte(`{
	  "type": "object",
	  "properties": {"id": {"type": "integer"}},
	  "required": ["id"]
	}`))
	if err != nil {
		panic(err)
	}
	fmt.Println(strings.Split(gbnf, "\n")[0]) // the root rule of the generated grammar

	var vocab []string // one token per printable ASCII character
	for c := rune(32); c < 127; c++ {
		vocab = append(vocab, string(c))
	}
	g, err := nlp.NewGrammarGuide(gbnf, vocab)
	if err != nil {
		panic(err)
	}
	st := g.Start()
	for _, c := range `{"id":42}` {
		st = g.Advance(st, int(c)-32)
	}
	fmt.Println(g.Accepting(st))
	// Output:
	// root ::= "{" sp "\"id\"" sp ":" sp json-integer sp "}"
	// true
}
