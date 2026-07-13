package nlp_test

import (
	"encoding/json"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// asciiVocab builds a single-character vocabulary covering ASCII printables plus \n,
// approximating a byte-level tokenizer for guide tests.
func asciiVocab() []string {
	var v []string
	for c := rune(32); c < 127; c++ {
		v = append(v, string(c))
	}
	return append(v, "\n")
}

// jsonGBNF is the classic compact JSON grammar (llama.cpp json.gbnf class) — the
// canonical beyond-regular workload: objects nest inside arrays inside objects.
const jsonGBNF = `
root ::= value
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" [a-zA-Z0-9 ]? [a-zA-Z0-9 ]? [a-zA-Z0-9 ]? "\""
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)?
ws ::= " "?
`

// §V16 tier-1: the PDA accepts exactly the balanced-nesting strings a regex cannot
// express — S ::= "(" S ")" | "" over vocab {(,)}.
func TestGrammarGuideBalancedParens(t *testing.T) {
	vocab := []string{"(", ")"}
	g, err := nlp.NewGrammarGuide(`root ::= "(" root ")" | ""`, vocab)
	if err != nil {
		t.Fatal(err)
	}
	st := g.Start()
	if !g.Accepting(st) {
		t.Error("empty string is in the language")
	}
	walk := func(s string) int {
		cur := g.Start()
		for _, c := range s {
			id := 0
			if c == ')' {
				id = 1
			}
			cur = g.Advance(cur, id)
		}
		return cur
	}
	for depth := 1; depth <= 24; depth++ {
		s := strings.Repeat("(", depth) + strings.Repeat(")", depth)
		if st := walk(s); !g.Accepting(st) {
			t.Fatalf("balanced depth %d must accept", depth)
		}
	}
	for _, bad := range []string{")", "(()", "())", "()("} {
		st := walk(bad)
		if st >= 0 && g.Accepting(st) {
			t.Errorf("%q must not be accepted", bad)
		}
	}
	// "()(" — after a complete match, further input must be dead or non-accepting.
	if st := walk("()("); st >= 0 && g.Accepting(st) {
		t.Error("continuation past a complete root match must not accept")
	}
}

// §V16 tier-1: masking semantics — from the start of a keyword-only grammar exactly the
// keyword's first characters are allowed, EOS only at accepting states, dead state masks all.
func TestGrammarGuideMaskLogitsAndTermination(t *testing.T) {
	vocab := []string{"t", "r", "u", "e", "x"}
	const eos = 5
	g, err := nlp.NewGrammarGuide(`root ::= "true"`, vocab)
	if err != nil {
		t.Fatal(err)
	}
	st := g.Start()
	logits := make([]float64, 6)
	if eosOK := g.MaskLogits(st, logits, eos); eosOK {
		t.Error("EOS must be disallowed before the keyword is complete")
	}
	if logits[0] != 0 { // 't' allowed
		t.Error("'t' must stay allowed at start")
	}
	for i := 1; i < 5; i++ {
		if !math.IsInf(logits[i], -1) {
			t.Errorf("token %d must be masked at start", i)
		}
	}
	if !math.IsInf(logits[eos], -1) {
		t.Error("EOS logit must be masked when not accepting")
	}
	for _, tok := range []int{0, 1, 2, 3} { // t r u e
		st = g.Advance(st, tok)
		if st < 0 {
			t.Fatal("keyword prefix must stay alive")
		}
	}
	if !g.Accepting(st) {
		t.Error("full keyword must accept")
	}
	logits = make([]float64, 6)
	if eosOK := g.MaskLogits(st, logits, eos); !eosOK {
		t.Error("EOS must be allowed at the accepting state")
	}
	if g.Advance(st, 0) >= 0 {
		t.Error("input past the complete keyword must be dead")
	}
	if g.Allowed(-1, 0) {
		t.Error("the dead state allows nothing")
	}
}

// §V16 tier-1: multi-character tokens advance through several grammar positions at once,
// and a token is allowed only if EVERY character survives.
func TestGrammarGuideMultiCharTokens(t *testing.T) {
	vocab := []string{"tr", "ue", "tx", "true"}
	g, err := nlp.NewGrammarGuide(`root ::= "true"`, vocab)
	if err != nil {
		t.Fatal(err)
	}
	st := g.Start()
	if !g.Allowed(st, 0) || g.Allowed(st, 2) {
		t.Error(`"tr" must be allowed, "tx" not`)
	}
	if !g.Accepting(g.Advance(g.Advance(st, 0), 1)) {
		t.Error(`"tr"+"ue" must reach the accepting state`)
	}
	if !g.Accepting(g.Advance(st, 3)) {
		t.Error(`the single token "true" must reach the accepting state`)
	}
}

// §V16 tier-1 e2e: guided generation with a random inner sampler over the full JSON
// grammar produces strings that ALWAYS parse as JSON — nesting included (§T575: the
// property RegexGuide structurally cannot deliver).
func TestGrammarGuideGeneratesValidJSON(t *testing.T) {
	vocab := asciiVocab()
	g, err := nlp.NewGrammarGuide(jsonGBNF, vocab)
	if err != nil {
		t.Fatal(err)
	}
	eos := len(vocab)
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 32; trial++ {
		s := g.Sampler(randomAllowedSampler{rng}, eos)
		var out strings.Builder
		done := false
		for step := 0; step < 4000 && !done; step++ {
			logits := make([]float64, len(vocab)+1)
			for i := range logits {
				logits[i] = rng.NormFloat64()
			}
			logits[eos] += 3 // bias toward finishing so trials stay short
			tok := s.Sample(logits)
			if tok == eos {
				done = true
				break
			}
			out.WriteString(vocab[tok])
		}
		if !done {
			t.Fatalf("trial %d: generation did not reach EOS (fixed seed: deterministic)", trial)
		}
		text := out.String()
		if !json.Valid([]byte(text)) {
			t.Fatalf("trial %d: generated %q is not valid JSON", trial, text)
		}
	}
}

// randomAllowedSampler picks uniformly among non-masked tokens — an adversarial inner
// sampler: any grammar hole would surface as invalid output.
type randomAllowedSampler struct{ rng *rand.Rand }

func (r randomAllowedSampler) Sample(logits []float64) int {
	var live []int
	for i, l := range logits {
		if !math.IsInf(l, -1) {
			live = append(live, i)
		}
	}
	if len(live) == 0 {
		return 0
	}
	return live[r.rng.Intn(len(live))]
}

func (r randomAllowedSampler) SampleWithHistory(logits []float64, _ []int) int {
	return r.Sample(logits)
}

// §V16 tier-1: grammar-level rejections — left recursion, missing root, undefined
// references, and syntax errors all fail at compile time with an error, never at decode.
func TestGrammarGuideCompileErrors(t *testing.T) {
	vocab := []string{"a"}
	cases := map[string]string{
		"left recursion":       `root ::= root "a" | "a"`,
		"indirect left rec":    "root ::= b\nb ::= root | \"a\"",
		"nullable left rec":    "root ::= e root \"a\" | \"a\"\ne ::= \"\"",
		"no root":              `top ::= "a"`,
		"undefined ref":        `root ::= missing`,
		"unterminated literal": `root ::= "a`,
		"unterminated class":   `root ::= [a-z`,
		"inverted range":       `root ::= [z-a]`,
		"double definition":    "root ::= \"a\"\nroot ::= \"b\"",
		"dangling escape":      `root ::= "\`,
	}
	for name, src := range cases {
		if _, err := nlp.NewGrammarGuide(src, vocab); err == nil {
			t.Errorf("%s: grammar %q must be rejected", name, src)
		}
	}
}

// §V16 tier-1: GBNF surface — comments, multi-line alternates, groups, all repetition
// operators, escapes, and negated classes parse and constrain as documented.
func TestGrammarGuideGBNFSurface(t *testing.T) {
	src := `
# a comment
root ::= greeting (" " name)? "!"+
greeting ::= "hi"
	| "hey"          # continuation line
name ::= [A-Z] [^ !]*
`
	vocab := asciiVocab()
	g, err := nlp.NewGrammarGuide(src, vocab)
	if err != nil {
		t.Fatal(err)
	}
	accepts := func(s string) bool {
		st := g.Start()
		for _, c := range s {
			id := -1
			for i, v := range vocab {
				if v == string(c) {
					id = i
					break
				}
			}
			st = g.Advance(st, id)
			if st < 0 {
				return false
			}
		}
		return g.Accepting(st)
	}
	for _, ok := range []string{"hi!", "hey!!", "hi Bob!", "hey X!!!"} {
		if !accepts(ok) {
			t.Errorf("%q must be accepted", ok)
		}
	}
	for _, bad := range []string{"hi", "yo!", "hi bob!", "hi Bob", "hi  Bob!"} {
		if accepts(bad) {
			t.Errorf("%q must be rejected", bad)
		}
	}
}

// FuzzGrammarParse: arbitrary input must never panic or hang the GBNF parser — it either
// compiles or returns an error (§T575 hostile-input discipline, §V16 tier-2).
func FuzzGrammarParse(f *testing.F) {
	f.Add(jsonGBNF)
	f.Add(`root ::= "(" root ")" | ""`)
	f.Add("root ::= [a-z]+ (\"x\" | y)*\ny ::= \"y\"?")
	f.Add(`root ::= "\x41B" [^\]-]`)
	f.Fuzz(func(t *testing.T, src string) {
		g, err := nlp.NewGrammarGuide(src, []string{"a", "b"})
		if err != nil {
			return
		}
		st := g.Start() // exercising the automaton must be safe on any accepted grammar
		g.Accepting(st)
		g.Advance(st, 0)
		g.Advance(st, 1)
	})
}
