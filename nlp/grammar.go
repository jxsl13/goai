package nlp

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// GrammarGuide implements grammar-constrained decoding at CONTEXT-FREE level (§T575, the
// llama.cpp GBNF class): a grammar written in a GBNF subset is compiled to a pushdown
// automaton over the token vocabulary, and during generation the next-token logits are
// MASKED so only tokens whose characters keep at least one parse stack alive are allowed.
// After a token is chosen the automaton is ADVANCED by consuming its characters. Generation
// may stop (EOS) only when some stack has fully consumed the root rule. By construction
// every produced string is in the grammar's language.
//
// This is strictly more expressive than RegexGuide (§R111): a context-free grammar can
// enforce NESTED structure — JSON objects inside arrays inside objects, balanced brackets —
// which no finite-state machine can (pumping lemma). Use RegexGuide for flat patterns
// (dates, identifiers, enums); use GrammarGuide when the output must recurse. For JSON
// outputs conforming to a schema, JSONSchemaToGrammar compiles the schema to a grammar
// for this guide.
//
// The automaton state is a SET of parse stacks (the standard nondeterministic PDA
// construction, as in llama.cpp's llama_grammar): each stack records where in which rule
// alternates parsing stands. States are interned to small integer ids and per-token
// transitions are memoized, mirroring RegexGuide's lazy Willard-&-Louf-style indexing, so
// per-step masking after warm-up is an O(1) lookup rather than an O(vocab) parse.
//
// Supported GBNF subset: `name ::= ...` rules with alternation `|`, sequencing, grouping
// `( ... )`, repetition `* + ?`, string literals `"..."` (escapes \" \\ \n \r \t \xHH
// \uXXXX), character classes `[a-z0-9]` / negated `[^"\\]` (same escapes), rule
// references, and `#` line comments. Generation starts at the rule named "root".
// Left-recursive grammars are rejected at compile time (a left-recursive rule would make
// the expansion step loop forever; rewrite with repetition instead).
type GrammarGuide struct {
	alts     [][]gElem // every alternate of every rule, flattened
	ruleAlts [][]int   // rule id → indices into alts
	vocab    []string  // token id → its character string
	states   []*gramState
	intern   map[string]int
}

// gElem is one grammar element in an alternate's sequence: a character class
// (rule < 0) or a reference to another rule (rule >= 0).
type gElem struct {
	ranges []runeRange
	neg    bool
	rule   int
}

type runeRange struct{ lo, hi rune }

func (e gElem) matches(r rune) bool {
	in := false
	for _, rr := range e.ranges {
		if rr.lo <= r && r <= rr.hi {
			in = true
			break
		}
	}
	return in != e.neg
}

// gPos is a position inside one alternate: the element at alts[alt][idx] is next to match.
type gPos struct{ alt, idx int32 }

// gStack is one parse stack, bottom to top; the TOP (last) position is matched next.
type gStack []gPos

// gramState is one interned automaton state: the set of resting parse stacks (each empty
// or with a character class on top), whether any stack is empty (accepting), and the
// lazily-filled per-token transition memo.
type gramState struct {
	stacks  []gStack
	accept  bool
	tokNext []int // token id → next state id, or -1; nil until first use
}

// NewGrammarGuide compiles a GBNF grammar into a guide over vocab (token id → the token's
// character string). The grammar must define a rule named "root"; the guide enforces that
// the ENTIRE generated string derives from it. Returns an error for syntax errors,
// references to undefined rules, and left-recursive grammars.
func NewGrammarGuide(grammar string, vocab []string) (*GrammarGuide, error) {
	g := &GrammarGuide{vocab: vocab, intern: map[string]int{}}
	root, err := g.parseGBNF(grammar)
	if err != nil {
		return nil, fmt.Errorf("nlp: GrammarGuide: %w", err)
	}
	if err := g.checkLeftRecursion(); err != nil {
		return nil, fmt.Errorf("nlp: GrammarGuide: %w", err)
	}
	// normalize: rule 0 is root (parseGBNF guarantees root exists and returns its id).
	if root != 0 {
		g.ruleAlts[0], g.ruleAlts[root] = g.ruleAlts[root], g.ruleAlts[0]
		for i := range g.alts {
			for j := range g.alts[i] {
				switch g.alts[i][j].rule {
				case 0:
					g.alts[i][j].rule = root
				case root:
					g.alts[i][j].rule = 0
				}
			}
		}
	}
	return g, nil
}

// Start returns the guide's initial automaton state id.
func (g *GrammarGuide) Start() int {
	var out []gStack
	seen := map[string]bool{}
	for _, alt := range g.ruleAlts[0] {
		if len(g.alts[alt]) == 0 { // an empty root alternate: accepting from the start
			g.expand(gStack{}, &out, seen, 0)
		} else {
			g.expand(gStack{{int32(alt), 0}}, &out, seen, 0)
		}
	}
	return g.internStacks(out)
}

// Accepting reports whether state is an accepting state — the root rule is fully derived
// and generation may terminate (EOS) here, and only here.
func (g *GrammarGuide) Accepting(state int) bool {
	if state < 0 || state >= len(g.states) {
		return false
	}
	return g.states[state].accept
}

// Allowed reports whether the token with the given id is valid from state (every one of
// its characters keeps at least one parse stack alive). A dead state (-1) allows nothing.
func (g *GrammarGuide) Allowed(state, token int) bool {
	return g.Advance(state, token) >= 0
}

// Advance consumes the token's characters from state and returns the resulting state id,
// or -1 if the token is invalid (kills every parse stack). Results are memoized per
// (state, token).
func (g *GrammarGuide) Advance(state, token int) int {
	if state < 0 || state >= len(g.states) || token < 0 || token >= len(g.vocab) {
		return -1
	}
	s := g.states[state]
	if s.tokNext == nil {
		s.tokNext = make([]int, len(g.vocab))
		for i := range s.tokNext {
			s.tokNext[i] = -2 // not yet computed
		}
	}
	if s.tokNext[token] != -2 {
		return s.tokNext[token]
	}
	stacks := s.stacks
	for _, r := range g.vocab[token] {
		stacks = g.consume(stacks, r)
		if len(stacks) == 0 {
			s.tokNext[token] = -1
			return -1
		}
	}
	next := g.internStacks(stacks)
	s.tokNext[token] = next
	return next
}

// MaskLogits sets the logit of every token disallowed from state to −Inf, in place, and
// returns whether EOS (ending generation) is allowed — i.e. whether state is accepting.
// eosID, if in range, is force-allowed exactly when the state is accepting (and masked out
// otherwise). A dead state masks everything. logits must be vocab-sized.
func (g *GrammarGuide) MaskLogits(state int, logits []float64, eosID int) (eosAllowed bool) {
	acc := g.Accepting(state)
	for t := range logits {
		if t == eosID {
			continue
		}
		if !g.Allowed(state, t) {
			logits[t] = math.Inf(-1)
		}
	}
	if eosID >= 0 && eosID < len(logits) && !acc {
		logits[eosID] = math.Inf(-1)
	}
	return acc
}

// Sampler wraps inner into a TokenSampler that enforces the grammar during generation,
// exactly like RegexGuide.Sampler (§T439): each draw masks disallowed tokens to −Inf from
// the current automaton state, lets inner pick, and advances by the picked token. eosID
// (−1 for none) is only ever allowed in accepting states. The wrapper is stateful: use a
// fresh Sampler per generated sequence.
func (g *GrammarGuide) Sampler(inner TokenSampler, eosID int) TokenSampler {
	return &guidedSampler{g: g, inner: inner, eos: eosID, state: g.Start()}
}

// consume advances every resting stack by one rune and returns the new resting stack set
// (deduplicated). Empty result means the rune is invalid from this state.
func (g *GrammarGuide) consume(stacks []gStack, r rune) []gStack {
	var out []gStack
	seen := map[string]bool{}
	for _, st := range stacks {
		if len(st) == 0 {
			continue // fully matched stacks cannot consume more input
		}
		top := st[len(st)-1]
		elem := g.alts[top.alt][top.idx]
		if !elem.matches(r) { // resting stacks always have a char class on top
			continue
		}
		next := append(gStack(nil), st[:len(st)-1]...)
		if int(top.idx)+1 < len(g.alts[top.alt]) {
			next = append(next, gPos{top.alt, top.idx + 1})
		}
		g.expand(next, &out, seen, 0)
	}
	return out
}

// expand rewrites st into resting form — empty, or a character class on top — by
// replacing top rule references with each of their alternates (the nondeterministic PDA
// expansion). Results are appended to out, deduplicated via seen. Left recursion is
// rejected at compile time, so the recursion terminates; depth is a defensive cap.
func (g *GrammarGuide) expand(st gStack, out *[]gStack, seen map[string]bool, depth int) {
	if depth > 4096 {
		return // defensive: drop pathological expansions rather than recurse forever
	}
	if len(st) == 0 {
		if !seen[""] {
			seen[""] = true
			*out = append(*out, gStack{})
		}
		return
	}
	top := st[len(st)-1]
	elem := g.alts[top.alt][top.idx]
	if elem.rule < 0 { // character class: resting
		key := stackKey(st)
		if !seen[key] {
			seen[key] = true
			*out = append(*out, st)
		}
		return
	}
	base := append(gStack(nil), st[:len(st)-1]...)
	if int(top.idx)+1 < len(g.alts[top.alt]) {
		base = append(base, gPos{top.alt, top.idx + 1})
	}
	for _, alt := range g.ruleAlts[elem.rule] {
		if len(g.alts[alt]) == 0 {
			g.expand(append(gStack(nil), base...), out, seen, depth+1)
		} else {
			g.expand(append(append(gStack(nil), base...), gPos{int32(alt), 0}), out, seen, depth+1)
		}
	}
}

// internStacks interns a resting stack set to a stable state id (-1 when empty = dead).
func (g *GrammarGuide) internStacks(stacks []gStack) int {
	if len(stacks) == 0 {
		return -1
	}
	keys := make([]string, len(stacks))
	order := make([]int, len(stacks))
	for i, st := range stacks {
		keys[i] = stackKey(st)
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return keys[order[a]] < keys[order[b]] })
	var sb strings.Builder
	sorted := make([]gStack, len(stacks))
	accept := false
	for i, o := range order {
		sorted[i] = stacks[o]
		sb.WriteString(keys[o])
		sb.WriteByte(';')
		if len(stacks[o]) == 0 {
			accept = true
		}
	}
	key := sb.String()
	if id, ok := g.intern[key]; ok {
		return id
	}
	id := len(g.states)
	g.states = append(g.states, &gramState{stacks: sorted, accept: accept})
	g.intern[key] = id
	return id
}

func stackKey(st gStack) string {
	b := make([]byte, 0, len(st)*6)
	for _, p := range st {
		b = strconv.AppendInt(b, int64(p.alt), 36)
		b = append(b, '.')
		b = strconv.AppendInt(b, int64(p.idx), 36)
		b = append(b, ',')
	}
	return string(b)
}

// checkLeftRecursion rejects grammars where a rule can re-enter itself without consuming
// input (direct or via nullable prefixes) — those would loop the expansion step forever.
func (g *GrammarGuide) checkLeftRecursion() error {
	nullable := g.nullableRules()
	// first[r] = rules reachable at the leftmost position of r, skipping nullable refs.
	first := make([][]int, len(g.ruleAlts))
	for r, alts := range g.ruleAlts {
		set := map[int]bool{}
		for _, a := range alts {
			for _, e := range g.alts[a] {
				if e.rule < 0 {
					break // a char class consumes input: nothing further is leftmost
				}
				set[e.rule] = true
				if !nullable[e.rule] {
					break
				}
			}
		}
		for k := range set {
			first[r] = append(first[r], k)
		}
	}
	// cycle detection over the first-graph.
	state := make([]int, len(g.ruleAlts)) // 0 unvisited, 1 in progress, 2 done
	var visit func(r int) bool
	visit = func(r int) bool {
		if state[r] == 1 {
			return false
		}
		if state[r] == 2 {
			return true
		}
		state[r] = 1
		for _, n := range first[r] {
			if !visit(n) {
				return false
			}
		}
		state[r] = 2
		return true
	}
	for r := range g.ruleAlts {
		if !visit(r) {
			return fmt.Errorf("grammar is left-recursive (rule cycle without consuming input); rewrite with repetition")
		}
	}
	return nil
}

// nullableRules computes which rules can derive the empty string (fixpoint iteration).
func (g *GrammarGuide) nullableRules() []bool {
	nullable := make([]bool, len(g.ruleAlts))
	for changed := true; changed; {
		changed = false
		for r, alts := range g.ruleAlts {
			if nullable[r] {
				continue
			}
			for _, a := range alts {
				all := true
				for _, e := range g.alts[a] {
					if e.rule < 0 || !nullable[e.rule] {
						all = false
						break
					}
				}
				if all {
					nullable[r] = true
					changed = true
					break
				}
			}
		}
	}
	return nullable
}
