package nlp

import (
	"fmt"
	"math"
	"regexp/syntax"
	"slices"
	"strconv"
)

// RegexGuide implements regex/FSM-guided constrained decoding (Willard & Louf 2023, §R111, "Efficient
// Guided Generation for Large Language Models", arXiv:2307.09702 — the method behind Outlines). A
// regular expression is compiled to a finite-state machine; during generation the next-token logits
// are MASKED so only tokens whose characters keep the FSM in a live (non-dead) state are allowed, and
// after a token is chosen the FSM state is ADVANCED by consuming that token's characters. Sampling may
// stop (EOS) only when the FSM is in an ACCEPTING state. By construction every produced string is in
// the regular language — the output is guaranteed to match the regex.
//
// The FSM is the standard-library regexp NFA (regexp/syntax): a state is the ε-closed set of NFA
// program counters. Per the paper's efficiency contribution (§4), the state→token transitions are
// precomputed lazily and memoized, so per-step masking is an O(1) lookup rather than O(vocab) regex
// tests. States are interned to small integer ids.
//
// Empty-width assertions (^, $, \b) are treated permissively (as ε) — guided generation over the
// whole output rarely needs them, and the common structured-output patterns (character classes,
// literals, repetition, alternation) do not.
type RegexGuide struct {
	prog   *syntax.Prog
	vocab  []string      // token id → its character string
	states []*guideState // interned states, indexed by state id
	intern map[string]int
}

// guideState is one interned FSM state: the ε-closed resting program counters, whether it is
// accepting, and a lazily-filled memo of per-token transitions.
type guideState struct {
	pcs     []uint32
	accept  bool
	tokNext []int // token id → next state id, or -1 if the token is invalid; nil until computed
}

// NewRegexGuide compiles pattern into a guide over vocab (token id → the token's character string).
// The pattern must match the ENTIRE generated string.
func NewRegexGuide(pattern string, vocab []string) (*RegexGuide, error) {
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return nil, fmt.Errorf("nlp: RegexGuide parse %q: %w", pattern, err)
	}
	prog, err := syntax.Compile(re.Simplify())
	if err != nil {
		return nil, fmt.Errorf("nlp: RegexGuide compile %q: %w", pattern, err)
	}
	g := &RegexGuide{prog: prog, vocab: vocab, intern: map[string]int{}}
	g.intern[""] = -1 // reserve the dead state as id -1 conceptually (never stored)
	delete(g.intern, "")
	return g, nil
}

// Start returns the guide's initial FSM state id.
func (g *RegexGuide) Start() int {
	return g.internClosure([]uint32{uint32(g.prog.Start)})
}

// Accepting reports whether state is a match/accept state — the only states from which generation may
// terminate (EOS).
func (g *RegexGuide) Accepting(state int) bool {
	if state < 0 || state >= len(g.states) {
		return false
	}
	return g.states[state].accept
}

// Allowed reports whether the token with the given id is valid from state (every character has a live
// transition). A dead state (-1) allows nothing.
func (g *RegexGuide) Allowed(state, token int) bool {
	return g.Advance(state, token) >= 0
}

// Advance consumes the token's characters from state and returns the resulting state id, or -1 if the
// token is invalid (hits a dead state). Results are memoized per (state, token).
func (g *RegexGuide) Advance(state, token int) int {
	if state < 0 || state >= len(g.states) || token < 0 || token >= len(g.vocab) {
		return -1
	}
	s := g.states[state]
	if s.tokNext == nil {
		s.tokNext = make([]int, len(g.vocab))
		for i := range s.tokNext {
			s.tokNext[i] = -2 // -2 = not yet computed
		}
	}
	if s.tokNext[token] != -2 {
		return s.tokNext[token]
	}
	next := g.walk(s.pcs, g.vocab[token])
	s.tokNext[token] = next
	return next
}

// MaskLogits sets the logit of every token disallowed from state to −Inf, in place, and returns
// whether EOS (ending generation) is allowed — i.e. whether state is accepting. eosID, if in range, is
// force-allowed exactly when the state is accepting (and masked out otherwise). A dead state masks
// everything. logits must be vocab-sized.
func (g *RegexGuide) MaskLogits(state int, logits []float64, eosID int) (eosAllowed bool) {
	acc := g.Accepting(state)
	for t := range logits {
		if t == eosID {
			continue // handled below
		}
		if !g.Allowed(state, t) {
			logits[t] = math.Inf(-1)
		}
	}
	if eosID >= 0 && eosID < len(logits) {
		if acc {
			// leave the EOS logit as-is (allowed to end)
		} else {
			logits[eosID] = math.Inf(-1)
		}
	}
	return acc
}

// Sampler wraps inner into a TokenSampler that enforces the guide during generation
// (§T439): each draw masks the disallowed tokens to −Inf from the CURRENT FSM state
// (MaskLogits on a copy), lets inner pick, and advances the state by the picked
// token — so it plugs straight into any generation loop that takes a TokenSampler.
// eosID (−1 for none) is only ever allowed in accepting states; picking it stops
// advancing. The wrapper is stateful: use a fresh Sampler per generated sequence,
// and choose a pattern that always has a continuation (a dead end masks everything).
//
// Pass [WithEOS] to the Generate call to have eosID actually END the loop: the
// returned sampler implements [StopTokener], so Generate honours it without the id
// being repeated. WITHOUT that option the loop still runs the full maxNew steps,
// and every token after the EOS comes from the frozen accepting state — call
// EOSEmitted ([EOSReporter]) on the sampler afterwards to detect that tail (§B76).
func (g *RegexGuide) Sampler(inner TokenSampler, eosID int) TokenSampler {
	return &guidedSampler{g: g, inner: inner, eos: eosID, state: g.Start()}
}

// tokenGuide is the mask-and-advance contract shared by RegexGuide (FSM) and
// GrammarGuide (PDA): both drive the same guidedSampler wrapper.
type tokenGuide interface {
	Start() int
	Advance(state, token int) int
	MaskLogits(state int, logits []float64, eosID int) bool
}

type guidedSampler struct {
	g          tokenGuide
	inner      TokenSampler
	eos, state int
	drewEOS    bool // sticky: set the first time eos is drawn
}

// StopTokens reports the guide's eos id (none when it was built with eos < 0),
// implementing [StopTokener]. A Generate loop armed with [WithEOS] therefore stops
// on this sampler's own eos without the caller repeating the id — which is what
// keeps a guide that has reached an accepting state from drawing EOS and then
// emitting filler from the frozen FSM state for the rest of maxNew (§B76).
func (s *guidedSampler) StopTokens() []int {
	if s.eos < 0 {
		return nil
	}
	return []int{s.eos}
}

// EOSEmitted reports whether this sampler has drawn its eos id, implementing
// [EOSReporter]. It stays true once set. A caller that does NOT pass [WithEOS] —
// and so still receives the full maxNew tokens — can check it after Generate to
// detect that the tail is post-EOS filler rather than guided output.
func (s *guidedSampler) EOSEmitted() bool { return s.drewEOS }

func (s *guidedSampler) masked(logits []float64) []float64 {
	out := append([]float64(nil), logits...)
	s.g.MaskLogits(s.state, out, s.eos)
	return out
}

func (s *guidedSampler) advance(tok int) {
	if tok != s.eos {
		s.state = s.g.Advance(s.state, tok)
		return
	}
	// EOS: the FSM state deliberately stops advancing (it is only reachable from an
	// accepting state). Record it, so the freeze is reportable via EOSEmitted and
	// stoppable via StopTokens instead of silently producing filler (§B76).
	s.drewEOS = true
}

func (s *guidedSampler) Sample(logits []float64) int {
	tok := s.inner.Sample(s.masked(logits))
	s.advance(tok)
	return tok
}

func (s *guidedSampler) SampleWithHistory(logits []float64, history []int) int {
	tok := s.inner.SampleWithHistory(s.masked(logits), history)
	s.advance(tok)
	return tok
}

// walk feeds s's characters through the FSM from the resting PC set and returns the interned resulting
// state id, or -1 if any character has no live transition.
func (g *RegexGuide) walk(pcs []uint32, s string) int {
	cur := append([]uint32(nil), pcs...)
	for _, r := range s { // range over runes
		next := g.consume(cur, r)
		if len(next) == 0 {
			return -1 // dead
		}
		cur = next
	}
	return g.internClosure(cur)
}

// consume advances a resting PC set by one rune, returning the ε-closed next resting set (raw PCs; the
// caller interns). Empty result means a dead state.
func (g *RegexGuide) consume(resting []uint32, r rune) []uint32 {
	var raw []uint32
	for _, pc := range resting {
		inst := &g.prog.Inst[pc]
		switch inst.Op {
		case syntax.InstRune, syntax.InstRune1:
			if inst.MatchRune(r) {
				raw = append(raw, inst.Out)
			}
		case syntax.InstRuneAny:
			raw = append(raw, inst.Out)
		case syntax.InstRuneAnyNotNL:
			if r != '\n' {
				raw = append(raw, inst.Out)
			}
		}
	}
	return g.closure(raw)
}

// closure ε-closes a raw PC set into its resting set (InstRune*/InstMatch), following Alt/Nop/Capture/
// EmptyWidth as ε. It does not record acceptance (see internClosure).
func (g *RegexGuide) closure(raw []uint32) []uint32 {
	seen := map[uint32]bool{}
	var stack, rest []uint32
	stack = append(stack, raw...)
	for len(stack) > 0 {
		pc := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[pc] {
			continue
		}
		seen[pc] = true
		inst := &g.prog.Inst[pc]
		switch inst.Op {
		case syntax.InstAlt, syntax.InstAltMatch:
			stack = append(stack, inst.Out, inst.Arg)
		case syntax.InstNop, syntax.InstCapture, syntax.InstEmptyWidth:
			stack = append(stack, inst.Out)
		case syntax.InstRune, syntax.InstRune1, syntax.InstRuneAny, syntax.InstRuneAnyNotNL, syntax.InstMatch:
			rest = append(rest, pc)
		case syntax.InstFail:
			// drop
		}
	}
	// slices.Sort, not sort.Slice: the latter reaches its swap through reflectlite.Swapper and
	// ALLOCATES on every call (PS6009), and this runs once per (state, token) pair explored — a
	// few thousand times while an automaton warms up. The comparator was the IDENTITY on a
	// []uint32, so the ordered generic needs no closure at all: same ascending order, no
	// indirect call per comparison, no swapper.
	slices.Sort(rest)
	return rest
}

// internClosure ε-closes raw, interns the resulting resting set to a stable state id, and records
// whether it is accepting (contains InstMatch).
func (g *RegexGuide) internClosure(raw []uint32) int {
	rest := g.closure(raw)
	if len(rest) == 0 {
		return -1
	}
	key := pcKey(rest)
	if id, ok := g.intern[key]; ok {
		return id
	}
	accept := false
	for _, pc := range rest {
		if g.prog.Inst[pc].Op == syntax.InstMatch {
			accept = true
			break
		}
	}
	id := len(g.states)
	g.states = append(g.states, &guideState{pcs: rest, accept: accept})
	g.intern[key] = id
	return id
}

// pcKey builds a stable map key from a sorted PC set.
func pcKey(pcs []uint32) string {
	b := make([]byte, 0, len(pcs)*4)
	for _, pc := range pcs {
		b = strconv.AppendUint(b, uint64(pc), 36)
		b = append(b, ',')
	}
	return string(b)
}
