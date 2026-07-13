package nlp

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// parseGBNF parses the GBNF subset documented on GrammarGuide into g's rule tables and
// returns the id of the "root" rule. Repetition and grouping are desugared into fresh
// unnamed rules, so the automaton only ever sees character classes, references, and
// alternation — the minimal PDA alphabet.
func (g *GrammarGuide) parseGBNF(src string) (int, error) {
	p := &gbnfParser{g: g, src: src, ruleID: map[string]int{}}
	for {
		p.skipSpace(true)
		if p.pos >= len(p.src) {
			break
		}
		if err := p.rule(); err != nil {
			return 0, err
		}
	}
	for name, id := range p.ruleID {
		if g.ruleAlts[id] == nil {
			return 0, fmt.Errorf("rule %q is referenced but never defined", name)
		}
	}
	root, ok := p.ruleID["root"]
	if !ok {
		return 0, fmt.Errorf(`grammar has no "root" rule`)
	}
	return root, nil
}

type gbnfParser struct {
	g      *GrammarGuide
	src    string
	pos    int
	ruleID map[string]int
}

// ruleFor returns (creating on first sight) the rule id for name; definedness is checked
// after the whole grammar is parsed so rules may be referenced before their definition.
func (p *gbnfParser) ruleFor(name string) int {
	if id, ok := p.ruleID[name]; ok {
		return id
	}
	id := p.newRule()
	p.ruleID[name] = id
	return id
}

// newRule allocates a fresh rule id with a nil (not yet defined) alternate list.
func (p *gbnfParser) newRule() int {
	p.g.ruleAlts = append(p.g.ruleAlts, nil)
	return len(p.g.ruleAlts) - 1
}

// define installs the given alternates (each a sequence of elements) as a rule's body.
func (p *gbnfParser) define(rule int, alternates [][]gElem) {
	ids := make([]int, len(alternates))
	for i, seq := range alternates {
		if seq == nil {
			seq = []gElem{}
		}
		p.g.alts = append(p.g.alts, seq)
		ids[i] = len(p.g.alts) - 1
	}
	p.g.ruleAlts[rule] = ids
}

func (p *gbnfParser) rule() error {
	name := p.ident()
	if name == "" {
		return fmt.Errorf("expected rule name at offset %d", p.pos)
	}
	p.skipSpace(false)
	if !strings.HasPrefix(p.src[p.pos:], "::=") {
		return fmt.Errorf("rule %q: expected ::=", name)
	}
	p.pos += 3
	id := p.ruleFor(name)
	if p.g.ruleAlts[id] != nil {
		return fmt.Errorf("rule %q defined twice", name)
	}
	alternates, err := p.alternates(name)
	if err != nil {
		return err
	}
	p.define(id, alternates)
	return nil
}

// alternates parses `sequence ("|" sequence)*`; a rule body ends at a newline that is not
// followed by a continuation ("|" or more of the same alternate), a ")" or end of input.
func (p *gbnfParser) alternates(rule string) ([][]gElem, error) {
	var out [][]gElem
	for {
		seq, err := p.sequence(rule)
		if err != nil {
			return nil, err
		}
		out = append(out, seq)
		p.skipSpace(true)
		if p.pos < len(p.src) && p.src[p.pos] == '|' {
			p.pos++
			continue
		}
		return out, nil
	}
}

// sequence parses terms until `|`, `)`, end of line (the rule-body boundary — a
// continuation `|` on the next line is picked up by alternates), or end of input.
func (p *gbnfParser) sequence(rule string) ([]gElem, error) {
	seq := []gElem{}
	for {
		if p.skipSpace(false) {
			return seq, nil // stopped at a newline: this body line is complete
		}
		if p.pos >= len(p.src) {
			return seq, nil
		}
		c := p.src[p.pos]
		if c == '|' || c == ')' {
			return seq, nil
		}
		elems, err := p.term(rule)
		if err != nil {
			return nil, err
		}
		seq = append(seq, elems...)
	}
}

// term parses one base (literal, class, group, or reference) plus an optional * + ?.
func (p *gbnfParser) term(rule string) ([]gElem, error) {
	base, err := p.base(rule)
	if err != nil {
		return nil, err
	}
	if p.pos >= len(p.src) {
		return base, nil
	}
	switch p.src[p.pos] {
	case '*': // R ::= base R | ε
		p.pos++
		r := p.newRule()
		p.define(r, [][]gElem{append(append([]gElem{}, base...), gElem{rule: r}), nil})
		return []gElem{{rule: r}}, nil
	case '+': // base R  with  R ::= base R | ε
		p.pos++
		r := p.newRule()
		p.define(r, [][]gElem{append(append([]gElem{}, base...), gElem{rule: r}), nil})
		return append(append([]gElem{}, base...), gElem{rule: r}), nil
	case '?': // R ::= base | ε
		p.pos++
		r := p.newRule()
		p.define(r, [][]gElem{base, nil})
		return []gElem{{rule: r}}, nil
	}
	return base, nil
}

func (p *gbnfParser) base(rule string) ([]gElem, error) {
	switch c := p.src[p.pos]; {
	case c == '"':
		return p.literal(rule)
	case c == '[':
		e, err := p.charClass(rule)
		if err != nil {
			return nil, err
		}
		return []gElem{e}, nil
	case c == '(':
		p.pos++
		alternates, err := p.groupAlternates(rule)
		if err != nil {
			return nil, err
		}
		if p.pos >= len(p.src) || p.src[p.pos] != ')' {
			return nil, fmt.Errorf("rule %q: unclosed group", rule)
		}
		p.pos++
		r := p.newRule()
		p.define(r, alternates)
		return []gElem{{rule: r}}, nil
	default:
		if name := p.ident(); name != "" {
			return []gElem{{rule: p.ruleFor(name)}}, nil
		}
		return nil, fmt.Errorf("rule %q: unexpected character %q at offset %d", rule, p.src[p.pos], p.pos)
	}
}

// groupAlternates parses alternates inside ( ... ), where newlines are plain whitespace.
func (p *gbnfParser) groupAlternates(rule string) ([][]gElem, error) {
	var out [][]gElem
	for {
		seq := []gElem{}
		for {
			p.skipSpace(true)
			if p.pos >= len(p.src) {
				return nil, fmt.Errorf("rule %q: unclosed group", rule)
			}
			if c := p.src[p.pos]; c == '|' || c == ')' {
				break
			}
			elems, err := p.term(rule)
			if err != nil {
				return nil, err
			}
			seq = append(seq, elems...)
		}
		out = append(out, seq)
		if p.src[p.pos] == '|' {
			p.pos++
			continue
		}
		return out, nil // at ')'
	}
}

// literal parses "..." into one single-rune character class per rune.
func (p *gbnfParser) literal(rule string) ([]gElem, error) {
	p.pos++ // opening quote
	var seq []gElem
	for {
		if p.pos >= len(p.src) {
			return nil, fmt.Errorf("rule %q: unterminated string literal", rule)
		}
		if p.src[p.pos] == '"' {
			p.pos++
			return seq, nil
		}
		r, err := p.escapedRune(rule, '"')
		if err != nil {
			return nil, err
		}
		seq = append(seq, gElem{ranges: []runeRange{{r, r}}, rule: -1})
	}
}

// charClass parses [...] (with optional leading ^) into one character-class element.
func (p *gbnfParser) charClass(rule string) (gElem, error) {
	p.pos++ // opening bracket
	e := gElem{rule: -1}
	if p.pos < len(p.src) && p.src[p.pos] == '^' {
		e.neg = true
		p.pos++
	}
	for {
		if p.pos >= len(p.src) {
			return e, fmt.Errorf("rule %q: unterminated character class", rule)
		}
		if p.src[p.pos] == ']' {
			p.pos++
			if len(e.ranges) == 0 {
				return e, fmt.Errorf("rule %q: empty character class", rule)
			}
			return e, nil
		}
		lo, err := p.escapedRune(rule, ']')
		if err != nil {
			return e, err
		}
		hi := lo
		if p.pos+1 < len(p.src) && p.src[p.pos] == '-' && p.src[p.pos+1] != ']' {
			p.pos++
			hi, err = p.escapedRune(rule, ']')
			if err != nil {
				return e, err
			}
			if hi < lo {
				return e, fmt.Errorf("rule %q: inverted range in character class", rule)
			}
		}
		e.ranges = append(e.ranges, runeRange{lo, hi})
	}
}

// escapedRune decodes the next (possibly escaped) rune of a literal or character class.
func (p *gbnfParser) escapedRune(rule string, delim byte) (rune, error) {
	r, size := utf8.DecodeRuneInString(p.src[p.pos:])
	if r != '\\' {
		p.pos += size
		return r, nil
	}
	p.pos++
	if p.pos >= len(p.src) {
		return 0, fmt.Errorf("rule %q: dangling escape", rule)
	}
	c := p.src[p.pos]
	p.pos++
	switch c {
	case 'n':
		return '\n', nil
	case 'r':
		return '\r', nil
	case 't':
		return '\t', nil
	case '\\', '/', '-', '^', ']', '[', delim:
		return rune(c), nil
	case 'x', 'u':
		n := 2
		if c == 'u' {
			n = 4
		}
		if p.pos+n > len(p.src) {
			return 0, fmt.Errorf("rule %q: truncated \\%c escape", rule, c)
		}
		var v rune
		for i := 0; i < n; i++ {
			d := hexVal(p.src[p.pos+i])
			if d < 0 {
				return 0, fmt.Errorf("rule %q: invalid \\%c escape", rule, c)
			}
			v = v<<4 | rune(d)
		}
		p.pos += n
		return v, nil
	}
	return 0, fmt.Errorf("rule %q: unknown escape \\%c", rule, c)
}

func hexVal(b byte) int {
	switch {
	case '0' <= b && b <= '9':
		return int(b - '0')
	case 'a' <= b && b <= 'f':
		return int(b-'a') + 10
	case 'A' <= b && b <= 'F':
		return int(b-'A') + 10
	}
	return -1
}

// ident consumes a rule name ([a-zA-Z_][a-zA-Z0-9_-]*).
func (p *gbnfParser) ident() string {
	start := p.pos
	for p.pos < len(p.src) {
		c := p.src[p.pos]
		ok := c == '_' || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')
		if p.pos > start {
			ok = ok || c == '-' || ('0' <= c && c <= '9')
		}
		if !ok {
			break
		}
		p.pos++
	}
	return p.src[start:p.pos]
}

// skipSpace consumes spaces, tabs, comments and — when newlines is true — line breaks.
// It reports whether a line break was crossed.
func (p *gbnfParser) skipSpace(newlines bool) (crossed bool) {
	for p.pos < len(p.src) {
		switch c := p.src[p.pos]; {
		case c == ' ' || c == '\t' || c == '\r':
			p.pos++
		case c == '#':
			for p.pos < len(p.src) && p.src[p.pos] != '\n' {
				p.pos++
			}
		case c == '\n':
			crossed = true
			if !newlines {
				return crossed
			}
			p.pos++
		default:
			return crossed
		}
	}
	return crossed
}
