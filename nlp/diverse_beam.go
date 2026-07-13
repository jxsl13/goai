package nlp

import (
	"fmt"
	"math"
	"sort"
)

// Diverse Beam Search (Vijayakumar, Cogswell, Selvaraju, Sun, Lee, Crandall & Batra 2018,
// "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models", AAAI,
// arXiv:1610.02424). Ordinary beam search (nlp.BeamSearch, §R54) returns near-identical
// hypotheses that differ only in a few tokens; DBS fixes that by splitting the beam into G
// GROUPS decoded SEQUENTIALLY at each time step, adding a between-group DIVERSITY PENALTY so
// later groups are pushed away from the tokens earlier groups already chose this step.
//
// With B=width beams in G=groups groups of B'=B/G each, at time step t group g scores every
// candidate extension by
//
//	augmented = Σ log p  +  λ·Δ(token) ,   Δ(token) = −(# earlier groups g'<g that picked token at t)
//
// (the standard Hamming diversity), so a token already selected by k earlier groups is penalized
// by λ·k. Each group keeps its own top B' by the augmented score; the diversity term steers the
// SEARCH only — the returned hypotheses are ranked by their raw length-normalized log-probability
// (Wu 2016 penalty lp(n)=((5+n)/6)^alpha, as in BeamSearch). λ=0 reduces to G independent beam
// searches (identical groups); larger λ trades likelihood for diversity (the paper reports λ in
// 0.2–0.8 works well, grid-searched per task — there is no single canonical default).

// DiverseBeamSearch decodes width beams in `groups` groups (width must be divisible by groups),
// up to maxNew new tokens, completing a hypothesis on `eos` (eos<0 disables). lambda is the
// diversity strength and alpha the length penalty. It returns the completed/decoded hypotheses,
// best raw length-normalized log-prob first (at most `width`).
func DiverseBeamSearch(next NextLogits, start []int, width, groups, maxNew, eos int, alpha, lambda float64) ([]Beam, error) {
	if width <= 0 || groups <= 0 {
		return nil, fmt.Errorf("nlp: DiverseBeamSearch width and groups must be positive, got %d/%d", width, groups)
	}
	if width%groups != 0 {
		return nil, fmt.Errorf("nlp: DiverseBeamSearch width %d not divisible by groups %d", width, groups)
	}
	bPrime := width / groups
	lenPenalty := func(n int) float64 {
		if alpha == 0 {
			return 1
		}
		return math.Pow((5+float64(n))/6, alpha)
	}
	type node struct {
		toks   []int
		score  float64 // raw cumulative log-prob
		newLen int
		done   bool
	}
	last := func(nd node) int { return nd.toks[len(nd.toks)-1] }

	grp := make([][]node, groups)
	for g := range grp {
		grp[g] = []node{{append([]int(nil), start...), 0, 0, false}}
	}

	for step := 0; step < maxNew; step++ {
		stepCount := map[int]int{} // token → # earlier groups (this step) that selected it
		anyLive := false
		for g := 0; g < groups; g++ {
			cand := make([]node, 0, len(grp[g])*8)
			for _, b := range grp[g] {
				if b.done {
					cand = append(cand, b) // carry a finished hypothesis unchanged
					continue
				}
				anyLive = true
				ls := logSoftmaxRow(next(b.toks))
				for tok, lp := range ls {
					nt := make([]int, len(b.toks)+1)
					copy(nt, b.toks)
					nt[len(b.toks)] = tok
					cand = append(cand, node{nt, b.score + lp, b.newLen + 1, eos >= 0 && tok == eos})
				}
			}
			// select this group's top B' by the diversity-augmented score; the penalty applies
			// only to candidates freshly extended THIS step (carried done beams keep their score).
			fresh := func(nd node) bool { return nd.newLen == step+1 }
			aug := func(nd node) float64 {
				if fresh(nd) {
					return nd.score - lambda*float64(stepCount[last(nd)])
				}
				return nd.score
			}
			sort.SliceStable(cand, func(i, j int) bool { return aug(cand[i]) > aug(cand[j]) })
			if len(cand) > bPrime {
				cand = cand[:bPrime]
			}
			grp[g] = cand
			// record this group's fresh picks so later groups diversify against them.
			for _, b := range grp[g] {
				if fresh(b) {
					stepCount[last(b)]++
				}
			}
		}
		if !anyLive {
			break
		}
	}

	var out []Beam
	for g := range grp {
		for _, b := range grp[g] {
			out = append(out, Beam{b.toks, b.score / lenPenalty(b.newLen)})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if len(out) > width {
		out = out[:width]
	}
	return out, nil
}
