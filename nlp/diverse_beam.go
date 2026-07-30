package nlp

import (
	"fmt"
	"math"
	"slices"
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

// dbsCand is a scored candidate extension, kept as a lightweight backpointer into the group's
// CURRENT beams instead of a materialized token sequence (T942). The full toks slice
// (make+copy+append) is built ONLY for the B' survivors that advance — not for every ~vocab
// candidate that is scored and then discarded, which dominated allocation.
//
// aug is the diversity-augmented sort key, computed ONCE when the candidate is appended. It
// used to be recomputed inside the comparator, which called it twice per comparison — about
// 98 thousand evaluations per group-step at benchmark geometry where 4096 suffice. Computing
// it at append time is valid because it reads stepCount, which is frozen for the whole of a
// group's candidate construction and selection and only mutated after the group's survivors
// are chosen.
//
// Declared at package scope, with its comparator, so both the selection and the prefix sort
// receive a STATIC function value rather than a shared closure.
type dbsCand struct {
	parent int     // index into the group's current beams
	tok    int     // token appended to parent (meaningful only for a freshly extended candidate)
	score  float64 // raw cumulative log-prob (parent.score+lp fresh; parent.score for a carried done beam)
	aug    float64 // score minus the diversity penalty; the sort key
	newLen int
	done   bool
}

// dbsCandByAug is a STRICT TOTAL ORDER: augmented score descending, then parent ascending,
// then token ascending. Totality is what lets the selection below stand in for a sort — with
// ties the retained set would not be unique. Candidates are appended parent-outer, and within
// a parent there is either one carried-done candidate or the fresh vocabulary expansion with
// distinct tokens, so (parent, tok) uniquely orders the appends and this reproduces the
// stable order exactly.
func dbsCandByAug(a, b dbsCand) int {
	if a.aug != b.aug {
		if a.aug > b.aug {
			return -1
		}
		return 1
	}
	if a.parent != b.parent {
		if a.parent < b.parent {
			return -1
		}
		return 1
	}
	if a.tok < b.tok {
		return -1
	}
	if a.tok > b.tok {
		return 1
	}
	return 0
}

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

	grp := make([][]node, groups)
	for g := range grp {
		grp[g] = []node{{append([]int(nil), start...), 0, 0, false}}
	}

	// Reused across every group of every step: the candidate buffer (whose old per-group
	// capacity hint of len(beams)*8 was ~256x short of the bPrime*vocab it grows to, costing a
	// full doubling chain per group-step) and the log-softmax row. Both are reset, never
	// aliased — survivors copy candidate VALUES out, and each log-softmax row is consumed
	// before the next is produced.
	var cands []dbsCand
	var lsBuf []float64

	for step := 0; step < maxNew; step++ {
		// token -> #earlier groups (this step) that picked it. Dense []int over the
		// [0,vocab) token domain (sized on the first logits row) replaces a per-step
		// map[int]int: aug() reads it twice per sort comparison, so this drops the hash
		// from the comparator hot path. Zero-init []int == map zero for absent keys.
		var stepCount []int
		anyLive := false
		for g := 0; g < groups; g++ {
			beams := grp[g]
			cands = cands[:0]
			for pi := range beams {
				b := beams[pi]
				if b.done {
					// a carried finished hypothesis keeps its score and takes no diversity penalty
					cands = append(cands, dbsCand{pi, 0, b.score, b.score, b.newLen, true})
					continue
				}
				anyLive = true
				ls := logSoftmaxRowInto(&lsBuf, next(b.toks))
				if stepCount == nil {
					stepCount = make([]int, len(ls))
				}
				if cap(cands) < len(beams)*len(ls) {
					cands = make([]dbsCand, 0, len(beams)*len(ls))
				}
				for tok, lp := range ls {
					sc := b.score + lp
					cands = append(cands, dbsCand{pi, tok, sc, sc - lambda*float64(stepCount[tok]), b.newLen + 1, eos >= 0 && tok == eos})
				}
			}
			// Select this group's top B' by the diversity-augmented score; the penalty applies
			// only to candidates freshly extended THIS step (carried done beams keep their
			// score). Only bPrime candidates survive, so ordering the rest is wasted work
			// (PS6022): at width 8 in 4 groups over a 2048 vocabulary that is a sort of 4096
			// to keep 2. selectTopK is an INTROSELECT — a plain quickselect degenerates on
			// candidate arrays built parent-outer over a smooth logit curve.
			fresh := func(c dbsCand) bool { return c.newLen == step+1 }
			if bPrime < len(cands) {
				selectTopK(cands, bPrime, dbsCandByAug)
				cands = cands[:bPrime]
			}
			//perfscan:ignore PS3002 composite key (aug, parent, tok); a radix needs one monotonic key, and this sorts at most bPrime elements
			slices.SortFunc(cands, dbsCandByAug)
			// materialize toks ONLY for the B' survivors that advance.
			survivors := make([]node, len(cands))
			for i, c := range cands {
				p := beams[c.parent]
				if !fresh(c) {
					survivors[i] = p // carried done beam — shares toks (never mutated)
					continue
				}
				nt := make([]int, len(p.toks)+1)
				copy(nt, p.toks)
				nt[len(p.toks)] = c.tok
				survivors[i] = node{nt, c.score, c.newLen, c.done}
			}
			grp[g] = survivors
			// record this group's fresh picks so later groups diversify against them.
			for _, c := range cands {
				if fresh(c) {
					stepCount[c.tok]++
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
	// slices.SortStableFunc rather than sort.SliceStable: the latter reaches its swap through
	// reflectlite.Swapper and allocates on every call (PS6009). Score-only comparison is not a
	// total order, so stability is load-bearing and this one must stay a stable SORT.
	//perfscan:ignore PS6022 stability is load-bearing here — score alone is not a total order, so this must stay a stable SORT
	//perfscan:ignore PS3002 a radix would not preserve the stable order this relies on
	slices.SortStableFunc(out, func(a, b Beam) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		return 0
	})
	if len(out) > width {
		out = out[:width]
	}
	return out, nil
}
