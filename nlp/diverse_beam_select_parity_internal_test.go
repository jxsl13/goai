package nlp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// dbsFullSort is the pre-selection DiverseBeamSearch, transcribed verbatim: a per-group
// candidate buffer, a full sort with the augmented key recomputed inside the comparator, a
// per-beam log-softmax allocation, and the reflect-based stable sort at the end. It exists
// only as the reference for TestDiverseBeamSelectMatchesFullSort.
//
// Transcribed rather than rewritten, per the reference-mirrors-shape rule: the augmented-key
// expression, the append order, the comparator and the truncation must match the old code
// exactly, or the test measures the transcription instead of the change.
func dbsFullSort(next NextLogits, start []int, width, groups, maxNew, eos int, alpha, lambda float64) []Beam {
	bPrime := width / groups
	lenPenalty := func(n int) float64 {
		if alpha == 0 {
			return 1
		}
		return math.Pow((5+float64(n))/6, alpha)
	}
	type node struct {
		toks   []int
		score  float64
		newLen int
		done   bool
	}
	type cand struct {
		parent int
		tok    int
		score  float64
		newLen int
		done   bool
	}
	grp := make([][]node, groups)
	for g := range grp {
		grp[g] = []node{{append([]int(nil), start...), 0, 0, false}}
	}
	for step := 0; step < maxNew; step++ {
		var stepCount []int
		anyLive := false
		for g := 0; g < groups; g++ {
			beams := grp[g]
			cands := make([]cand, 0, len(beams)*8)
			for pi := range beams {
				b := beams[pi]
				if b.done {
					cands = append(cands, cand{pi, 0, b.score, b.newLen, true})
					continue
				}
				anyLive = true
				ls := logSoftmaxRow(next(b.toks))
				if stepCount == nil {
					stepCount = make([]int, len(ls))
				}
				for tok, lp := range ls {
					cands = append(cands, cand{pi, tok, b.score + lp, b.newLen + 1, eos >= 0 && tok == eos})
				}
			}
			fresh := func(c cand) bool { return c.newLen == step+1 }
			aug := func(c cand) float64 {
				if fresh(c) {
					return c.score - lambda*float64(stepCount[c.tok])
				}
				return c.score
			}
			sortSliceStableByCnd(cands, func(a, b cand) int {
				ai, aj := aug(a), aug(b)
				if ai != aj {
					if ai > aj {
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
			})
			if len(cands) > bPrime {
				cands = cands[:bPrime]
			}
			survivors := make([]node, len(cands))
			for i, c := range cands {
				p := beams[c.parent]
				if !fresh(c) {
					survivors[i] = p
					continue
				}
				nt := make([]int, len(p.toks)+1)
				copy(nt, p.toks)
				nt[len(p.toks)] = c.tok
				survivors[i] = node{nt, c.score, c.newLen, c.done}
			}
			grp[g] = survivors
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
	sortSliceStableByBeam(out, func(a, b Beam) int {
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
	return out
}

// TestDiverseBeamSelectMatchesFullSort is the §V22 gate for the four changes to
// DiverseBeamSearch: the bounded selection, the precomputed augmented key, the reused
// candidate buffer and the reused log-softmax row. Equality is EXACT on tokens and on score
// bit patterns.
//
// The sweep attacks the specific ways each change could diverge. Ties, because a selection
// keeps an arbitrary member of an equal-key band — hence arms whose logits contain long runs
// of exactly equal values. lambda, because the diversity penalty is what makes the augmented
// key differ from the raw score, and lambda=0 collapses them. groups, because stepCount
// accumulates across groups within a step and the precomputed key must see the same snapshot
// the closure did. eos, because a completed beam is carried with a different key rule.
func TestDiverseBeamSelectMatchesFullSort(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 23))
	cases := 0
	for _, vocab := range []int{4, 9, 32} {
		// tieRun 0 is the FLAT arm: every token carries the identical logit and no
		// prefix-dependent perturbation is applied, so all candidates from one parent tie
		// exactly on score and, at lambda 0, on the augmented key as well. That is the only
		// arm that reaches the comparator's (parent, token) tie-break — mutation testing
		// showed the perturbed arms below break every tie, leaving the tie-break unexercised.
		for _, tieRun := range []int{0, 1, 4} {
			base := make([]float64, vocab)
			for i := range base {
				switch {
				case tieRun == 0:
					base[i] = 0.75
				case tieRun > 1:
					base[i] = float64(i/tieRun) * 0.3
				default:
					base[i] = rng.NormFloat64() * 2
				}
			}
			flat := tieRun == 0
			next := func(prefix []int) []float64 {
				out := make([]float64, len(base))
				if flat {
					copy(out, base)
					return out
				}
				last := prefix[len(prefix)-1]
				for i := range base {
					out[i] = base[i] + math.Sin(float64((last+1)*(i+1)))*0.5
				}
				return out
			}
			for _, wg := range [][2]int{{4, 2}, {6, 3}, {8, 4}, {4, 1}} {
				for _, maxNew := range []int{1, 2, 4} {
					for _, eos := range []int{-1, 1} {
						for _, lambda := range []float64{0, 0.5, 1.5} {
							for _, alpha := range []float64{0, 0.6} {
								width, groups := wg[0], wg[1]
								got, err := DiverseBeamSearch(next, []int{1}, width, groups, maxNew, eos, alpha, lambda)
								if err != nil {
									t.Fatal(err)
								}
								want := dbsFullSort(next, []int{1}, width, groups, maxNew, eos, alpha, lambda)
								cases++
								id := func() string {
									return "vocab/tie/width/groups/maxNew/eos/lambda/alpha"
								}
								if len(got) != len(want) {
									t.Fatalf("%s = %d/%d/%d/%d/%d/%d/%g/%g: %d beams, want %d",
										id(), vocab, tieRun, width, groups, maxNew, eos, lambda, alpha, len(got), len(want))
								}
								for i := range want {
									if len(got[i].Tokens) != len(want[i].Tokens) {
										t.Fatalf("%s = %d/%d/%d/%d/%d/%d/%g/%g beam %d: %v, want %v",
											id(), vocab, tieRun, width, groups, maxNew, eos, lambda, alpha, i, got[i].Tokens, want[i].Tokens)
									}
									for j := range want[i].Tokens {
										if got[i].Tokens[j] != want[i].Tokens[j] {
											t.Fatalf("%s = %d/%d/%d/%d/%d/%d/%g/%g beam %d: %v, want %v",
												id(), vocab, tieRun, width, groups, maxNew, eos, lambda, alpha, i, got[i].Tokens, want[i].Tokens)
										}
									}
									if math.Float64bits(got[i].Score) != math.Float64bits(want[i].Score) {
										t.Fatalf("%s = %d/%d/%d/%d/%d/%d/%g/%g beam %d: score %016x, want %016x",
											id(), vocab, tieRun, width, groups, maxNew, eos, lambda, alpha, i,
											math.Float64bits(got[i].Score), math.Float64bits(want[i].Score))
									}
								}
							}
						}
					}
				}
			}
		}
	}
	t.Logf("diverse-beam selection matches the full sort exactly across %d configurations", cases)
}

// TestLogSoftmaxRowIntoContract pins the two properties of the shared row buffer that the
// diverse-beam sweep cannot reach, because every row within one search has the same width.
// Mutation testing found the reslice-to-len clause uncovered; this is its floor.
func TestLogSoftmaxRowIntoContract(t *testing.T) {
	var buf []float64
	long := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	short := []float64{0.5, -1.5, 2.25}

	got := logSoftmaxRowInto(&buf, long)
	if len(got) != len(long) {
		t.Fatalf("length %d, want %d", len(got), len(long))
	}
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(logSoftmaxRow(long)[i]) {
			t.Fatalf("element %d differs from logSoftmaxRow", i)
		}
	}
	// Reused at a SHORTER width: the result must be resliced, not left showing stale tail
	// values from the previous, longer row.
	got = logSoftmaxRowInto(&buf, short)
	if len(got) != len(short) {
		t.Fatalf("reused buffer length %d, want %d — the result was not resliced", len(got), len(short))
	}
	want := logSoftmaxRow(short)
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("reused buffer element %d = %v, want %v", i, got[i], want[i])
		}
	}
}
