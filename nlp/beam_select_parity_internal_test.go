package nlp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// beamSearchFullSort is the pre-selection BeamSearch, transcribed verbatim: sort every
// candidate, walk them all, and let the frontier check do the stopping. It exists only as the
// reference for TestBeamSelectMatchesFullSort.
//
// Transcribed rather than rewritten, per the reference-mirrors-shape rule: the comparator, the
// append order, the walk, the completion predicate and the trailing stable sort must all match
// the old code exactly, or the test measures the transcription instead of the change.
func beamSearchFullSort(next NextLogits, start []int, width, maxNew, eos int, alpha float64) []Beam {
	type node struct {
		toks   []int
		score  float64
		newLen int
	}
	type cnd struct {
		parent int
		tok    int
		score  float64
		newLen int
	}
	lenPenalty := func(n int) float64 {
		if alpha == 0 {
			return 1
		}
		return math.Pow((5+float64(n))/6, alpha)
	}
	live := []node{{append([]int(nil), start...), 0, 0}}
	var done []Beam
	for len(live) > 0 {
		cand := make([]cnd, 0, len(live)*8)
		for p, h := range live {
			ls := logSoftmaxRow(next(h.toks))
			for tok, l := range ls {
				cand = append(cand, cnd{p, tok, h.score + l, h.newLen + 1})
			}
		}
		// insertion-independent full order: score desc, then parent asc, then tok asc
		sortSliceStableByCnd(cand, func(a, b cnd) int {
			if a.score != b.score {
				if a.score > b.score {
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
		var nextLive []node
		for _, c := range cand {
			if len(nextLive) >= width {
				break
			}
			pt := live[c.parent].toks
			nt := make([]int, len(pt)+1)
			copy(nt, pt)
			nt[len(pt)] = c.tok
			complete := (eos >= 0 && c.tok == eos) || c.newLen >= maxNew
			if complete {
				done = append(done, Beam{nt, c.score / lenPenalty(c.newLen)})
			} else {
				nextLive = append(nextLive, node{nt, c.score, c.newLen})
			}
		}
		live = nextLive
		if len(done) >= width {
			break
		}
	}
	sortSliceStableByBeam(done, func(a, b Beam) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		return 0
	})
	if len(done) > width {
		done = done[:width]
	}
	return done
}

// Deliberately naive stable sorts, so the reference cannot inherit a bug from whatever sort
// the implementation happens to use.
func sortSliceStableByCnd[T any](s []T, cmp func(a, b T) int) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && cmp(s[j], s[j-1]) < 0; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

func sortSliceStableByBeam[T any](s []T, cmp func(a, b T) int) { sortSliceStableByCnd(s, cmp) }

// TestBeamSelectMatchesFullSort is the §V22 gate for replacing the full candidate sort with a
// bounded selection. Equality is EXACT on both the token sequences and the score bit patterns.
//
// The configurations are chosen to attack the two ways a selection can legitimately diverge
// from a sort. Ties: a selection keeps an arbitrary member of an equal-key band, so the
// tie-heavy arms use a logit vector with long runs of exactly-equal values, which makes
// candidate scores collide exactly. Terminal steps: the tail is dropped there even though the
// old walk read all of it, so maxNew is swept small enough that the terminal step dominates,
// and alpha is swept because a non-zero length penalty is what makes raw and final order
// differ across steps.
func TestBeamSelectMatchesFullSort(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 41))
	mkLogits := func(vocab int, tieRun int) []float64 {
		v := make([]float64, vocab)
		for i := range v {
			if tieRun > 1 {
				// long runs of exactly-equal logits: candidate scores then collide exactly,
				// which is where a selection and a sort may keep different elements.
				v[i] = float64(i/tieRun) * 0.25
				continue
			}
			v[i] = rng.NormFloat64() * 2
		}
		return v
	}
	cases := 0
	for _, vocab := range []int{3, 7, 16, 64} {
		for _, tieRun := range []int{1, 3, 8} {
			for _, width := range []int{1, 2, 3, 8} {
				for _, maxNew := range []int{1, 2, 3, 5} {
					for _, eos := range []int{-1, 0, 2} {
						for _, alpha := range []float64{0, 0.6, 1.5} {
							if eos >= vocab {
								continue
							}
							base := mkLogits(vocab, tieRun)
							// A prefix-dependent perturbation, so different beams see
							// different distributions and the frontier genuinely reorders.
							next := func(prefix []int) []float64 {
								out := make([]float64, len(base))
								last := prefix[len(prefix)-1]
								for i := range base {
									out[i] = base[i] + math.Sin(float64((last+1)*(i+1)))*0.5
								}
								return out
							}
							got := BeamSearch(next, []int{1}, width, maxNew, eos, alpha)
							want := beamSearchFullSort(next, []int{1}, width, maxNew, eos, alpha)
							cases++
							if len(got) != len(want) {
								t.Fatalf("vocab=%d tieRun=%d width=%d maxNew=%d eos=%d alpha=%g: %d beams, want %d",
									vocab, tieRun, width, maxNew, eos, alpha, len(got), len(want))
							}
							for i := range want {
								if len(got[i].Tokens) != len(want[i].Tokens) {
									t.Fatalf("vocab=%d tieRun=%d width=%d maxNew=%d eos=%d alpha=%g beam %d: %v, want %v",
										vocab, tieRun, width, maxNew, eos, alpha, i, got[i].Tokens, want[i].Tokens)
								}
								for j := range want[i].Tokens {
									if got[i].Tokens[j] != want[i].Tokens[j] {
										t.Fatalf("vocab=%d tieRun=%d width=%d maxNew=%d eos=%d alpha=%g beam %d: %v, want %v",
											vocab, tieRun, width, maxNew, eos, alpha, i, got[i].Tokens, want[i].Tokens)
									}
								}
								if math.Float64bits(got[i].Score) != math.Float64bits(want[i].Score) {
									t.Fatalf("vocab=%d tieRun=%d width=%d maxNew=%d eos=%d alpha=%g beam %d score %016x, want %016x",
										vocab, tieRun, width, maxNew, eos, alpha, i,
										math.Float64bits(got[i].Score), math.Float64bits(want[i].Score))
								}
							}
						}
					}
				}
			}
		}
	}
	t.Logf("selection matches the full sort exactly across %d configurations", cases)
}

// TestSelectTopKPartitions pins selectTopK itself: after the call the prefix must be exactly
// the k first elements under the comparator, as a SET. It is checked against a sorted copy, so
// the assertion does not depend on selectTopK's own internal ordering.
func TestSelectTopKPartitions(t *testing.T) {
	cmp := func(a, b int) int { return a - b }
	rng := rand.New(rand.NewPCG(7, 8))
	for _, n := range []int{1, 2, 5, 17, 64, 501} {
		for _, k := range []int{0, 1, 2, n / 3, n - 1, n, n + 5} {
			if k < 0 {
				continue
			}
			s := make([]int, n)
			for i := range s {
				s[i] = rng.IntN(1 << 20) // distinct with overwhelming probability
			}
			ref := append([]int(nil), s...)
			sortSliceStableByCnd(ref, cmp)
			selectTopK(s, k, cmp)
			if k <= 0 || k >= n {
				continue // no partition promised
			}
			gotSet := map[int]int{}
			for _, v := range s[:k] {
				gotSet[v]++
			}
			for _, v := range ref[:k] {
				if gotSet[v] == 0 {
					t.Fatalf("n=%d k=%d: %d belongs in the top-k prefix but is not there", n, k, v)
				}
				gotSet[v]--
			}
		}
	}
}
