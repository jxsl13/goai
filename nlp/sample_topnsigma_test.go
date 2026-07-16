package nlp_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// nsigmaSupport returns the number of tokens with nonzero probability in dist.
func nsigmaSupport(dist []float64) int {
	n := 0
	for _, p := range dist {
		if p > 0 {
			n++
		}
	}
	return n
}

// nsigmaKeepSet independently recomputes the paper's keep set
// {i : logits[i] ≥ max − n·σ} (population σ) — the reference the sampler must match.
func nsigmaKeepSet(logits []float64, n float64) map[int]bool {
	m := math.Inf(-1)
	var sum float64
	for _, v := range logits {
		if v > m {
			m = v
		}
		sum += v
	}
	mean := sum / float64(len(logits))
	var ss float64
	for _, v := range logits {
		d := v - mean
		ss += d * d
	}
	thresh := m - n*math.Sqrt(ss/float64(len(logits)))
	keep := map[int]bool{}
	for i, v := range logits {
		if v >= thresh {
			keep[i] = true
		}
	}
	return keep
}

// §V16 anchor 1 — threshold correctness on hand-constructed logits. For
// [1,1,−1,−1] the mean is 0 and the population σ is EXACTLY 1, so the survivor
// set of M − n·σ is known by hand per n, including the inclusive boundary at
// n = 2 (threshold −1 keeps the tokens sitting exactly on it). For the distinct
// vector [5,3,2,1] (σ = √2.1875 ≈ 1.479) the set grows one token at a time as n
// rises, and a tiny n keeps only the arg-max (the greedy limit).
func TestTopNSigmaThresholdExactSet(t *testing.T) {
	cases := []struct {
		name   string
		logits []float64
		n      float64
		keep   []int // exact expected survivor indices
	}{
		{"sigma1-n0.5", []float64{1, 1, -1, -1}, 0.5, []int{0, 1}},                // thresh 0.5
		{"sigma1-n1.5", []float64{1, 1, -1, -1}, 1.5, []int{0, 1}},                // thresh −0.5
		{"sigma1-n2-boundary", []float64{1, 1, -1, -1}, 2, []int{0, 1, 2, 3}},     // thresh −1, inclusive ≥
		{"distinct-tiny-n-argmax", []float64{5, 3, 2, 1}, 1e-9, []int{0}},         // greedy limit n→0⁺
		{"distinct-n1", []float64{5, 3, 2, 1}, 1, []int{0}},                       // thresh ≈ 3.521
		{"distinct-n1.5", []float64{5, 3, 2, 1}, 1.5, []int{0, 1}},                // thresh ≈ 2.781
		{"distinct-n2.5", []float64{5, 3, 2, 1}, 2.5, []int{0, 1, 2}},             // thresh ≈ 1.302
		{"distinct-large-n-all", []float64{5, 3, 2, 1}, 100, []int{0, 1, 2, 3}},   // no-op
		{"neg-logits", []float64{-2, -4, -4, -4, -6}, 1, []int{0}},                // σ≈1.265, thresh ≈ −3.265
		{"neg-logits-wider", []float64{-2, -4, -4, -4, -6}, 2, []int{0, 1, 2, 3}}, // thresh ≈ −4.530
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := map[int]bool{}
			for _, i := range tc.keep {
				want[i] = true
			}
			// the hand-derived set must agree with the independent recomputation
			if got := nsigmaKeepSet(tc.logits, tc.n); len(got) != len(want) {
				t.Fatalf("test self-check: recomputed keep set %v != hand-derived %v", got, want)
			}
			dist := nlp.NewSampler(1, nlp.WithTopNSigma(tc.n)).Dist(tc.logits)
			var ksum float64
			for i, p := range dist {
				if want[i] && p == 0 {
					t.Errorf("token %d (logit %g) must survive M−n·σ, got prob 0", i, tc.logits[i])
				}
				if !want[i] && p != 0 {
					t.Errorf("token %d (logit %g) must be cut by M−n·σ, got prob %g", i, tc.logits[i], p)
				}
				ksum += p
			}
			if math.Abs(ksum-1) > 1e-9 {
				t.Errorf("survivors must be renormalized to 1, got %g", ksum)
			}
		})
	}
}

// §V16 anchor 2 — the defining temperature-stability property (Tang et al. 2024):
// the top-nσ survivor count is IDENTICAL across temperatures (the σ-unit threshold
// scales with the logits), while top-p's nucleus balloons as temperature flattens
// the distribution. This contrast is why top-nσ exists.
func TestTopNSigmaTemperatureStability(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 0x5a3d))
	logits := make([]float64, 256)
	for i := range logits {
		logits[i] = 3 * rng.NormFloat64()
	}
	temps := []float64{0.5, 1, 2, 5}
	nsSizes := make([]int, len(temps))
	tpSizes := make([]int, len(temps))
	for k, temp := range temps {
		nsSizes[k] = nsigmaSupport(nlp.NewSampler(1, nlp.WithTemperature(temp), nlp.WithTopNSigma(1)).Dist(logits))
		tpSizes[k] = nsigmaSupport(nlp.NewSampler(1, nlp.WithTemperature(temp), nlp.WithTopP(0.9)).Dist(logits))
	}
	t.Logf("survivor counts across T=%v: top-nσ %v, top-p(0.9) %v", temps, nsSizes, tpSizes)
	for k := 1; k < len(temps); k++ {
		if nsSizes[k] != nsSizes[0] {
			t.Errorf("top-nσ survivor count must be temperature-invariant: T=%g gives %d, T=%g gives %d",
				temps[0], nsSizes[0], temps[k], nsSizes[k])
		}
	}
	if want := nsigmaKeepSet(logits, 1); nsSizes[0] != len(want) {
		t.Errorf("top-nσ survivor count %d != independent keep-set size %d", nsSizes[0], len(want))
	}
	if tpSizes[len(temps)-1] <= 4*tpSizes[0] {
		t.Errorf("top-p nucleus should balloon with temperature (the contrast): T=%g→%d, T=%g→%d",
			temps[0], tpSizes[0], temps[len(temps)-1], tpSizes[len(temps)-1])
	}
	if nsSizes[len(temps)-1] >= tpSizes[len(temps)-1] {
		t.Errorf("at high T the top-nσ set (%d) should stay below the ballooned top-p set (%d)",
			nsSizes[len(temps)-1], tpSizes[len(temps)-1])
	}
}

// §V16 anchor 3 — pipeline integration: a Sampler with WithTopNSigma only ever
// returns tokens above the M − n·σ threshold (statistical check, seeded RNG),
// composed with a non-trivial temperature; and every survivor is reachable.
func TestTopNSigmaPipelineOnlySurvivors(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 0x5a3d))
	logits := make([]float64, 24)
	for i := range logits {
		logits[i] = 2 * rng.NormFloat64()
	}
	keep := nsigmaKeepSet(logits, 1.5)
	if len(keep) < 2 || len(keep) == len(logits) {
		t.Fatalf("test vector should keep a non-trivial strict subset, kept %d/%d", len(keep), len(logits))
	}
	s := nlp.NewSampler(42, nlp.WithTopNSigma(1.5), nlp.WithTemperature(2)) // hot: stresses the invariance
	seen := map[int]int{}
	for range 5000 {
		tok := s.Sample(logits)
		if !keep[tok] {
			t.Fatalf("sampled token %d (logit %g) is below the M−n·σ threshold", tok, logits[tok])
		}
		seen[tok]++
	}
	for i := range keep {
		if seen[i] == 0 {
			t.Errorf("surviving token %d was never sampled in 5000 draws", i)
		}
	}
}

// Disabled ≤ 0 (or unset) collapses to the baseline sampler: identical Dist and
// an identical draw sequence from the same seed — top-nσ off is a true no-op.
func TestTopNSigmaDisabledCollapse(t *testing.T) {
	base := nlp.NewSampler(9)
	zero := nlp.NewSampler(9, nlp.WithTopNSigma(0))
	neg := nlp.NewSampler(9, nlp.WithTopNSigma(-1))
	rng := rand.New(rand.NewPCG(3, 0x5a3d))
	for step := range 200 {
		logits := make([]float64, 32)
		for i := range logits {
			logits[i] = 4 * rng.NormFloat64()
		}
		dBase := base.Dist(logits)
		for _, s := range []*nlp.Sampler{zero, neg} {
			d := s.Dist(logits)
			for i := range dBase {
				if d[i] != dBase[i] {
					t.Fatalf("step %d: disabled top-nσ Dist[%d]=%g differs from baseline %g", step, i, d[i], dBase[i])
				}
			}
		}
		a, b, c := base.Sample(logits), zero.Sample(logits), neg.Sample(logits)
		if a != b || a != c {
			t.Fatalf("step %d: disabled top-nσ draw (0→%d, -1→%d) differs from baseline %d", step, b, c, a)
		}
	}
}

// Determinism: two samplers with the same seed and the same top-nσ config produce
// identical draw sequences.
func TestTopNSigmaDeterministic(t *testing.T) {
	s1 := nlp.NewSampler(77, nlp.WithTopNSigma(1))
	s2 := nlp.NewSampler(77, nlp.WithTopNSigma(1))
	logits := []float64{1.5, 1.2, 0.9, 0.3, -0.5, -1.1}
	for step := range 500 {
		a, b := s1.Sample(logits), s2.Sample(logits)
		if a != b {
			t.Fatalf("step %d: same seed diverged (%d vs %d)", step, a, b)
		}
	}
}

// Degenerate cases: σ = 0 (all logits equal — nothing stands out) keeps the whole
// vocabulary uniform; a single-token vocabulary is a fixed point.
func TestTopNSigmaDegenerate(t *testing.T) {
	dist := nlp.NewSampler(1, nlp.WithTopNSigma(1)).Dist([]float64{2, 2, 2, 2})
	for i, p := range dist {
		if math.Abs(p-0.25) > 1e-12 {
			t.Errorf("σ=0 must keep everything uniform, dist[%d]=%g", i, p)
		}
	}
	s := nlp.NewSampler(1, nlp.WithTopNSigma(0.5))
	if d := s.Dist([]float64{3.7}); len(d) != 1 || d[0] != 1 {
		t.Errorf("single-token vocab Dist = %v, want [1]", d)
	}
	if tok := s.Sample([]float64{3.7}); tok != 0 {
		t.Errorf("single-token vocab Sample = %d, want 0", tok)
	}
}

// Top-nσ thresholds the raw logits at max − n·σ. Here [2,0,0,0,−2] has σ ≈ 1.26,
// so n = 1 puts the cut at ≈ 0.74 and only the top logit survives — the sampler
// is decisive even though the runner-up logits are only 2 away.
func ExampleWithTopNSigma() {
	s := nlp.NewSampler(1, nlp.WithTopNSigma(1))
	dist := s.Dist([]float64{2, 0, 0, 0, -2})
	fmt.Printf("%.2f %.2f %.2f %.2f %.2f\n", dist[0], dist[1], dist[2], dist[3], dist[4])
	// Output: 1.00 0.00 0.00 0.00 0.00
}
