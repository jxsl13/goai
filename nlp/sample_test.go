package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// Greedy picks the argmax deterministically, ignoring temperature/RNG.
func TestGreedySampler(t *testing.T) {
	s := nlp.Greedy()
	for range 5 {
		if got := s.Sample([]float64{0.1, 2.5, -1, 0.3}); got != 1 {
			t.Fatalf("greedy = %d, want 1", got)
		}
	}
}

// top-k restricts the support to the k highest-logit tokens.
func TestTopKSupport(t *testing.T) {
	// logits favor tokens 3 and 1; k=2 → only {1,3} may be sampled
	logits := []float64{0.0, 3.0, 0.5, 4.0, 1.0}
	s := nlp.NewSampler(7, nlp.WithTopK(2))
	seen := map[int]int{}
	for range 2000 {
		seen[s.Sample(logits)]++
	}
	for tok := range seen {
		if tok != 1 && tok != 3 {
			t.Errorf("top-k=2 sampled disallowed token %d", tok)
		}
	}
	if seen[3] == 0 || seen[1] == 0 {
		t.Error("both top-2 tokens should appear")
	}
}

// top-p nucleus keeps the smallest desc-prob prefix reaching p; the crossing
// token is included and at least the top token always survives.
func TestTopPNucleus(t *testing.T) {
	// probs ∝ softmax; token 0 dominates. With p=0.5 and a dominant token, only
	// the top token(s) needed to reach 0.5 are kept.
	logits := []float64{5.0, 3.0, 1.0, 0.0} // softmax ≈ [0.84,0.11,0.015,0.006]
	s := nlp.NewSampler(3, nlp.WithTopP(0.5))
	seen := map[int]bool{}
	for range 3000 {
		seen[s.Sample(logits)] = true
	}
	// top token prob ≈0.84 ≥0.5 already → nucleus = {0} only
	if len(seen) != 1 || !seen[0] {
		t.Errorf("top-p=0.5 nucleus = %v, want {0}", seen)
	}

	// p=0.95 → need tokens 0 and 1 (0.84+0.11=0.95)
	s2 := nlp.NewSampler(3, nlp.WithTopP(0.95))
	seen2 := map[int]bool{}
	for range 5000 {
		seen2[s2.Sample(logits)] = true
	}
	if seen2[2] || seen2[3] {
		t.Errorf("top-p=0.95 leaked low-prob tokens: %v", seen2)
	}
	if !seen2[0] || !seen2[1] {
		t.Errorf("top-p=0.95 should keep tokens 0 and 1: %v", seen2)
	}
}

// Sampling is deterministic under a fixed seed.
func TestSamplerDeterministic(t *testing.T) {
	logits := []float64{1, 2, 3, 0.5, -1}
	a := nlp.NewSampler(42, nlp.WithTemperature(0.8))
	b := nlp.NewSampler(42, nlp.WithTemperature(0.8))
	for range 100 {
		if a.Sample(logits) != b.Sample(logits) {
			t.Fatal("same seed must produce same samples")
		}
	}
}

// End-to-end greedy generation using the KV-cache: deterministic and the right
// length.
func TestGenerateGreedy(t *testing.T) {
	model, prompt := loadGPTModel(t)
	out1, err := model.Generate(prompt, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out1) != len(prompt)+3 {
		t.Fatalf("generated %d tokens, want %d", len(out1), len(prompt)+3)
	}
	// greedy is deterministic
	out2, _ := model.Generate(prompt, 3, nlp.Greedy())
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatal("greedy generation must be deterministic")
		}
	}
	// generated tokens must be valid ids
	for _, tok := range out1 {
		if tok < 0 || tok >= model.Config.Vocab {
			t.Fatalf("invalid token %d", tok)
		}
	}
}

// Temperature → 0 approaches greedy; large temperature broadens the support.
func TestTemperatureEffect(t *testing.T) {
	logits := []float64{0.2, 2.0, 0.1, 1.0, 0.3}
	cold := nlp.NewSampler(1, nlp.WithTemperature(0.01))
	seen := map[int]int{}
	for range 500 {
		seen[cold.Sample(logits)]++
	}
	if seen[1] < 480 { // near-greedy → almost always the argmax (token 1)
		t.Errorf("cold temperature not near-greedy: %v", seen)
	}
	hot := nlp.NewSampler(1, nlp.WithTemperature(5.0))
	seenHot := map[int]bool{}
	for range 2000 {
		seenHot[hot.Sample(logits)] = true
	}
	if len(seenHot) < 4 { // hot → most tokens appear
		t.Errorf("hot temperature too narrow: %v", seenHot)
	}
}

// §B73 tier-1: min-p must never zero the arg-max. The threshold MinP·maxProb exceeds
// the peak once MinP > 1, so without the min_tokens_to_keep=1 guard EVERY token is
// zeroed, the renormalizer divides by a zero sum, and the whole distribution becomes
// NaN. Covers in-range (behaviour must be UNCHANGED), the exact boundary MinP == 1,
// out-of-range, and a flat distribution where every token ties for the peak.
func TestMinPArgmaxAlwaysSurvives(t *testing.T) {
	peaked := []float64{1, 0, 0, 0, 0} // one clear winner
	flat := []float64{0, 0, 0, 0}      // every token tied
	for _, tc := range []struct {
		name  string
		logit []float64
		minP  float64
		want  []float64 // nil = only check the invariants
	}{
		{"in-range/peaked", peaked, 0.05, nil},
		{"in-range/half", peaked, 0.5, nil},
		{"boundary/one", peaked, 1.0, []float64{1, 0, 0, 0, 0}},
		{"out-of-range/1.5", peaked, 1.5, []float64{1, 0, 0, 0, 0}},
		{"out-of-range/5", peaked, 5, []float64{1, 0, 0, 0, 0}},
		{"flat/half", flat, 0.5, []float64{0.25, 0.25, 0.25, 0.25}},
		{"flat/one", flat, 1.0, []float64{0.25, 0.25, 0.25, 0.25}},
		{"flat/out-of-range", flat, 1.5, []float64{1, 0, 0, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// MinP is set on the FIELD, not via WithMinP: out-of-range values are
			// rejected by the option (TestSamplerOptionRejectsOutOfRange), and the
			// filter itself must be total regardless of how the field got its value.
			s := &nlp.Sampler{Temperature: 1, MinP: tc.minP}
			d := s.Dist(tc.logit)

			var sum float64
			for i, p := range d {
				if math.IsNaN(p) {
					t.Fatalf("MinP=%v produced NaN at %d: %v", tc.minP, i, d)
				}
				sum += p
			}
			if math.Abs(sum-1) > 1e-12 {
				t.Errorf("MinP=%v distribution sums to %v, want 1: %v", tc.minP, sum, d)
			}
			// THE invariant the old comment claimed but did not enforce.
			top := 0
			for i, v := range tc.logit {
				if v > tc.logit[top] {
					top = i
				}
			}
			if d[top] <= 0 {
				t.Errorf("MinP=%v zeroed the arg-max (token %d): %v", tc.minP, top, d)
			}
			for i, want := range tc.want {
				if math.Abs(d[i]-want) > 1e-12 {
					t.Errorf("MinP=%v dist[%d] = %v, want %v (full %v)", tc.minP, i, d[i], want, d)
				}
			}
		})
	}
}

// §B73 tier-1: the user-visible symptom. An all-NaN distribution makes every `u <= cum`
// comparison in Sample false, so the draw fell through to `return len(probs)-1` — the
// LAST vocabulary id, deterministically, every single step. The arg-max is the only
// defensible answer here.
func TestMinPOutOfRangeSamplesArgmaxNotLastToken(t *testing.T) {
	s := nlp.NewSampler(1) // via the constructor, so the rng is seeded...
	s.MinP = 1.5           // ...then out of range on the field, bypassing WithMinP
	logits := []float64{1, 0, 0, 0, 0}
	seen := map[int]int{}
	for range 1000 {
		seen[s.Sample(logits)]++
	}
	if seen[0] != 1000 {
		t.Errorf("MinP=1.5 sampled %v, want all 1000 draws on the arg-max (token 0); "+
			"token %d is the last vocab id, the classic NaN fall-through", seen, len(logits)-1)
	}
}

// §B73: out-of-range knob values are REJECTED at construction, not silently absorbed.
// WithMinP(5) is min-p confused with a percentage; before the fix it emitted the last
// vocabulary token forever, in silence.
func TestSamplerOptionRejectsOutOfRange(t *testing.T) {
	for name, opt := range map[string]nlp.SamplerOption{
		"MinP>1":         nlp.WithMinP(5),
		"MinP=1.5":       nlp.WithMinP(1.5),
		"MinP<0":         nlp.WithMinP(-0.1),
		"TopP=90":        nlp.WithTopP(90),
		"TopP=1.5":       nlp.WithTopP(1.5),
		"TopP<0":         nlp.WithTopP(-1),
		"TopK<0":         nlp.WithTopK(-5),
		"Typical>1":      nlp.WithTypical(1.5),
		"Epsilon>1":      nlp.WithEpsilon(2),
		"Eta<0":          nlp.WithEta(-1),
		"XTCProb>1":      nlp.WithXTC(50),
		"XTCThreshold>1": nlp.WithXTCThreshold(1.5),
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("%s was silently accepted, want a panic", name)
				}
			}()
			nlp.NewSampler(1, opt)
		})
	}
}

// The documented SPECIAL VALUES are not errors and must keep working exactly as before:
// 0 disables every filter, TopP/Typical == 1 is the "keep everything" idiom, and
// Temperature ≤ 0 selects greedy. Rejecting these would break documented behaviour.
func TestSamplerOptionAcceptsDocumentedSpecialValues(t *testing.T) {
	for name, opt := range map[string]nlp.SamplerOption{
		"TopP=1":     nlp.WithTopP(1),
		"TopP=0":     nlp.WithTopP(0),
		"Typical=1":  nlp.WithTypical(1),
		"MinP=1":     nlp.WithMinP(1),
		"MinP=0":     nlp.WithMinP(0),
		"TopK=0":     nlp.WithTopK(0),
		"TopK=huge":  nlp.WithTopK(1 << 20),
		"Temp<0":     nlp.WithTemperature(-0.5),
		"Temp=0":     nlp.WithTemperature(0),
		"XTCThr=0.6": nlp.WithXTCThreshold(0.6),
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s is documented behaviour but panicked: %v", name, r)
				}
			}()
			nlp.NewSampler(1, opt).Sample([]float64{1, 2, 3, 0.5, -1})
		})
	}
}

// §B73: every knob is an exported field, which invites a struct literal — and rng is
// unexported, so a literal leaves it nil. Sample must seed it lazily rather than
// dereference nil. Covers the plain draw and the XTC path, which draws from the same
// rng before Sample reaches its own.
func TestSamplerZeroValueDoesNotPanic(t *testing.T) {
	logits := []float64{1, 5, 2}
	for name, s := range map[string]*nlp.Sampler{
		"greedy-by-default": {TopK: 40, TopP: 0.9},
		"with-temperature":  {TopK: 40, TopP: 0.9, Temperature: 0.8},
		"with-xtc":          {Temperature: 0.8, XTCProbability: 1, XTCThreshold: 0.1},
		"bare":              {Temperature: 1},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("zero-value Sampler panicked: %v", r)
				}
			}()
			for range 50 {
				if got := s.Sample(logits); got < 0 || got >= len(logits) {
					t.Fatalf("sampled out-of-range token %d", got)
				}
			}
		})
	}
}

// Degenerate input must not panic anywhere in the filter chain. An empty logit vector
// reaches every truncation block with a zero-length probs slice, and min-p in
// particular now indexes the peak to build its threshold — that index needs a guard.
func TestSamplerEmptyLogitsDoesNotPanic(t *testing.T) {
	for name, s := range map[string]*nlp.Sampler{
		"min-p":   {Temperature: 1, MinP: 0.05},
		"minp>1":  {Temperature: 1, MinP: 5},
		"top-p":   {Temperature: 1, TopP: 0.9},
		"typical": {Temperature: 1, Typical: 0.9},
		"epsilon": {Temperature: 1, Epsilon: 0.1},
		"eta":     {Temperature: 1, Eta: 0.1},
		"greedy":  {Temperature: 0},
		"all":     {Temperature: 1, TopK: 40, TopP: 0.9, MinP: 0.05, Typical: 0.9, Epsilon: 0.1, Eta: 0.1},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("empty logits panicked: %v", r)
				}
			}()
			if d := s.Dist(nil); len(d) != 0 {
				t.Errorf("Dist(nil) = %v, want empty", d)
			}
			s.Sample(nil)
		})
	}
}

// The arg-max survives every filter EXCEPT locally-typical, and the Sampler doc now
// says so. Typical deliberately trims the too-predictable, so at a small τ it zeroes
// the most probable token — that is the method working, not a bug like §B73's min-p.
// Pinned here so the documented exception stays honest in both directions.
func TestTypicalMayDropArgmaxButNeverEmpties(t *testing.T) {
	logits := []float64{math.Log(0.5), math.Log(0.25), math.Log(0.125), math.Log(0.125)}
	if d := (&nlp.Sampler{Temperature: 1, Typical: 0.2}).Dist(logits); d[0] != 0 {
		t.Errorf("typical τ=0.2 kept the arg-max (%v) — the doc claims it may drop it", d)
	}
	// ...but never leaves an empty or NaN distribution, at any τ.
	for _, tau := range []float64{0.01, 0.2, 0.5, 0.9, 1} {
		d := (&nlp.Sampler{Temperature: 1, Typical: tau}).Dist(logits)
		var sum float64
		for _, p := range d {
			if math.IsNaN(p) {
				t.Fatalf("typical τ=%v produced NaN: %v", tau, d)
			}
			sum += p
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Errorf("typical τ=%v sums to %v, want 1: %v", tau, sum, d)
		}
	}
}
