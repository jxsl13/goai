package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// firstGreen / firstRed return the lowest token id in / not in the green list for prevToken.
func pick(mask []bool, green bool) int {
	for i, g := range mask {
		if g == green {
			return i
		}
	}
	return -1
}

// buildSeq builds an all-green (green=true) or all-red sequence of the given scored length by
// choosing, at each step, a token in/out of the green list seeded by the previous token.
func buildSeq(w *nlp.Watermark, scored int, green bool) []int {
	toks := []int{0}
	for len(toks) <= scored {
		toks = append(toks, pick(w.GreenMask(toks[len(toks)-1]), green))
	}
	return toks
}

// §V16 tier-1 (Algorithm 2): the green list is a deterministic ⌊γ|V|⌋-sized partition seeded by
// the previous token.
func TestWatermarkGreenMaskDeterministic(t *testing.T) {
	w, err := nlp.NewWatermark(20, nlp.WithWatermarkGamma(0.25), nlp.WithWatermarkKey(9))
	if err != nil {
		t.Fatal(err)
	}
	m1, m2 := w.GreenMask(5), w.GreenMask(5)
	green := 0
	for i := range m1 {
		if m1[i] != m2[i] {
			t.Fatalf("GreenMask not deterministic at %d", i)
		}
		if m1[i] {
			green++
		}
	}
	if green != 5 { // ⌊0.25·20⌋
		t.Errorf("green size %d, want 5", green)
	}
	// a different predecessor should generally yield a different partition.
	if same := equalMask(m1, w.GreenMask(6)); same {
		t.Error("green list did not depend on the previous token")
	}
}

func equalMask(a, b []bool) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// §V16 tier-1 (soft watermark): BiasLogits adds exactly δ to green tokens and nothing to red.
func TestWatermarkBiasFavorsGreen(t *testing.T) {
	w, _ := nlp.NewWatermark(8, nlp.WithWatermarkGamma(0.5), nlp.WithWatermarkDelta(2.5), nlp.WithWatermarkKey(1))
	base := make([]float64, 8)
	for i := range base {
		base[i] = 0.1 * float64(i)
	}
	biased, err := w.BiasLogits(base, 3)
	if err != nil {
		t.Fatal(err)
	}
	mask := w.GreenMask(3)
	for i := range base {
		want := base[i]
		if mask[i] {
			want += 2.5
		}
		if math.Abs(biased[i]-want) > 1e-12 {
			t.Errorf("biased[%d]=%g, want %g", i, biased[i], want)
		}
	}
}

// §V16 tier-1 (z-statistic, §4): an all-green sequence gives z=(T−γT)/√(Tγ(1−γ)); with γ=0.5
// and T=16 that is exactly 4.0. An all-red sequence gives the negative mirror, −4.0.
func TestWatermarkDetectZScoreExact(t *testing.T) {
	w, _ := nlp.NewWatermark(10, nlp.WithWatermarkGamma(0.5), nlp.WithWatermarkKey(7))
	greenSeq := buildSeq(w, 16, true)
	z, g, s := w.Detect(greenSeq)
	if s != 16 || g != 16 {
		t.Fatalf("all-green: scored=%d green=%d, want 16/16", s, g)
	}
	if math.Abs(z-4.0) > 1e-9 {
		t.Errorf("all-green z=%.10g, want 4.0", z)
	}
	// a longer all-green run clears the z>4 threshold (T=25, γ=0.5 ⇒ z=5).
	if !w.IsWatermarked(buildSeq(w, 25, true)) {
		t.Error("long all-green sequence should be flagged watermarked")
	}
	redSeq := buildSeq(w, 16, false)
	zr, gr, _ := w.Detect(redSeq)
	if gr != 0 || math.Abs(zr+4.0) > 1e-9 {
		t.Errorf("all-red: green=%d z=%.10g, want 0 / −4.0", gr, zr)
	}
	if w.IsWatermarked(redSeq) {
		t.Error("all-red sequence must not be flagged")
	}
}

// §V16 tier-1: Detect's z matches an independent recount of green tokens + the closed-form z.
func TestWatermarkDetectFormula(t *testing.T) {
	w, _ := nlp.NewWatermark(50, nlp.WithWatermarkGamma(0.25), nlp.WithWatermarkKey(42))
	toks := []int{3, 17, 5, 40, 22, 8, 1, 33, 29, 14, 7}
	// independent count.
	gg := 0
	for i := 1; i < len(toks); i++ {
		if w.GreenMask(toks[i-1])[toks[i]] {
			gg++
		}
	}
	T := float64(len(toks) - 1)
	want := (float64(gg) - 0.25*T) / math.Sqrt(T*0.25*0.75)
	z, g, s := w.Detect(toks)
	if g != gg || s != len(toks)-1 || math.Abs(z-want) > 1e-12 {
		t.Errorf("Detect=(z %.10g, g %d, s %d), want (%.10g, %d, %d)", z, g, s, want, gg, len(toks)-1)
	}
}

// §V2 (edges): empty / single-token sequences have nothing to score.
func TestWatermarkEdges(t *testing.T) {
	w, _ := nlp.NewWatermark(10)
	for _, toks := range [][]int{nil, {5}} {
		if z, g, s := w.Detect(toks); z != 0 || g != 0 || s != 0 {
			t.Errorf("toks=%v: (z %g, g %d, s %d), want 0/0/0", toks, z, g, s)
		}
	}
}

func TestWatermarkErrors(t *testing.T) {
	if _, err := nlp.NewWatermark(0); err == nil {
		t.Error("vocabSize 0: want error")
	}
	if _, err := nlp.NewWatermark(10, nlp.WithWatermarkGamma(1.5)); err == nil {
		t.Error("gamma 1.5: want error")
	}
	w, _ := nlp.NewWatermark(10)
	if _, err := w.BiasLogits(make([]float64, 9), 0); err == nil {
		t.Error("logits length mismatch: want error")
	}
}

// §V15 fuzz: Detect is deterministic and satisfies its invariants for any sequence/key, and an
// EMBEDDED watermark (bias then argmax → a green token) is detected as watermarked.
func FuzzWatermark(f *testing.F) {
	f.Add([]byte("watermark me"), uint64(1))
	f.Add([]byte{}, uint64(0))
	f.Add([]byte{5, 5, 5, 5, 5}, uint64(99))
	f.Fuzz(func(t *testing.T, data []byte, key uint64) {
		const vocab = 37
		w, _ := nlp.NewWatermark(vocab, nlp.WithWatermarkKey(key), nlp.WithWatermarkGamma(0.3))
		toks := make([]int, len(data))
		for i, b := range data {
			toks[i] = int(b) % vocab
		}
		z1, g1, s1 := w.Detect(toks)
		z2, g2, s2 := w.Detect(toks)
		if z1 != z2 || g1 != g2 || s1 != s2 {
			t.Fatal("Detect not deterministic")
		}
		if g1 < 0 || g1 > s1 {
			t.Fatalf("green %d out of [0,%d]", g1, s1)
		}
		if s1 > 0 {
			exp := (float64(g1) - 0.3*float64(s1)) / math.Sqrt(float64(s1)*0.3*0.7)
			if math.Abs(z1-exp) > 1e-9 {
				t.Fatalf("z %.12g != formula %.12g", z1, exp)
			}
		}
		// embed → detect: greedily generate green tokens; the detector must fire.
		start := 0
		if len(data) > 0 {
			start = int(data[0]) % vocab
		}
		wm := []int{start}
		zero := make([]float64, vocab)
		for len(wm) < 30 {
			biased, _ := w.BiasLogits(zero, wm[len(wm)-1])
			best := 0
			for i := range biased {
				if biased[i] > biased[best] {
					best = i
				}
			}
			wm = append(wm, best)
		}
		if !w.IsWatermarked(wm) {
			z, _, _ := w.Detect(wm)
			t.Fatalf("embedded watermark not detected (z=%.4g)", z)
		}
	})
}

func ExampleWatermark() {
	w, _ := nlp.NewWatermark(32, nlp.WithWatermarkGamma(0.25), nlp.WithWatermarkKey(2024))
	// Generate greedily from flat logits under the bias: every picked token is green.
	toks := []int{0}
	zero := make([]float64, 32)
	for len(toks) < 40 {
		biased, _ := w.BiasLogits(zero, toks[len(toks)-1])
		best := 0
		for i := range biased {
			if biased[i] > biased[best] {
				best = i
			}
		}
		toks = append(toks, best)
	}
	fmt.Println(w.IsWatermarked(toks))
	// Output:
	// true
}
