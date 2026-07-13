package nlp_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §V16 tier-1 (arXiv:2205.05131 Table 1): the standard mixture is exactly the seven
// (mode, μ, r) denoisers R(3,.15) R(8,.15) X(3,.5) X(8,.5) X(64,.15) X(64,.5) S(r=.25),
// mixed with equal weight.
func TestUL2MixtureConfig(t *testing.T) {
	want := []nlp.UL2Denoiser{
		{Mode: nlp.UL2R, Rate: 0.15, MeanSpan: 3, Weight: 1},
		{Mode: nlp.UL2R, Rate: 0.15, MeanSpan: 8, Weight: 1},
		{Mode: nlp.UL2X, Rate: 0.5, MeanSpan: 3, Weight: 1},
		{Mode: nlp.UL2X, Rate: 0.5, MeanSpan: 8, Weight: 1},
		{Mode: nlp.UL2X, Rate: 0.15, MeanSpan: 64, Weight: 1},
		{Mode: nlp.UL2X, Rate: 0.5, MeanSpan: 64, Weight: 1},
		{Mode: nlp.UL2S, Rate: 0.25, MeanSpan: 0, Weight: 1},
	}
	if got := nlp.UL2MixtureOfDenoisers(); !slices.Equal(got, want) {
		t.Errorf("mixture:\n got %+v\nwant %+v", got, want)
	}
}

// §V16 tier-1: each example is prefixed with its paradigm token (sentinelBase+mode); span
// sentinels live above the three reserved paradigm ids.
func TestUL2ParadigmToken(t *testing.T) {
	tokens := make([]int, 40)
	for i := range tokens {
		tokens[i] = i
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for _, d := range nlp.UL2MixtureOfDenoisers() {
		input, _ := nlp.UL2Denoise(tokens, d, sentBase, rng)
		if input[0] != sentBase+int(d.Mode) {
			t.Errorf("mode %d: leading token %d, want paradigm %d", d.Mode, input[0], sentBase+int(d.Mode))
		}
		for _, v := range input[1:] {
			if v >= sentBase && v < sentBase+3 {
				t.Errorf("mode %d: span stream reused a paradigm id %d", d.Mode, v)
			}
		}
	}
}

// §V16 tier-1 (S-denoiser = PrefixLM): a single trailing span whose length is ≈ r·n; the
// input is prefix+sentinel and the target is sentinel+suffix+final sentinel.
func TestUL2Sequential(t *testing.T) {
	tokens := make([]int, 20)
	for i := range tokens {
		tokens[i] = 100 + i
	}
	d := nlp.UL2Denoiser{Mode: nlp.UL2S, Rate: 0.25}
	input, target := nlp.UL2Denoise(tokens, d, sentBase, rng2())
	// input: [modeTok] + prefix + [sent0]; exactly one span sentinel in the input.
	spanBase := sentBase + 3
	sentsIn := 0
	for _, v := range input[1:] {
		if v >= spanBase {
			sentsIn++
		}
	}
	if sentsIn != 1 {
		t.Errorf("S-denoiser input has %d span sentinels, want 1", sentsIn)
	}
	// suffix length (target minus its 2 sentinels) ≈ round(0.25·20) = 5.
	suf := len(target) - 2
	if suf != 5 {
		t.Errorf("S-denoiser suffix length %d, want 5 (≈0.25·20)", suf)
	}
	// the suffix must be the true tail of the document.
	if !slices.Equal(target[1:1+suf], tokens[len(tokens)-suf:]) {
		t.Errorf("S-denoiser target suffix %v != document tail %v", target[1:1+suf], tokens[len(tokens)-suf:])
	}
}

func rng2() *rand.Rand { return rand.New(rand.NewPCG(7, 11)) }

// §V15 (round-trip): every denoiser in the mixture reconstructs the original document and
// reports the right mode.
func TestUL2RoundTrip(t *testing.T) {
	doc := make([]int, 50)
	for i := range doc {
		doc[i] = 200 + i
	}
	rng := rand.New(rand.NewPCG(3, 9))
	for _, d := range nlp.UL2MixtureOfDenoisers() {
		for trial := 0; trial < 20; trial++ {
			input, target := nlp.UL2Denoise(doc, d, sentBase, rng)
			got, mode, ok := nlp.UL2Reconstruct(input, target, sentBase)
			if !ok || mode != d.Mode || !slices.Equal(got, doc) {
				t.Fatalf("mode %d: reconstruct ok=%v mode=%d got=%v", d.Mode, ok, mode, got)
			}
		}
	}
}

// §V2 (edges): empty and single-token documents round-trip under every mode.
func TestUL2Edges(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 8))
	for _, mode := range []nlp.UL2Mode{nlp.UL2R, nlp.UL2S, nlp.UL2X} {
		for _, doc := range [][]int{{}, {7}, {1, 2, 3}} {
			d := nlp.UL2Denoiser{Mode: mode, Rate: 0.5, MeanSpan: 3}
			in, tg := nlp.UL2Denoise(doc, d, sentBase, rng)
			got, gotMode, ok := nlp.UL2Reconstruct(in, tg, sentBase)
			if !ok || gotMode != mode || !slices.Equal(got, doc) {
				t.Errorf("mode %d doc=%v: got %v mode=%d ok=%v", mode, doc, got, gotMode, ok)
			}
		}
	}
}

// §V2: a pair whose leading token is not a paradigm token is rejected.
func TestUL2RejectsMalformed(t *testing.T) {
	if _, _, ok := nlp.UL2Reconstruct([]int{5, 6}, []int{sentBase + 3}, sentBase); ok {
		t.Error("non-paradigm leading token should not reconstruct")
	}
	if _, _, ok := nlp.UL2Reconstruct(nil, nil, sentBase); ok {
		t.Error("empty input should not reconstruct")
	}
}

// §V16 tier-1 (mixing): sampling the mixture hits all seven denoisers with roughly equal
// frequency (uniform participation).
func TestUL2SampleUniform(t *testing.T) {
	mix := nlp.UL2MixtureOfDenoisers()
	rng := rand.New(rand.NewPCG(5, 6))
	counts := map[nlp.UL2Mode]int{}
	const N = 7000
	for i := 0; i < N; i++ {
		counts[nlp.SampleUL2Denoiser(mix, rng).Mode]++
	}
	// modes appear proportional to how many denoisers carry them: R×2, X×4, S×1 of 7.
	for mode, frac := range map[nlp.UL2Mode]float64{nlp.UL2R: 2.0 / 7, nlp.UL2X: 4.0 / 7, nlp.UL2S: 1.0 / 7} {
		if got := float64(counts[mode]) / N; math.Abs(got-frac) > 0.05 {
			t.Errorf("mode %d sampled frac %.3f, want ≈%.3f", mode, got, frac)
		}
	}
}

// §V15 fuzz: for any token stream, mode, rate, span and seed, UL2 denoising round-trips.
func FuzzUL2RoundTrip(f *testing.F) {
	f.Add([]byte("the quick brown fox"), uint64(1), 0, 15, 3)
	f.Add([]byte{}, uint64(0), 1, 25, 0)
	f.Add([]byte("x"), uint64(2), 2, 50, 64)
	f.Fuzz(func(t *testing.T, data []byte, seed uint64, modeSel, ratePct, span int) {
		mode := nlp.UL2Mode(((modeSel % 3) + 3) % 3)
		d := nlp.UL2Denoiser{
			Mode:     mode,
			Rate:     math.Min(1, math.Max(0.01, float64(((ratePct%100)+100)%100+1)/100)),
			MeanSpan: math.Max(1, float64(((span%64)+64)%64+1)),
		}
		tokens := make([]int, len(data)) // bytes 0..255, below sentBase
		for i, b := range data {
			tokens[i] = int(b)
		}
		rng := rand.New(rand.NewPCG(seed, 0x5ada))
		in, tg := nlp.UL2Denoise(tokens, d, sentBase, rng)
		got, gotMode, ok := nlp.UL2Reconstruct(in, tg, sentBase)
		if !ok {
			t.Fatalf("reconstruct failed for mode %d tokens %v", mode, tokens)
		}
		if gotMode != mode {
			t.Fatalf("recovered mode %d != %d", gotMode, mode)
		}
		if !slices.Equal(got, tokens) {
			t.Fatalf("round-trip mismatch (mode %d): %v != %v", mode, got, tokens)
		}
	})
}

// ExampleUL2Mode shows the three denoiser paradigms UL2 mixes.
func ExampleUL2Mode() {
	for _, m := range []nlp.UL2Mode{nlp.UL2R, nlp.UL2S, nlp.UL2X} {
		fmt.Print(int(m), " ")
	}
	fmt.Println()
	// Output:
	// 0 1 2
}

func ExampleUL2Denoise() {
	doc := []int{10, 11, 12, 13, 14, 15, 16, 17}
	// An S-denoiser (prefix-LM): the model sees a prefix and predicts the tail.
	d := nlp.UL2Denoiser{Mode: nlp.UL2S, Rate: 0.25}
	in, tg := nlp.UL2Denoise(doc, d, 10000, rand.New(rand.NewPCG(1, 1)))
	back, mode, ok := nlp.UL2Reconstruct(in, tg, 10000)
	fmt.Println(mode == nlp.UL2S, ok, slices.Equal(back, doc))
	// Output:
	// true true true
}
