package nlp_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// jlensDemoCorpus is a small deterministic token corpus for fitting a J-lens in
// tests — short sequences within the model's vocab, no randomness.
func jlensDemoCorpus() [][]int {
	return [][]int{
		{1, 5, 9, 3, 7, 2, 8, 4},
		{2, 6, 0, 4, 8, 1, 5, 9},
		{3, 7, 1, 9, 4, 2, 6, 0},
		{4, 8, 2, 6, 0, 5, 1, 7},
	}
}

// TestJLensOnGGUFLoadedModel proves the "download a model, see what it holds"
// pipeline: a Llama round-tripped through GGUF is a normal *Llama, so it
// satisfies the J-lens ResidualModel / LensReadoutModel interfaces directly. A
// J-lens fitted on the GGUF-loaded model must match the fit on the original to
// within the F32 floor of the GGUF weight storage (LlamaToGGUF/gguf.Write store
// F32), confirming the load preserves everything the lens depends on.
func TestJLensOnGGUFLoadedModel(t *testing.T) {
	orig, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 32, Dim: 32, Heads: 4, KVHeads: 2, Layers: 3, Hidden: 64,
		Eps: 1e-5, RopeBase: 10000,
	}, 11)
	if err != nil {
		t.Fatal(err)
	}
	meta, ts := nlp.LlamaToGGUF(orig)
	f := ggufByteTrip(t, meta, ts)
	gm, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("LlamaFromGGUF: %v", err)
	}

	corpus := jlensDemoCorpus()
	lensOrig, err := nlp.FitJLens(orig, corpus)
	if err != nil {
		t.Fatalf("FitJLens(orig): %v", err)
	}
	lensGGUF, err := nlp.FitJLens(gm, corpus)
	if err != nil {
		t.Fatalf("FitJLens(gguf): %v", err)
	}

	// The two fits agree to the F32 weight-storage floor — the GGUF-loaded model
	// is J-lens-ready and behaves like the original.
	if len(lensOrig.J) != len(lensGGUF.J) {
		t.Fatalf("layer count differs: %d vs %d", len(lensOrig.J), len(lensGGUF.J))
	}
	var maxAbs float64
	for l := range lensOrig.J {
		a, b := lensOrig.J[l], lensGGUF.J[l]
		if a == nil || b == nil {
			continue
		}
		for i := range a.Shape()[0] {
			for j := range a.Shape()[1] {
				if d := a.AtF64(i, j) - b.AtF64(i, j); d > maxAbs {
					maxAbs = d
				} else if -d > maxAbs {
					maxAbs = -d
				}
			}
		}
	}
	t.Logf("J-lens fit: original vs GGUF-loaded max abs diff %.3e (F32 weight floor)", maxAbs)
	if maxAbs > 2e-3 {
		t.Errorf("J-lens fits diverge across the GGUF round-trip: %.3e > 2e-3", maxAbs)
	}

	// The visualization renders on the GGUF-loaded model and is self-contained.
	tokens := []int{3, 7, 1, 9, 4, 2}
	page, err := nlp.JLensHTML(gm, lensGGUF, tokens)
	if err != nil {
		t.Fatalf("JLensHTML on GGUF-loaded model: %v", err)
	}
	if !strings.Contains(page, "<table") && !strings.Contains(page, "grid") {
		t.Error("rendered page has no grid structure")
	}
	for _, ext := range []string{"http://", "https://", "src=", "cdn"} {
		if strings.Contains(page, ext) {
			t.Errorf("rendered page contains an external reference %q", ext)
		}
	}
}

// ExampleLlamaFromGGUF_jlens shows the end-to-end interpretability pipeline on a
// GGUF-loaded model: load, fit a J-lens, and render the layer-by-position view
// of what each layer holds at each position.
func ExampleLlamaFromGGUF_jlens() {
	// A stand-in for gguf.ReadFile("model.gguf") + nlp.LlamaFromGGUF — here we
	// build a model and round-trip it so the example is self-contained.
	base, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 32, Dim: 32, Heads: 4, KVHeads: 2, Layers: 3, Hidden: 64,
		Eps: 1e-5, RopeBase: 10000,
	}, 11)
	meta, ts := nlp.LlamaToGGUF(base)
	model, _ := nlp.LlamaFromGGUF(meta, ts)

	lens, _ := nlp.FitJLens(model, [][]int{{1, 5, 9, 3, 7, 2, 8, 4}, {2, 6, 0, 4, 8, 1, 5, 9}})
	page, _ := nlp.JLensHTML(model, lens, []int{3, 7, 1, 9})

	fmt.Printf("lens layers: %d, self-contained page: %v\n",
		len(lens.J), len(page) > 0 && !strings.Contains(page, "http"))
	// Output: lens layers: 4, self-contained page: true
}
