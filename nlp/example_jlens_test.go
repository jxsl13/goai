package nlp_test

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// exampleJLensModel builds a small deterministic Llama for the J-lens
// examples (seeded init → identical output on every run).
func exampleJLensModel() *nlp.Llama {
	model, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 16, Ctx: 8, Dim: 8, Heads: 2, Layers: 2, Hidden: 16, Eps: 1e-5,
	}, 5)
	if err != nil {
		panic(err)
	}
	return model
}

// ExampleFitJLens fits a Jacobian lens (§R250) on a token corpus, merges a
// disjoint-slice fit, and round-trips it through the native file format.
func ExampleFitJLens() {
	model := exampleJLensModel()
	// Llama (like GPT) exposes the two seams the lens machinery is typed
	// against: residual taps (ResidualModel) and the final-norm+head readout
	// (LensReadoutModel).
	var _ nlp.ResidualModel = model
	var _ nlp.LensReadoutModel = model
	corpus := [][]int{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 0}}

	// Options are the reference implementation's knobs: skip_first etc.
	var skip nlp.FitJLensOption = nlp.WithJLensSkipFirst(0)
	lens, err := nlp.FitJLens(model, corpus, skip)
	if err != nil {
		panic(err)
	}
	fmt.Println("layers:", lens.Layers, "dim:", lens.Dim, "weight:", lens.Weight)
	// At the last layer the transport is the identity by construction.
	fmt.Printf("J[last] diagonal: %.0f\n", lens.J[lens.Layers].AtF64(0, 0))

	// Disjoint half-corpus fits merge exactly into the full fit.
	a, _ := nlp.FitJLens(model, corpus[:2])
	b, _ := nlp.FitJLens(model, corpus[2:])
	if err := a.Merge(b, b.Weight); err != nil {
		panic(err)
	}
	fmt.Println("merged weight:", a.Weight)

	// Save → LoadJLens is bit-exact.
	path := filepath.Join(os.TempDir(), "example_jlens.safetensors")
	defer os.Remove(path)
	if err := lens.Save(path); err != nil {
		panic(err)
	}
	back, err := nlp.LoadJLens(path)
	if err != nil {
		panic(err)
	}
	fmt.Println("reloaded layers:", back.Layers)
	// Output:
	// layers: 2 dim: 8 weight: 4
	// J[last] diagonal: 1
	// merged weight: 4
	// reloaded layers: 2
}

// ExampleJLens_ApplyAt reads one position of a prompt out through the lens —
// the transported logit lens W_U·norm(J_l·h) — and inspects the ranked
// vocabulary, plus the raw Apply/Unembed seams it is built from.
func ExampleJLens_ApplyAt() {
	model := exampleJLensModel()
	lens, err := nlp.FitJLens(model, [][]int{{1, 2, 3, 4}, {5, 6, 7, 8}})
	if err != nil {
		panic(err)
	}
	tokens := []int{3, 1, 4, 1, 5}

	// Ranked readout of the layer-1 residual at the last position (-1,
	// Python-style indexing like the reference implementation).
	var readout *nlp.JLensReadout
	readout, err = lens.ApplyAt(model, backend.NewContext(), tokens, 1, -1)
	if err != nil {
		panic(err)
	}
	var top nlp.JLensToken = readout.Ranked[0]
	fmt.Println("rank of the top token:", readout.RankOf(top.Token))
	fmt.Println("vocab entries ranked:", len(readout.Ranked))

	// The pieces: capture a residual, transport+decode it with Apply, or
	// decode it untransported with the model's own Unembed.
	var h1 *tensor.Tensor
	var capture nlp.ResidualCapture = func(l int, h *tensor.Tensor) {
		if l == 1 {
			h1 = h
		}
	}
	if _, err := model.ForwardResiduals(backend.NewContext(), tokens, capture); err != nil {
		panic(err)
	}
	transported, err := lens.Apply(model, backend.NewContext(), h1, 1)
	if err != nil {
		panic(err)
	}
	plain, err := model.Unembed(backend.NewContext(), h1)
	if err != nil {
		panic(err)
	}
	fmt.Println("lens logits:", transported.Shape(), "plain logits:", plain.Shape())
	// Output:
	// rank of the top token: 0
	// vocab entries ranked: 16
	// lens logits: (5, 16) plain logits: (5, 16)
}

// ExampleJLens_Slice computes the layer × position grid behind the paper's
// slice visualisation: every fitted layer's top-1 readout plus the rank of
// the model's real next-token prediction under each cell.
func ExampleJLens_Slice() {
	model := exampleJLensModel()
	lens, err := nlp.FitJLens(model, [][]int{{1, 2, 3, 4}, {5, 6, 7, 8}})
	if err != nil {
		panic(err)
	}
	var grid *nlp.JLensSlice
	grid, err = lens.Slice(model, backend.NewContext(), []int{3, 1, 4, 1, 5})
	if err != nil {
		panic(err)
	}
	fmt.Println("rows:", len(grid.Rows), "positions:", len(grid.Output))
	var bottom nlp.JLensSliceRow = grid.Rows[len(grid.Rows)-1]
	fmt.Println("bottom row is the final layer:", bottom.Layer == lens.Layers)
	fmt.Println("bottom row agrees with the model:", bottom.Top[4] == grid.Output[4] && bottom.Rank[4] == 0)
	// Output:
	// rows: 3 positions: 5
	// bottom row is the final layer: true
	// bottom row agrees with the model: true
}

// ExampleJLensHTML renders the self-contained slice page (§T814) for the
// §T812 golden model — the tiny HF Llama with the ACTUAL reference-fitted
// lens artifact — on the reference probe prompt, with the §T813 J-space
// occupancy overlay. The page is a single file with zero external requests:
// grid, pin charts (inline SVG) and rank heatmap are all hand-rolled inline.
func ExampleJLensHTML() {
	ts, _, err := safetensors.LoadFile("testdata/jlens_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.LlamaFromHF(ts, nlp.LlamaConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 64,
	})
	if err != nil {
		panic(err)
	}
	// The Anthropic reference-implementation .pt lens artifact, directly.
	lens, err := nlp.JLensFromPT("testdata/jlens_ref.pt")
	if err != nil {
		panic(err)
	}
	raw, err := npy.LoadFile("testdata/jlens_probe.npy")
	if err != nil {
		panic(err)
	}
	shape := raw.Shape()
	probe := make([]int, shape[len(shape)-1])
	for i := range probe {
		if len(shape) == 2 {
			probe[i] = int(raw.AtF64(0, i))
		} else {
			probe[i] = int(raw.AtF64(i))
		}
	}

	// Generate-integrated tap: greedily extend the probe with the model's own
	// continuation and render the slice over prompt+continuation — the page's
	// bottom row then shows exactly the outputs that produced the new tokens.
	seq, err := model.Generate(probe, 4, nlp.Greedy())
	if err != nil {
		panic(err)
	}

	page, err := nlp.JLensHTML(model, lens, seq,
		nlp.WithJLensHTMLTitle("J-lens golden probe"),
		nlp.WithJLensHTMLOccupancy(8))
	if err != nil {
		panic(err)
	}
	dir, err := os.MkdirTemp("", "jlens-viz")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	if err := os.WriteFile(filepath.Join(dir, "slice.html"), []byte(page), 0o644); err != nil {
		panic(err)
	}

	// Stable summary (the page embeds only integers, greedy decode is
	// deterministic — byte-identical runs).
	fmt.Println("prompt:", len(probe), "generated:", len(seq)-len(probe))
	fmt.Println("layer rows:", strings.Count(page, `class="jlv-rowhead"`), "positions:", len(seq))
	fmt.Println("occupancy badges:", strings.Count(page, `class="jlv-occ"`) == 4*len(seq))
	external := strings.Contains(page, "http") || strings.Contains(page, "src=") || strings.Contains(page, "href=")
	fmt.Println("external refs:", external)
	// Output:
	// prompt: 12 generated: 4
	// layer rows: 4 positions: 16
	// occupancy badges: true
	// external refs: false
}
