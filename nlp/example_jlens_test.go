package nlp_test

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
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
