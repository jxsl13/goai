package vision_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// A tiny Swin Transformer classifies one 8×8 grayscale image: the image becomes a
// 4×4 token grid, two stages of windowed (W-MSA/SW-MSA) blocks with a patch merge
// between them build a feature pyramid, and a global-average-pooled head scores
// each class. Params feeds any optimizer for training.
func ExampleSwinTransformer() {
	m, err := vision.NewSwin(tensor.F32, 8, 2, 2, 8, []int{2, 2}, []int{2, 4}, 3, 7,
		vision.WithSwinChannels(1))
	if err != nil {
		panic(err)
	}
	img := tensor.New(tensor.F32, tensor.Shape{1, 8, 8}) // zeros: just the shapes
	logits, err := m.Forward(backend.NewContext(), img)
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape(), len(m.Params()) > 0)
	// Output: (1, 3) true
}

// A single Swin block runs windowed attention over an H×W token grid in place
// (same shape in and out); here a 4×4 grid of width 8. Blocks are the reusable
// unit stages are built from.
func ExampleSwinBlock() {
	m, err := vision.NewSwin(tensor.F32, 8, 2, 2, 8, []int{2}, []int{2}, 3, 7,
		vision.WithSwinChannels(1))
	if err != nil {
		panic(err)
	}
	blk := m.Stages[0][0] // the first stage's first (W-MSA) block
	grid := tensor.New(tensor.F32, tensor.Shape{4 * 4, 8})
	out, err := blk.Forward(backend.NewContext(), grid, 4, 4)
	if err != nil {
		panic(err)
	}
	fmt.Println(out.Shape())
	// Output: (16, 8)
}

// NewSwin validates geometry: the patch size must divide the image size.
func ExampleNewSwin() {
	_, err := vision.NewSwin(tensor.F32, 8, 3, 2, 8, []int{1}, []int{1}, 3, 7)
	fmt.Println(err != nil)
	// Output: true
}
