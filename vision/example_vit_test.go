package vision_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// A tiny ViT classifies one 8×8 grayscale image: the image becomes 4 patch
// "words" plus a [class] token, runs through the encoder, and the head reads
// the class token. Params feeds any optimizer for training.
func ExampleViT() {
	m, err := vision.NewViT(1, 8, 3, 7, vision.WithViTDim(16), vision.WithViTHeads(2), vision.WithViTDepth(1))
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

// NewViT validates geometry: the patch size must divide the image size.
func ExampleNewViT() {
	_, err := vision.NewViT(1, 8, 3, 7, vision.WithViTPatch(3))
	fmt.Println(err != nil)
	// Output: true
}
