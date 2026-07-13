package vision_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// The projector turns ViT patch features into soft tokens of the decoder's
// width — Forward maps [tokens, vitDim] to [tokens, gptDim], Params feeds the
// optimizer. Features exposes the encoder tokens the projector consumes.
func ExampleVLMProjector() {
	vit, err := vision.NewViT(1, 8, 3, 7,
		vision.WithViTDim(16), vision.WithViTHeads(2), vision.WithViTDepth(1))
	if err != nil {
		panic(err)
	}
	img := tensor.New(tensor.F32, tensor.Shape{1, 8, 8})
	feats, err := vit.Features(backend.NewContext(), img) // [1+4 tokens, 16]
	if err != nil {
		panic(err)
	}
	proj, err := vision.NewVLMProjector(tensor.F32, 16, 24, 1)
	if err != nil {
		panic(err)
	}
	soft, err := proj.Forward(backend.NewContext(), feats)
	if err != nil {
		panic(err)
	}
	fmt.Println(feats.Shape(), soft.Shape(), len(proj.Params()))
	// Output: (5, 16) (5, 24) 4
}
