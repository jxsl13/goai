package vision_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// The default constructor uses the conventional AdamW beta and epsilon values.
func ExampleViTAdamWConfig() {
	config := vision.DefaultViTAdamWConfig(1e-3, 0.1)
	fmt.Println(config.Beta1, config.Beta2, config.Eps)
	// Output: 0.9 0.999 1e-08
}

// A session owns repeated objective-plus-AdamW steps and materializes model
// parameters only when Sync or Close is called.
func ExampleViTAdamWSession() {
	model, err := vision.NewViT(1, 4, 2, 1,
		vision.WithViTPatch(2), vision.WithViTDim(4), vision.WithViTDepth(1),
		vision.WithViTHeads(1), vision.WithViTMLP(8))
	if err != nil {
		panic(err)
	}
	session, err := model.NewAdamWSession(
		backend.NewContext(), 1, vision.DefaultViTAdamWConfig(1e-3, 0.1))
	if err != nil {
		panic(err)
	}
	var _ *vision.ViTAdamWSession = session
	images := tensor.New(tensor.F32, tensor.Shape{1, 1, 4, 4})
	targets := tensor.FromFloat32(tensor.Shape{1}, []float32{1})
	loss, err := session.Step(images, targets)
	if err != nil {
		panic(err)
	}
	if err := session.Sync(); err != nil {
		panic(err)
	}
	if err := session.Close(); err != nil {
		panic(err)
	}
	fmt.Println(loss.Numel())
	// Output: 1
}
