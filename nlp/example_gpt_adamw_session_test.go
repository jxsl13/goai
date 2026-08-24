package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// A session owns repeated objective-plus-AdamW steps and materializes model
// parameters only when Sync or Close is called.
func ExampleGPTAdamWSession() {
	cfg := nlp.GPTConfig{Vocab: 4, Ctx: 2, Dim: 2, Heads: 1, Layers: 1, Eps: 1e-5}
	model, err := nlp.FromSafetensors(cfg, exampleGPTParameters(cfg))
	if err != nil {
		panic(err)
	}
	config := nlp.GPTAdamWConfig{
		LR: 1e-3, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: 0.1,
	}
	session, err := model.NewAdamWSession(backend.NewContext(), 2, config)
	if err != nil {
		panic(err)
	}
	var _ *nlp.GPTAdamWSession = session
	targets := tensor.FromFloat32(tensor.Shape{2}, []float32{1, 2})
	loss, err := session.Step([]int{0, 1}, targets)
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
