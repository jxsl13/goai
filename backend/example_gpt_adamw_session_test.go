package backend_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type exampleGPTAdamWSession struct{}

func (exampleGPTAdamWSession) Step(*tensor.Tensor, *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.FromFloat32(tensor.Shape{}, []float32{0}), nil
}

func (exampleGPTAdamWSession) Sync([]*tensor.Tensor) error { return nil }
func (exampleGPTAdamWSession) Close() error                { return nil }

// Backends expose resident GPT optimizer state through one small lifecycle.
func ExampleGPTAdamWSession() {
	var session backend.GPTAdamWSession = exampleGPTAdamWSession{}
	loss, _ := session.Step(nil, nil)
	_ = session.Sync(nil)
	_ = session.Close()
	fmt.Println(loss.Numel())
	// Output: 1
}
