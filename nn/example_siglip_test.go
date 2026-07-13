package nn_test

import (
	"fmt"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// SigLIPHead carries the loss's two learned scalars; Temp reports the applied
// temperature (paper init: t=10, b=−10). One loss call scores a batch of
// image/text embedding pairs with independent sigmoids.
func ExampleSigLIPHead() {
	head := nn.NewSigLIPHead(tensor.F64)
	fmt.Printf("t=%.0f params=%d\n", head.Temp(), len(head.Params()))

	img := tensor.Randn(tensor.F64, 1, tensor.Shape{4, 8})
	txt := tensor.Randn(tensor.F64, 2, tensor.Shape{4, 8})
	tape := autograd.NewTape()
	loss, err := nn.SigLIPLoss(tape.Context(), img, txt, head)
	if err != nil {
		panic(err)
	}
	fmt.Println(loss.AtF64() > 0, tape.Backward(loss) == nil)
	// Output:
	// t=10 params=2
	// true true
}
