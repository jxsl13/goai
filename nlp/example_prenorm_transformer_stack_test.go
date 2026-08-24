package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// PreNormTransformerBlock groups one block's layers for
// ForwardPreNormTransformerStack. A stack may contain independently owned
// parameters while sharing its batch, head, and hidden geometry.
func ExamplePreNormTransformerBlock() {
	weight := tensor.Zeros(tensor.F32, tensor.Shape{2, 2})
	attention, err := nlp.NewMHA(1, weight, weight, weight, weight)
	if err != nil {
		panic(err)
	}
	block := nlp.PreNormTransformerBlock{
		Attention: attention,
		Norm1:     &nn.LayerNorm{},
		Norm2:     &nn.LayerNorm{},
		Up:        &nn.Linear{},
		Down:      &nn.Linear{},
	}
	fmt.Println(block.Attention.Heads, block.Norm1 != nil, block.Down != nil)
	// Output: 1 true true
}
