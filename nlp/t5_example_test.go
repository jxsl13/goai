package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

// ExampleT5 loads a Hugging Face encoder checkpoint and turns token ids into
// contextual states suitable for a classifier or retrieval head.
func ExampleT5() {
	weights, _, err := safetensors.LoadFile("testdata/t5v10_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.T5FromHF(weights, nlp.T5Config{Heads: 2, HeadDim: 8})
	if err != nil {
		panic(err)
	}
	states, err := model.Forward(backend.NewContext(), []int{3, 7, 1})
	if err != nil {
		panic(err)
	}
	var firstBlock *nlp.T5Block = model.Blocks[0]
	fmt.Println(states.Shape(), len(model.Params()), firstBlock != nil)
	// Output: (3, 16) 19 true
}
