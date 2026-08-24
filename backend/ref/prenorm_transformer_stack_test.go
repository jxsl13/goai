package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func TestPreNormTransformerStackReferenceMatchesCompleteBlockLoop(t *testing.T) {
	const depth, batch, seq, dim, heads, hidden = 2, 2, 3, 8, 2, 12
	x := bench.RandF32(tensor.Shape{batch * seq, dim}, 101)
	dOut := bench.RandF32(tensor.Shape{batch * seq, dim}, 102)
	inputs := make([]*tensor.Tensor, 1, 1+12*depth)
	inputs[0] = x
	blocks := make([]nlp.PreNormTransformerBlock, depth)
	seed := uint64(103)
	for block := range depth {
		rand := func(shape tensor.Shape) *tensor.Tensor {
			value := bench.RandF32(shape, seed)
			seed++
			return value
		}
		gamma1, beta1 := rand(tensor.Shape{dim}), rand(tensor.Shape{dim})
		wq, wk := rand(tensor.Shape{dim, dim}), rand(tensor.Shape{dim, dim})
		wv, wo := rand(tensor.Shape{dim, dim}), rand(tensor.Shape{dim, dim})
		gamma2, beta2 := rand(tensor.Shape{dim}), rand(tensor.Shape{dim})
		w1, b1 := rand(tensor.Shape{dim, hidden}), rand(tensor.Shape{hidden})
		w2, b2 := rand(tensor.Shape{hidden, dim}), rand(tensor.Shape{dim})
		mha, err := nlp.NewMHA(heads, wq, wk, wv, wo)
		if err != nil {
			t.Fatal(err)
		}
		blocks[block] = nlp.PreNormTransformerBlock{
			Attention: mha,
			Norm1:     &nn.LayerNorm{Gamma: gamma1, Beta: beta1, Eps: 2e-5},
			Norm2:     &nn.LayerNorm{Gamma: gamma2, Beta: beta2, Eps: 3e-5},
			Up:        &nn.Linear{W: w1, B: b1},
			Down:      &nn.Linear{W: w2, B: b2},
		}
		inputs = append(inputs, gamma1, beta1, wq, wk, wv, wo, gamma2, beta2, w1, b1, w2, b2)
	}

	controlTape := autograd.NewTapeOn(backend.Reference())
	control := x
	var err error
	for _, block := range blocks {
		control, err = nlp.ForwardPreNormTransformerBlock(controlTape.Context(), control,
			block.Attention, block.Norm1, block.Norm2, block.Up, block.Down, batch)
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := controlTape.BackwardGrad(control, dOut); err != nil {
		t.Fatal(err)
	}

	fusedTape := autograd.NewTapeOn(backend.Reference())
	fused, err := backend.Execute(fusedTape.Context(), backend.OpPreNormTransformerStack, inputs,
		backend.PreNormTransformerStackAttrs{Depth: depth, Heads: heads, Batch: batch, Eps1: 2e-5, Eps2: 3e-5})
	if err != nil {
		t.Fatal(err)
	}
	if err := fusedTape.BackwardGrad(fused[0], dOut); err != nil {
		t.Fatal(err)
	}

	closeTensor := func(name string, got, want *tensor.Tensor) {
		t.Helper()
		for i := range got.Numel() {
			g, w := float64(got.Storage().F32()[i]), float64(want.Storage().F32()[i])
			if diff := math.Abs(g - w); diff > 8e-5*math.Max(1, math.Abs(w)) {
				t.Fatalf("%s[%d]: stack=%g loop=%g diff=%g", name, i, g, w, diff)
			}
		}
	}
	closeTensor("Y", fused[0], control)
	for i, input := range inputs {
		got, want := fusedTape.Grad(input), controlTape.Grad(input)
		if got == nil || want == nil {
			t.Fatalf("gradient %d is nil", i)
		}
		closeTensor("gradient", got, want)
	}
}
