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

func TestPreNormTransformerBlockReferenceMatchesTwoBoundaryComposite(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dtype tensor.Dtype
		tol   float64
	}{
		{name: "f32", dtype: tensor.F32, tol: 8e-5},
		{name: "f64", dtype: tensor.F64, tol: 3e-11},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const batch, seq, dim, heads, hidden = 2, 3, 8, 2, 12
			randTensor := func(shape tensor.Shape, seed uint64) *tensor.Tensor {
				if tc.dtype == tensor.F32 {
					return bench.RandF32(shape, seed)
				}
				return bench.RandF64(shape, seed)
			}
			x := randTensor(tensor.Shape{batch * seq, dim}, 1)
			gamma1 := randTensor(tensor.Shape{dim}, 2)
			beta1 := randTensor(tensor.Shape{dim}, 3)
			wq := randTensor(tensor.Shape{dim, dim}, 4)
			wk := randTensor(tensor.Shape{dim, dim}, 5)
			wv := randTensor(tensor.Shape{dim, dim}, 6)
			wo := randTensor(tensor.Shape{dim, dim}, 7)
			gamma2 := randTensor(tensor.Shape{dim}, 8)
			beta2 := randTensor(tensor.Shape{dim}, 9)
			w1 := randTensor(tensor.Shape{dim, hidden}, 10)
			b1 := randTensor(tensor.Shape{hidden}, 11)
			w2 := randTensor(tensor.Shape{hidden, dim}, 12)
			b2 := randTensor(tensor.Shape{dim}, 13)
			dOut := randTensor(tensor.Shape{batch * seq, dim}, 14)
			mha, err := nlp.NewMHA(heads, wq, wk, wv, wo)
			if err != nil {
				t.Fatal(err)
			}
			norm1 := &nn.LayerNorm{Gamma: gamma1, Beta: beta1, Eps: 2e-5}
			norm2 := &nn.LayerNorm{Gamma: gamma2, Beta: beta2, Eps: 3e-5}
			up := &nn.Linear{W: w1, B: b1}
			down := &nn.Linear{W: w2, B: b2}

			controlTape := autograd.NewTapeOn(backend.Reference())
			control, err := mha.ForwardPreNorm(controlTape.Context(), x, norm1, batch)
			if err != nil {
				t.Fatal(err)
			}
			control, err = nn.ForwardPreNormFFN(controlTape.Context(), control, norm2, up, down)
			if err != nil {
				t.Fatal(err)
			}
			if err := controlTape.BackwardGrad(control, dOut); err != nil {
				t.Fatal(err)
			}

			inputs := []*tensor.Tensor{x, gamma1, beta1, wq, wk, wv, wo, gamma2, beta2, w1, b1, w2, b2}
			fusedTape := autograd.NewTapeOn(backend.Reference())
			fused, err := backend.Execute(fusedTape.Context(), backend.OpPreNormTransformerBlock, inputs,
				backend.PreNormTransformerBlockAttrs{Heads: heads, Batch: batch, Eps1: norm1.Eps, Eps2: norm2.Eps})
			if err != nil {
				t.Fatal(err)
			}
			if err := fusedTape.BackwardGrad(fused[0], dOut); err != nil {
				t.Fatal(err)
			}

			check := func(name string, got, want *tensor.Tensor) {
				t.Helper()
				got = got.Contiguous()
				want = want.Contiguous()
				for i := range got.Numel() {
					var g, w float64
					if tc.dtype == tensor.F32 {
						g, w = float64(got.Storage().F32()[i]), float64(want.Storage().F32()[i])
					} else {
						g, w = got.Storage().F64()[i], want.Storage().F64()[i]
					}
					if diff := math.Abs(g - w); diff > tc.tol*math.Max(1, math.Abs(w)) {
						t.Fatalf("%s[%d]: fused=%g composite=%g diff=%g", name, i, g, w, diff)
					}
				}
			}
			check("Y", fused[0], control)
			for i, input := range inputs {
				got, want := fusedTape.Grad(input), controlTape.Grad(input)
				if got == nil || want == nil {
					t.Fatalf("gradient %d is nil", i)
				}
				check("gradient", got, want)
			}
		})
	}
}

func TestPreNormTransformerBlockReferenceRejectsInvalidPackedBatch(t *testing.T) {
	x := bench.RandF32(tensor.Shape{5, 8}, 21)
	vector := bench.RandF32(tensor.Shape{8}, 22)
	weight := bench.RandF32(tensor.Shape{8, 8}, 23)
	hiddenWeight := bench.RandF32(tensor.Shape{8, 12}, 24)
	hidden := bench.RandF32(tensor.Shape{12}, 25)
	downWeight := bench.RandF32(tensor.Shape{12, 8}, 26)
	_, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpPreNormTransformerBlock,
		[]*tensor.Tensor{x, vector, vector, weight, weight, weight, weight, vector, vector, hiddenWeight, hidden, downWeight, vector},
		backend.PreNormTransformerBlockAttrs{Heads: 2, Batch: 2})
	if err == nil {
		t.Fatal("expected packed-batch validation error")
	}
}
