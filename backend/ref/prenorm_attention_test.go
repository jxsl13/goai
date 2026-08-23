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

func TestPreNormAttentionReferenceMatchesComposite(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dtype tensor.Dtype
		tol   float64
	}{
		{name: "f32", dtype: tensor.F32, tol: 3e-5},
		{name: "f64", dtype: tensor.F64, tol: 2e-12},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const batch, seq, dim, heads = 2, 3, 8, 2
			randTensor := func(shape tensor.Shape, seed uint64) *tensor.Tensor {
				if tc.dtype == tensor.F32 {
					return bench.RandF32(shape, seed)
				}
				return bench.RandF64(shape, seed)
			}
			x := randTensor(tensor.Shape{batch * seq, dim}, 1)
			gamma := randTensor(tensor.Shape{dim}, 2)
			beta := randTensor(tensor.Shape{dim}, 3)
			wq := randTensor(tensor.Shape{dim, dim}, 4)
			wk := randTensor(tensor.Shape{dim, dim}, 5)
			wv := randTensor(tensor.Shape{dim, dim}, 6)
			wo := randTensor(tensor.Shape{dim, dim}, 7)
			dOut := randTensor(tensor.Shape{batch * seq, dim}, 8)
			mha, err := nlp.NewMHA(heads, wq, wk, wv, wo)
			if err != nil {
				t.Fatal(err)
			}

			controlTape := autograd.NewTapeOn(backend.Reference())
			h, err := (&nn.LayerNorm{Gamma: gamma, Beta: beta, Eps: 1e-5}).Forward(controlTape.Context(), x)
			if err != nil {
				t.Fatal(err)
			}
			h, err = mha.ForwardBatched(controlTape.Context(), h, batch)
			if err != nil {
				t.Fatal(err)
			}
			control, err := backend.Execute(controlTape.Context(), backend.OpAdd, []*tensor.Tensor{x, h}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if err := controlTape.BackwardGrad(control[0], dOut); err != nil {
				t.Fatal(err)
			}

			fusedTape := autograd.NewTapeOn(backend.Reference())
			fused, err := backend.Execute(fusedTape.Context(), backend.OpPreNormAttention,
				[]*tensor.Tensor{x, gamma, beta, wq, wk, wv, wo},
				backend.PreNormAttentionAttrs{Heads: heads, Batch: batch, Eps: 1e-5})
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
			check("Y", fused[0], control[0])
			for _, pair := range []struct {
				name string
				in   *tensor.Tensor
			}{
				{"dX", x}, {"dGamma", gamma}, {"dBeta", beta}, {"dWq", wq},
				{"dWk", wk}, {"dWv", wv}, {"dWo", wo},
			} {
				check(pair.name, fusedTape.Grad(pair.in), controlTape.Grad(pair.in))
			}
		})
	}
}

func TestPreNormAttentionReferenceRejectsInvalidPackedBatch(t *testing.T) {
	x := bench.RandF64(tensor.Shape{5, 8}, 11)
	vector := bench.RandF64(tensor.Shape{8}, 12)
	weight := bench.RandF64(tensor.Shape{8, 8}, 13)
	_, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpPreNormAttention,
		[]*tensor.Tensor{x, vector, vector, weight, weight, weight, weight},
		backend.PreNormAttentionAttrs{Heads: 2, Batch: 2})
	if err == nil {
		t.Fatal("expected packed-batch validation error")
	}
}
