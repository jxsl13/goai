//go:build darwin && cgo

package metal_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func TestMetalMHABatchedCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
	ref, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext()
	for _, tc := range []struct {
		name                       string
		batch, seq, heads, kvh, dk int
		causals                    []bool
	}{
		{"mha", 3, 7, 4, 4, 4, []bool{false, true}},
		{"gqa", 2, 6, 4, 2, 4, []bool{false, true}},
		{"mqa", 2, 5, 4, 1, 4, []bool{false, true}},
		{"vit", 8, 65, 4, 4, 32, []bool{false}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dm, dkv := tc.heads*tc.dk, tc.kvh*tc.dk
			q := bench.RandF32(tensor.Shape{tc.batch * tc.seq, dm}, 31)
			k := bench.RandF32(tensor.Shape{tc.batch * tc.seq, dkv}, 32)
			v := bench.RandF32(tensor.Shape{tc.batch * tc.seq, dkv}, 33)
			g := bench.RandF32(tensor.Shape{tc.batch * tc.seq, dm}, 34)
			for _, causal := range tc.causals {
				attrs := backend.AttnAttrs{Heads: tc.heads, KVHeads: tc.kvh, Batch: tc.batch, Causal: causal}
				got, err := backend.Execute(ctx.WithBackend(mb), backend.OpMHA, []*tensor.Tensor{q, k, v}, attrs)
				if err != nil {
					t.Fatal(err)
				}
				want, err := backend.Execute(ctx.WithBackend(ref), backend.OpMHA, []*tensor.Tensor{q, k, v}, attrs)
				if err != nil {
					t.Fatal(err)
				}
				for i := range got[0].Numel() {
					idx := tensor.Unravel(i, got[0].Shape())
					a, e := got[0].AtF64(idx...), want[0].AtF64(idx...)
					den := math.Abs(e)
					if den < 1 {
						den = 1
					}
					if math.Abs(a-e) > crossTol(tc.seq+tc.dk)*den+1e-5 {
						t.Fatalf("causal=%v forward[%d]: metal=%g ref=%g", causal, i, a, e)
					}
				}

				gotGrad, err := backend.Execute(ctx.WithBackend(mb), backend.OpMHABackward, []*tensor.Tensor{q, k, v, g}, attrs)
				if err != nil {
					t.Fatal(err)
				}
				wantGrad, err := backend.Execute(ctx.WithBackend(ref), backend.OpMHABackward, []*tensor.Tensor{q, k, v, g}, attrs)
				if err != nil {
					t.Fatal(err)
				}
				for output := range gotGrad {
					for i := range gotGrad[output].Numel() {
						idx := tensor.Unravel(i, gotGrad[output].Shape())
						a, e := gotGrad[output].AtF64(idx...), wantGrad[output].AtF64(idx...)
						den := math.Abs(e)
						if den < 1 {
							den = 1
						}
						if math.Abs(a-e) > crossTol(3*(tc.seq+tc.dk))*den+3e-5 {
							t.Fatalf("causal=%v grad=%d[%d]: metal=%g ref=%g", causal, output, i, a, e)
						}
					}
				}
			}
		})
	}
}
