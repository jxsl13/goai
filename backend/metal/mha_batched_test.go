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
	const batch, seq, dm, heads = 3, 7, 16, 4
	q := bench.RandF32(tensor.Shape{batch * seq, dm}, 31)
	k := bench.RandF32(tensor.Shape{batch * seq, dm}, 32)
	v := bench.RandF32(tensor.Shape{batch * seq, dm}, 33)
	g := bench.RandF32(tensor.Shape{batch * seq, dm}, 34)
	ctx := backend.NewContext()
	for _, causal := range []bool{false, true} {
		attrs := backend.AttnAttrs{Heads: heads, Batch: batch, Causal: causal}
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
			if math.Abs(a-e) > crossTol(seq+dm/heads)*den+1e-5 {
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
				if math.Abs(a-e) > crossTol(3*(seq+dm/heads))*den+3e-5 {
					t.Fatalf("causal=%v grad=%d[%d]: metal=%g ref=%g", causal, output, i, a, e)
				}
			}
		}
	}
}
