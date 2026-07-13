package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T558 collapse: topK ≥ seq selects everything ⇒ DSA ≡ full causal attention
// (reference OpMHA, ≤1e−12) regardless of the indexer contents.
func TestDSACollapsesToFullAttention(t *testing.T) {
	const seq, dm, heads = 12, 8, 2
	q, k, v := mobaProbe(1, seq, dm), mobaProbe(2, seq, dm), mobaProbe(3, seq, dm)
	qi, ki := mobaProbe(4, seq, 4), mobaProbe(5, seq, 4)
	got, err := nn.DSAAttention(q, k, v, qi, ki, []float64{1, 0.5}, heads, seq, 0)
	if err != nil {
		t.Fatal(err)
	}
	full, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()),
		backend.OpMHA, []*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: heads, Causal: true})
	if err != nil {
		t.Fatal(err)
	}
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		if d := math.Abs(got.AtF64(idx...) - full[0].AtF64(idx...)); d > 1e-12 {
			t.Fatalf("collapse broken at %v (Δ %g)", idx, d)
		}
	}
}

// §T558 indexer routing: a token whose indexer key aligns with the query MUST be
// selected (its value moves the output); a repellent token stays invisible.
func TestDSAIndexerRoutes(t *testing.T) {
	const seq, dm, heads, idxDim = 8, 4, 1, 4
	q, k := mobaProbe(6, seq, dm), mobaProbe(7, seq, dm)
	v := mobaProbe(8, seq, dm)
	qi := tensor.New(tensor.F64, tensor.Shape{seq, idxDim})
	ki := tensor.New(tensor.F64, tensor.Shape{seq, idxDim})
	qi.SetF64(3, 7, 0)  // query 7 points along +e0 in indexer space
	ki.SetF64(4, 2, 0)  // token 2: affine
	ki.SetF64(-4, 4, 0) // token 4: repellent (ReLU zeroes it)
	run := func(vv *tensor.Tensor) *tensor.Tensor {
		out, err := nn.DSAAttention(q, k, vv, qi, ki, []float64{1}, heads, 2, 0) // self + 1
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	base := run(v)
	vAff := mobaProbe(8, seq, dm)
	for d := range dm {
		vAff.SetF64(vAff.AtF64(2, d)+7, 2, d)
	}
	changed := false
	for d := range dm {
		if run(vAff).AtF64(7, d) != base.AtF64(7, d) {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("affine token not selected — indexer routing broken")
	}
	vRep := mobaProbe(8, seq, dm)
	for d := range dm {
		vRep.SetF64(vRep.AtF64(4, d)+7, 4, d)
	}
	for d := range dm {
		if run(vRep).AtF64(7, d) != base.AtF64(7, d) {
			t.Fatal("repellent token leaked into the output")
		}
	}
}

// DSA: the lightning indexer ranks tokens, attention runs over the top-k —
// O(n·k) sparse attention with query-adaptive token granularity.
func ExampleDSAAttention() {
	q := tensor.New(tensor.F64, tensor.Shape{6, 4})
	k := tensor.New(tensor.F64, tensor.Shape{6, 4})
	v := tensor.New(tensor.F64, tensor.Shape{6, 4})
	qi := tensor.New(tensor.F64, tensor.Shape{6, 2})
	ki := tensor.New(tensor.F64, tensor.Shape{6, 2})
	out, _ := nn.DSAAttention(q, k, v, qi, ki, []float64{1}, 1, 3, 0)
	fmt.Println(out.Shape())
	// Output: (6, 4)
}
