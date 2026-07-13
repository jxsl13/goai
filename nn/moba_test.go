package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func mobaProbe(seed, seq, dm int) *tensor.Tensor {
	x := tensor.New(tensor.F64, tensor.Shape{seq, dm})
	for i := range x.Numel() {
		x.SetF64(math.Sin(float64(seed*100+i)*1.3)*0.7, tensor.Unravel(i, x.Shape())...)
	}
	return x
}

// §T557 collapse: topK ≥ #blocks selects every past block ⇒ MoBA IS full causal
// attention — compare against the reference OpMHA at 1e−12.
func TestMoBACollapsesToFullAttention(t *testing.T) {
	const seq, dm, heads, block = 16, 8, 2, 4
	q, k, v := mobaProbe(1, seq, dm), mobaProbe(2, seq, dm), mobaProbe(3, seq, dm)
	moba, err := nn.MoBAAttention(q, k, v, heads, block, seq/block, 0)
	if err != nil {
		t.Fatal(err)
	}
	full, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()),
		backend.OpMHA, []*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: heads, Causal: true})
	if err != nil {
		t.Fatal(err)
	}
	for i := range moba.Numel() {
		idx := tensor.Unravel(i, moba.Shape())
		if d := math.Abs(moba.AtF64(idx...) - full[0].AtF64(idx...)); d > 1e-12 {
			t.Fatalf("collapse broken at %v (Δ %g)", idx, d)
		}
	}
}

// §T557 isolation (the structural property block selection exists for): with
// topK=1 only the query's OWN block is attended — perturbing any EARLIER block's
// values must leave the output bit-identical.
func TestMoBATopOneIsBlockLocal(t *testing.T) {
	const seq, dm, heads, block = 16, 8, 2, 4
	q, k, v := mobaProbe(1, seq, dm), mobaProbe(2, seq, dm), mobaProbe(3, seq, dm)
	base, err := nn.MoBAAttention(q, k, v, heads, block, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	v2 := mobaProbe(3, seq, dm)
	for d := range dm {
		v2.SetF64(v2.AtF64(0, d)+5, 0, d) // mutate block 0's values
	}
	got, err := nn.MoBAAttention(q, k, v2, heads, block, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := block; i < seq; i++ { // queries outside block 0 must not see the change
		for d := range dm {
			if got.AtF64(i, d) != base.AtF64(i, d) {
				t.Fatalf("block-local violated: row %d changed after mutating block 0", i)
			}
		}
	}
}

// §T557 gating: plant a past block whose keys align overwhelmingly with the
// query — it must be selected (its values move the output), while a repellent
// block stays unselected (its values are invisible).
func TestMoBAGateSelectsAffineBlock(t *testing.T) {
	const seq, dm, heads, block = 16, 4, 1, 4
	q := tensor.New(tensor.F64, tensor.Shape{seq, dm})
	k := tensor.New(tensor.F64, tensor.Shape{seq, dm})
	v := mobaProbe(3, seq, dm)
	// query at row 15 points along +e0; block 1 keys align (+e0), block 0 keys repel (−e0).
	q.SetF64(4, 15, 0)
	for j := range block {
		k.SetF64(-3, j, 0)      // block 0: repellent
		k.SetF64(3, block+j, 0) // block 1: affine
	}
	run := func(vv *tensor.Tensor) *tensor.Tensor {
		out, err := nn.MoBAAttention(q, k, vv, heads, block, 2, 0) // own + 1 gated block
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	base := run(v)
	vAff := mobaProbe(3, seq, dm)
	for d := range dm {
		vAff.SetF64(vAff.AtF64(block, d)+7, block, d) // mutate the AFFINE block
	}
	changed := false
	for d := range dm {
		if run(vAff).AtF64(15, d) != base.AtF64(15, d) {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("the affine block was not selected — gating broken")
	}
	vRep := mobaProbe(3, seq, dm)
	for d := range dm {
		vRep.SetF64(vRep.AtF64(0, d)+7, 0, d) // mutate the REPELLENT block
	}
	for d := range dm {
		if run(vRep).AtF64(15, d) != base.AtF64(15, d) {
			t.Fatal("the repellent block leaked into the output — selection broken")
		}
	}
}

// MoBA: MoE-style routing over key blocks — full attention when topK covers all
// blocks, block-local at topK=1.
func ExampleMoBAAttention() {
	q := tensor.New(tensor.F64, tensor.Shape{8, 4})
	k := tensor.New(tensor.F64, tensor.Shape{8, 4})
	v := tensor.New(tensor.F64, tensor.Shape{8, 4})
	out, _ := nn.MoBAAttention(q, k, v, 1, 4, 2, 0)
	fmt.Println(out.Shape())
	// Output: (8, 4)
}
