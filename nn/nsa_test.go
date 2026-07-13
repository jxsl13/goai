package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T556 collapse 1: the SELECTION branch with topN ≥ #blocks is full causal
// attention (reference OpMHA, ≤1e−12).
func TestNSASelectionCollapsesToFull(t *testing.T) {
	const seq, dm, heads, block = 16, 8, 2, 4
	q, k, v := mobaProbe(1, seq, dm), mobaProbe(2, seq, dm), mobaProbe(3, seq, dm)
	_, slc, _, err := nn.NSABranches(q, k, v, heads, block, seq/block, 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	full, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()),
		backend.OpMHA, []*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: heads, Causal: true})
	if err != nil {
		t.Fatal(err)
	}
	for i := range slc.Numel() {
		idx := tensor.Unravel(i, slc.Shape())
		if d := math.Abs(slc.AtF64(idx...) - full[0].AtF64(idx...)); d > 1e-12 {
			t.Fatalf("selection collapse broken at %v (Δ %g)", idx, d)
		}
	}
}

// §T556 collapse 2: the WINDOW branch with window ≥ seq is full causal attention;
// the COMPRESSION branch with blockSize=1 is causal attention WITHOUT self
// (mean-pool of a single token is the token; only complete past blocks visible).
func TestNSAWindowAndCompressionCollapses(t *testing.T) {
	const seq, dm, heads = 12, 8, 2
	q, k, v := mobaProbe(4, seq, dm), mobaProbe(5, seq, dm), mobaProbe(6, seq, dm)
	cmp, _, win, err := nn.NSABranches(q, k, v, heads, 1, 1, seq, 0)
	if err != nil {
		t.Fatal(err)
	}
	full, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()),
		backend.OpMHA, []*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: heads, Causal: true})
	if err != nil {
		t.Fatal(err)
	}
	for i := range win.Numel() {
		idx := tensor.Unravel(i, win.Shape())
		if d := math.Abs(win.AtF64(idx...) - full[0].AtF64(idx...)); d > 1e-12 {
			t.Fatalf("window collapse broken at %v", idx)
		}
	}
	// cmp(blockSize=1) at row i == softmax over keys j<i (no self): check row 3, head 0.
	const dk = dm / heads
	for h := range heads {
		off := h * dk
		i := 3
		var m float64 = math.Inf(-1)
		sc := make([]float64, i)
		for j := range i {
			var s float64
			for d := range dk {
				s += q.AtF64(i, off+d) * k.AtF64(j, off+d)
			}
			s /= math.Sqrt(dk)
			sc[j] = s
			if s > m {
				m = s
			}
		}
		var sum float64
		for j := range i {
			sc[j] = math.Exp(sc[j] - m)
			sum += sc[j]
		}
		for d := range dk {
			var want float64
			for j := range i {
				want += sc[j] / sum * v.AtF64(j, off+d)
			}
			if diff := math.Abs(cmp.AtF64(i, off+d) - want); diff > 1e-12 {
				t.Fatalf("cmp(block=1) != causal-no-self at h%d d%d (Δ %g)", h, d, diff)
			}
		}
	}
}

// §T556 selection isolation: with topN=1 (own block only), earlier blocks'
// values are invisible to the selection branch.
func TestNSASelectionIsolation(t *testing.T) {
	const seq, dm, heads, block = 16, 8, 2, 4
	q, k, v := mobaProbe(7, seq, dm), mobaProbe(8, seq, dm), mobaProbe(9, seq, dm)
	_, base, _, err := nn.NSABranches(q, k, v, heads, block, 1, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	v2 := mobaProbe(9, seq, dm)
	for d := range dm {
		v2.SetF64(v2.AtF64(1, d)+9, 1, d) // mutate block 0
	}
	_, got, _, err := nn.NSABranches(q, k, v2, heads, block, 1, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := block; i < seq; i++ {
		for d := range dm {
			if got.AtF64(i, d) != base.AtF64(i, d) {
				t.Fatalf("selection isolation violated at row %d", i)
			}
		}
	}
}

// NSA: three branches — compressed overview, selected detail, local window —
// gated together by the caller.
func ExampleNSABranches() {
	q := tensor.New(tensor.F64, tensor.Shape{8, 4})
	k := tensor.New(tensor.F64, tensor.Shape{8, 4})
	v := tensor.New(tensor.F64, tensor.Shape{8, 4})
	cmp, slc, win, _ := nn.NSABranches(q, k, v, 1, 4, 2, 4, 0)
	fmt.Println(cmp.Shape(), slc.Shape(), win.Shape())
	// Output: (8, 4) (8, 4) (8, 4)
}
