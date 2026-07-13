package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func mkQKV(seed, rows, cols int) *tensor.Tensor {
	x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	for i := range x.Numel() {
		x.SetF64(math.Sin(float64(seed*1000+i))*0.4, tensor.Unravel(i, x.Shape())...)
	}
	return x
}

// §T507: the masked-attention primitive collapses onto OpMHA exactly — an explicit causal mask
// (−Inf above the diagonal, 0 elsewhere) must reproduce OpMHA's causal output bit-identically,
// and an all-zero mask must reproduce non-causal OpMHA.
func TestMHAMaskedCollapsesToMHA(t *testing.T) {
	const seq, dm, heads = 6, 8, 2
	q, k, v := mkQKV(1, seq, dm), mkQKV(2, seq, dm), mkQKV(3, seq, dm)
	ctx := backend.NewContext()
	attrs := backend.AttnAttrs{Heads: heads}
	causalAttrs := backend.AttnAttrs{Heads: heads, Causal: true}

	zero := tensor.New(tensor.F64, tensor.Shape{seq, seq})
	causalMask := tensor.New(tensor.F64, tensor.Shape{seq, seq})
	for i := range seq {
		for j := range seq {
			if j > i {
				causalMask.SetF64(math.Inf(-1), i, j)
			}
		}
	}

	check := func(mask *tensor.Tensor, refAttrs backend.AttnAttrs, name string) {
		got, err := backend.Execute(ctx, backend.OpMHAMasked, []*tensor.Tensor{q, k, v, mask}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		want, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{q, k, v}, refAttrs)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want[0].Numel() {
			idx := tensor.Unravel(i, want[0].Shape())
			if got[0].AtF64(idx...) != want[0].AtF64(idx...) {
				t.Fatalf("%s: diverges at %v", name, idx)
			}
		}
	}
	check(zero, attrs, "zero mask vs plain MHA")
	check(causalMask, causalAttrs, "explicit causal mask vs causal MHA")
}

// §T507: tree attention isolation — under a MedusaTreeMask, a node's output depends ONLY on its
// ancestors: mutating a SIBLING-branch token must leave the node's output bit-identical, while
// mutating an ANCESTOR must change it.
func TestMHAMaskedTreeIsolation(t *testing.T) {
	// tree: 0 → {1, 2}; 3 is child of 1; siblings: 2 vs (1,3).
	parents := []int{-1, 0, 0, 1}
	mask, _ := nlp.MedusaTreeMask(parents)
	const dm, heads = 8, 2
	k, v := mkQKV(2, 4, dm), mkQKV(3, 4, dm)
	q := mkQKV(1, 4, dm)
	ctx := backend.NewContext()
	attrs := backend.AttnAttrs{Heads: heads}

	run := func(kk, vv *tensor.Tensor) *tensor.Tensor {
		out, err := backend.Execute(ctx, backend.OpMHAMasked, []*tensor.Tensor{q, kk, vv, mask}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		return out[0]
	}
	base := run(k, v)

	mutate := func(row int) (*tensor.Tensor, *tensor.Tensor) {
		k2, v2 := mkQKV(2, 4, dm), mkQKV(3, 4, dm)
		for d := range dm {
			k2.SetF64(k2.AtF64(row, d)+1, row, d)
			v2.SetF64(v2.AtF64(row, d)+1, row, d)
		}
		return k2, v2
	}
	// mutate node 2 (sibling branch): node 3's output (ancestors 0,1,3) must be unchanged.
	sib := run(mutate(2))
	for d := range dm {
		if sib.AtF64(3, d) != base.AtF64(3, d) {
			t.Fatalf("sibling mutation leaked into node 3 at dim %d", d)
		}
	}
	// mutate node 1 (an ancestor of 3): node 3's output must change.
	anc := run(mutate(1))
	changed := false
	for d := range dm {
		if anc.AtF64(3, d) != base.AtF64(3, d) {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("ancestor mutation did not affect node 3 — the tree mask is not being honored")
	}
}
