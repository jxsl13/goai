package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func selCausal(seq int) *tensor.Tensor {
	m := tensor.New(tensor.F64, tensor.Shape{seq, seq})
	for i := range seq {
		for j := range seq {
			if j > i {
				m.SetF64(math.Inf(-1), i, j)
			}
		}
	}
	return m
}

func bitEqual(t *testing.T, got, want *tensor.Tensor, name string) {
	t.Helper()
	for i := range want.Numel() {
		idx := tensor.Unravel(i, want.Shape())
		if got.AtF64(idx...) != want.AtF64(idx...) {
			t.Fatalf("%s: diverges at %v: %g vs %g", name, idx, got.AtF64(idx...), want.AtF64(idx...))
		}
	}
}

// §T512: with an all-source-1 selector the second source must be ignored entirely —
// mha_select collapses onto mha_masked (same exclusion pattern) and onto causal OpMHA,
// bit-identically, regardless of what garbage sits in Q2/K2.
func TestMHASelectCollapsesToMasked(t *testing.T) {
	const seq, dm, heads = 6, 8, 2
	q1, k1, v := mkQKV(1, seq, dm), mkQKV(2, seq, dm), mkQKV(3, seq, dm)
	q2, k2 := mkQKV(7, seq, dm), mkQKV(8, seq, dm) // garbage: must not matter
	ctx := backend.NewContext()
	attrs := backend.AttnAttrs{Heads: heads}
	sel := selCausal(seq)

	got, err := backend.Execute(ctx, backend.OpMHASelect, []*tensor.Tensor{q1, k1, q2, k2, v, sel}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	masked, err := backend.Execute(ctx, backend.OpMHAMasked, []*tensor.Tensor{q1, k1, v, sel}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	causal, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{q1, k1, v},
		backend.AttnAttrs{Heads: heads, Causal: true})
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, got[0], masked[0], "select(all source 1) vs masked")
	bitEqual(t, got[0], causal[0], "select(all source 1) vs causal MHA")
}

// §T512: selecting between IDENTICAL sources is a no-op — any 0/1 pattern over
// Q1=Q2, K1=K2 must reproduce mha_masked with the same exclusions bit-identically.
func TestMHASelectSourceInvariance(t *testing.T) {
	const seq, dm, heads = 6, 8, 2
	q, k, v := mkQKV(1, seq, dm), mkQKV(2, seq, dm), mkQKV(3, seq, dm)
	ctx := backend.NewContext()
	attrs := backend.AttnAttrs{Heads: heads}
	sel := selCausal(seq)
	for i := range seq {
		for j := 0; j <= i; j++ {
			if (i+j)%2 == 0 { // checkerboard: half the pairs take source 2
				sel.SetF64(1, i, j)
			}
		}
	}
	got, err := backend.Execute(ctx, backend.OpMHASelect, []*tensor.Tensor{q, k, q, k, v, sel}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	masked, err := backend.Execute(ctx, backend.OpMHAMasked, []*tensor.Tensor{q, k, v, selCausal(seq)}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, got[0], masked[0], "identical sources, checkerboard selector")
}

// §T512: source symmetry — swapping the sources AND inverting the selector (0↔1,
// exclusions kept) must give the bit-identical output. This exercises genuine
// per-pair mixing of two DIFFERENT score sources.
func TestMHASelectSwapSymmetry(t *testing.T) {
	const seq, dm, heads = 6, 8, 2
	q1, k1 := mkQKV(1, seq, dm), mkQKV(2, seq, dm)
	q2, k2 := mkQKV(4, seq, dm), mkQKV(5, seq, dm)
	v := mkQKV(3, seq, dm)
	ctx := backend.NewContext()
	attrs := backend.AttnAttrs{Heads: heads}
	sel := selCausal(seq)
	inv := selCausal(seq)
	for i := range seq {
		for j := 0; j <= i; j++ {
			if (i*3+j)%2 == 0 {
				sel.SetF64(1, i, j)
			} else {
				inv.SetF64(1, i, j)
			}
		}
	}
	a, err := backend.Execute(ctx, backend.OpMHASelect, []*tensor.Tensor{q1, k1, q2, k2, v, sel}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	b, err := backend.Execute(ctx, backend.OpMHASelect, []*tensor.Tensor{q2, k2, q1, k1, v, inv}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, a[0], b[0], "source swap + selector inversion")
}
