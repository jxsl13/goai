package tensor

import (
	"math"
	"math/rand"
	"testing"
)

// TestCoalesceDimsPreservesTraversal checks the property the whole optimization rests on: folding
// dense axis pairs re-describes the SAME offset sequence. It compares the coalesced walk against a
// per-element logical-index oracle, which is independent of the helper being tested — comparing
// against another implementation of the same idea would prove only that the idea is self-consistent.
//
// The sweep deliberately includes shapes where NOTHING may merge (transposes, permuted views,
// strides with gaps): a helper that merged those would be silently wrong, and every value would
// still look plausible.
func TestCoalesceDimsPreservesTraversal(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for range 400 {
		nd := 1 + rng.Intn(4)
		shape := make(Shape, nd)
		for i := range shape {
			shape[i] = 1 + rng.Intn(4)
		}
		x := New(F64, shape)
		s := x.Storage().F64()
		for i := range s {
			s[i] = rng.NormFloat64()
		}
		v := x
		// Randomly permute and slice so both mergeable and unmergeable layouts appear.
		if nd > 1 && rng.Intn(2) == 0 {
			perm := rng.Perm(nd)
			p, err := v.Permute(perm...)
			if err != nil {
				t.Fatal(err)
			}
			v = p
		}
		if v.Shape()[0] > 1 && rng.Intn(2) == 0 {
			sl, err := v.Slice(0, 1, v.Shape()[0])
			if err != nil {
				t.Fatal(err)
			}
			v = sl
		}
		got := v.Contiguous()
		// Oracle: read every logical position through the view's own accessor, in row-major order.
		n := v.Numel()
		idx := make([]int, v.Ndim())
		for k := range n {
			rem := k
			for d := v.Ndim() - 1; d >= 0; d-- {
				idx[d] = rem % v.Shape()[d]
				rem /= v.Shape()[d]
			}
			want := v.AtF64(idx...)
			g := got.AtF64(idx...)
			if math.Float64bits(g) != math.Float64bits(want) {
				t.Fatalf("shape %v strides %v element %d: contiguous %v, oracle %v",
					v.Shape(), v.Strides(), k, g, want)
			}
		}
	}
}

// TestCoalesceDimsOnlyMergesDensePairs pins the merge condition itself. A pair may fold exactly when
// strides[d] == strides[d+1]*shape[d+1]; anything else describes a gap or an overlap, and folding it
// would invent an offset sequence the view never had.
func TestCoalesceDimsOnlyMergesDensePairs(t *testing.T) {
	for _, c := range []struct {
		shape, strides []int
		wantSh, wantSt []int
		name           string
	}{
		// Axes 1-2-3 are dense with each other; axis 0 is NOT (its stride 32768 exceeds the
		// 24576 the merged run spans, because the slice dropped two of the six).
		{[]int{4, 6, 64, 64}, []int{32768, 4096, 64, 1}, []int{4, 24576}, []int{32768, 1},
			"a rank-4 slice of a middle axis becomes 4 runs, not 1536"},
		{[]int{4, 4}, []int{1, 4}, []int{4, 4}, []int{1, 4},
			"a transpose must not merge"},
		{[]int{3, 4}, []int{100, 1}, []int{3, 4}, []int{100, 1},
			"a gapped outer stride must not merge"},
		{[]int{5}, []int{1}, []int{5}, []int{1}, "rank 1 is returned untouched"},
	} {
		sh, st := coalesceDims(c.shape, c.strides)
		if len(sh) != len(c.wantSh) {
			t.Fatalf("%s: rank %d, want %d (%v/%v)", c.name, len(sh), len(c.wantSh), sh, st)
		}
		for i := range sh {
			if sh[i] != c.wantSh[i] || st[i] != c.wantSt[i] {
				t.Fatalf("%s: got %v/%v, want %v/%v", c.name, sh, st, c.wantSh, c.wantSt)
			}
		}
	}
}
