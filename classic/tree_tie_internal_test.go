package classic

import (
	"math"
	"testing"
)

// TestCARTSplitTieGoesToTheLowestFeature pins the rule the parallel feature scan has to
// reproduce, and pins it DIRECTLY rather than through a prediction digest. Two features that
// are exact copies of each other give identical impurity at every cut, so which one the tree
// records is decided purely by the tie rule: the serial loop visits features in ascending order
// and takes a new best only on strict <, so the lowest index wins.
//
// The bit-exact forest digest does NOT cover this — mutations flipping both strict < tests to
// <= left it green, because its data produces no exact-cost tie between different features.
//
// The width matters twice over. It has to clear the parallel gate at the root, and it has to be
// wide enough that a chunk holds SEVERAL features — with eight features the worker count clamps
// to eight, every chunk holds exactly one, and a mutation of the within-chunk tie rule has
// nothing to bite on. At sixty-four the duplicated pair at indices 0 and 1 shares a chunk, so
// both the within-chunk rule and the cross-chunk fold are exercised.
func TestCARTSplitTieGoesToTheLowestFeature(t *testing.T) {
	const n, d = 4096, 64
	// TWO PLACEMENTS, because the rule is enforced in two places and each fixture only
	// exercises one. Duplicating an ADJACENT pair puts both features in the same chunk, where
	// the within-chunk comparison decides; duplicating the FIRST and LAST puts them in
	// different chunks, where the fold across chunks decides. A mutation of either strict <
	// leaves the other placement green.
	for _, twin := range []int{1, d - 1} {
		x := make([][]float64, n)
		y := make([]int, n)
		for i := range x {
			row := make([]float64, d)
			v := math.Sin(float64(i) * 0.37)
			row[0], row[twin] = v, v // exact copies: identical impurity at every cut
			for j := 1; j < d; j++ {
				if j == twin {
					continue
				}
				row[j] = math.Cos(float64(i*j) * 0.11) // weaker, and different from each other
			}
			x[i] = row
			if v > 0 {
				y[i] = 1
			}
		}
		m := NewDecisionTreeClassifier(WithMaxDepth(3))
		if err := m.Fit(x, y); err != nil {
			t.Fatal(err)
		}
		if m.root == nil || m.root.leaf {
			t.Fatalf("twin=%d: want a split at the root, got a leaf", twin)
		}
		if m.root.feature != 0 {
			t.Fatalf("twin=%d: root split on feature %d, want 0 — features 0 and %d are"+
				" identical, so the tie must go to the LOWEST index", twin, m.root.feature, twin)
		}
	}
}
