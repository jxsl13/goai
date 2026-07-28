package classic

import "testing"

// TestKNNEquidistantTieBreaksByLowestIndex pins the (dist, idx) total order that both the
// ball-tree and brute-force neighbour paths sort by.
//
// Random data never exercises it: two points are essentially never bit-equidistant from a
// query, and mutation probing confirmed that INVERTING the index comparison leaves every
// pre-existing KNN test green. The order is documented as "identical to a full brute-force
// sort by (dist, idx)" — this is what checks that rather than trusting it.
//
// Here the query sits exactly between mirrored points, so distances tie to the bit and the
// index decides. Both paths are exercised: the ball tree is built above the brute-force
// cutoff, and nearest() is called directly below it.
func TestKNNEquidistantTieBreaksByLowestIndex(t *testing.T) {
	// Indices 0 and 1 are mirrored at distance 1 from the origin, so their distances tie
	// TO THE BIT and only the index can order them. The rest sit far away purely to push
	// the point count past ballLeafSize — below it buildBallTree returns nil and the tree
	// path would not be exercised at all.
	pts := [][]float64{{-1, 0}, {1, 0}}
	for i := range 64 {
		pts = append(pts, []float64{float64(10 + i), float64(10 + i)})
	}
	query := []float64{0, 0}

	got := nearest(pts, query, knnConfig{k: 2, metric: KNNEuclidean})
	if len(got) < 2 {
		t.Fatalf("nearest returned %d neighbours, want at least 2", len(got))
	}
	if got[0].idx != 0 || got[1].idx != 1 {
		t.Fatalf("brute force broke the tie as (%d,%d); equidistant neighbours must come "+
			"back in ascending index order (0,1)", got[0].idx, got[1].idx)
	}

	bt := buildBallTree(pts, ballL2)
	bn := bt.kNN(query, 2)
	if len(bn) < 2 {
		t.Fatalf("ball tree returned %d neighbours, want at least 2", len(bn))
	}
	if bn[0].idx != 0 || bn[1].idx != 1 {
		t.Fatalf("ball tree broke the tie as (%d,%d), want (0,1) — the two paths must agree "+
			"on the (dist, idx) order", bn[0].idx, bn[1].idx)
	}
}
