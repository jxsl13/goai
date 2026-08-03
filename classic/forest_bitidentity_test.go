package classic

import (
	"math"
	"testing"
)

// forestDigest hashes every bit of a fitted forest's and a lone tree's predictions. Pooling
// scratch is only safe if every buffer is fully written before it is read; a buffer that keeps
// one stale value from the previous tree changes a split without changing accuracy enough to
// notice, so a bit-exact comparison is the only gate that sees it.
func forestDigest(t *testing.T, n, d, nTrees int, entropy bool) uint64 {
	t.Helper()
	x := make([][]float64, n)
	y := make([]int, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			// Coarse quantization so many samples tie on a feature, which is where a stale
			// sort or sweep buffer would show.
			row[j] = math.Trunc(math.Sin(float64(i*13+j*29))*12) / 3
		}
		x[i] = row
		if row[0]+row[d-1] > 0 {
			y[i] = 1
		}
		if row[1] > 2 {
			y[i] = 2
		}
	}
	opts := []ForestOption{WithNumTrees(nTrees), WithSeed(7)}
	if entropy {
		opts = append(opts, WithForestCriterion(Entropy))
	}
	f := NewRandomForestClassifier(opts...)
	if err := f.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	pf, err := f.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	// A standalone tree too: the forest path always subsamples features, and the lone tree
	// with default options takes the other branch of the builder.
	tr := NewDecisionTreeClassifier(WithMaxDepth(6))
	if err := tr.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	pt, err := tr.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	for _, p := range [][]int{pf, pt} {
		for _, v := range p {
			u := uint64(v)
			for s := 0; s < 64; s += 8 {
				h = (h ^ (u>>s)&0xff) * 1099511628211
			}
		}
	}
	return h
}

// TestForestIsBitIdentical freezes the forest and the standalone tree across both split
// criteria and across the radix cutoff, which selects a different per-node sort.
func TestForestIsBitIdentical(t *testing.T) {
	cases := []struct {
		n, d, nTrees int
		entropy      bool
		want         uint64
	}{
		{300, 6, 8, false, 10262320782983183781},
		// The two n=300 rows agree BY COINCIDENCE OF THE DATA, not because the criterion is
		// ignored: at n=3000 they differ. The small pair is kept because it exercises the
		// comparison-sort branch below the radix cutoff.
		{300, 6, 8, true, 10262320782983183781}, // Entropy: exercises the clogc table
		{3000, 8, 4, false, 15365606350868434405},
		{3000, 8, 4, true, 10187654040800196197},
	}
	for _, c := range cases {
		got := forestDigest(t, c.n, c.d, c.nTrees, c.entropy)
		if got != c.want {
			t.Fatalf("n=%d d=%d trees=%d entropy=%v digest %d, want %d",
				c.n, c.d, c.nTrees, c.entropy, got, c.want)
		}
	}
}
