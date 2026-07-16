package classic

import (
	"math/rand/v2"
	"sort"
	"testing"
)

// spatialData builds a fixed synthetic dataset of Gaussian blobs at a given
// size and dimensionality — the §V22 perf-parity probe shape shared with the
// SVM benches, reused here for the kNN/DBSCAN neighbour-search grind.
func spatialData(n, d int) ([][]float64, []int) {
	const classes = 3
	rng := rand.New(rand.NewPCG(7, 7))
	centers := make([][]float64, classes)
	for c := range centers {
		centers[c] = make([]float64, d)
		for j := range centers[c] {
			centers[c][j] = rng.NormFloat64() * 3
		}
	}
	X := make([][]float64, n)
	y := make([]int, n)
	for i := range X {
		c := rng.IntN(classes)
		y[i] = c
		X[i] = make([]float64, d)
		for j := range X[i] {
			X[i][j] = centers[c][j] + rng.NormFloat64()
		}
	}
	return X, y
}

// BenchmarkKNNPredict times predicting n queries (the training set itself) with
// a fitted k-NN classifier at n=4000/d=20 — the §V22 A/B probe for the
// brute-force→ball-tree neighbour search. Run:
//
//	go test ./classic/ -run x -bench BenchmarkKNNPredict -benchmem
func BenchmarkKNNPredict(b *testing.B) {
	X, y := spatialData(4000, 20)
	m := NewKNNClassifier(WithKNNK(5))
	if err := m.Fit(X, y); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Predict(X); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDBSCANFit times a DBSCAN fit at n=4000/d=20 (eps=2, minSamples=5) —
// the §V22 A/B probe for the eps-neighbourhood search. Run:
//
//	go test ./classic/ -run x -bench BenchmarkDBSCANFit -benchmem
func BenchmarkDBSCANFit(b *testing.B) {
	X, _ := spatialData(4000, 20)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := NewDBSCAN(WithDBSCANEps(2), WithDBSCANMinSamples(5)).Fit(X); err != nil {
			b.Fatal(err)
		}
	}
}

// TestBallTreeEquivalence is the correctness gate for the spatial index: on
// random data that deliberately includes exact duplicate points (to force
// distance ties), the ball tree's kNN and radius queries must return the
// bit-identical neighbour sets the brute-force scans do — same members, same
// (dist, idx) tie-break — across both metrics. This backstops the sklearn
// goldens with a high-n, tie-dense stress test.
func TestBallTreeEquivalence(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 22))
	const n, d = 600, 12
	X := make([][]float64, n)
	for i := range X {
		if i > 0 && rng.IntN(6) == 0 { // ~1/6 exact duplicates → distance ties
			X[i] = append([]float64(nil), X[rng.IntN(i)]...)
			continue
		}
		X[i] = make([]float64, d)
		for j := range X[i] {
			X[i][j] = rng.NormFloat64()
		}
	}

	for _, metric := range []struct {
		name string
		knn  KNNMetric
		ball ballMetric
	}{
		{"euclidean", KNNEuclidean, ballL2}, {"manhattan", KNNManhattan, ballL1},
	} {
		bt := buildBallTree(X, metric.ball)
		if bt == nil {
			t.Fatalf("%s: expected a tree for n=%d", metric.name, n)
		}

		// kNN: compare against brute-force nearest for several k.
		for _, k := range []int{1, 3, 5, 9} {
			cfg := knnConfig{k: k, metric: metric.knn}
			for q := 0; q < 120; q++ {
				query := X[rng.IntN(n)]
				want := nearest(X, query, cfg)
				got := bt.kNN(query, k)
				if len(got) != len(want) {
					t.Fatalf("%s k=%d: len got %d want %d", metric.name, k, len(got), len(want))
				}
				for i := range want {
					if got[i].idx != want[i].idx || got[i].dist != want[i].dist {
						t.Fatalf("%s k=%d nb[%d]: got (idx=%d,d=%g) want (idx=%d,d=%g)",
							metric.name, k, i, got[i].idx, got[i].dist, want[i].idx, want[i].dist)
					}
				}
			}
		}

		// radius: compare against the brute-force eps-neighbourhood.
		for _, eps := range []float64{0.5, 1.0, 2.0} {
			eps2 := eps * eps
			for q := 0; q < 120; q++ {
				query := X[rng.IntN(n)]
				got := bt.radius(query, eps, nil)
				var want []int
				for j := range X {
					if bt.within(query, X[j], eps, eps2) {
						want = append(want, j)
					}
				}
				sort.Ints(want)
				if len(got) != len(want) {
					t.Fatalf("%s radius eps=%g: len got %d want %d", metric.name, eps, len(got), len(want))
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%s radius eps=%g: idx[%d] got %d want %d", metric.name, eps, i, got[i], want[i])
					}
				}
			}
		}
	}
}
