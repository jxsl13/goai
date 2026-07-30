package classic

import (
	"fmt"
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

// BenchmarkDBSCANFit times a DBSCAN fit at n=4000/d=20, minSamples=5. Run:
//
//	go test ./classic/ -run x -bench BenchmarkDBSCANFit -benchmem
//
// eps IS THE WHOLE BENCHMARK, and the original single eps=2 arm was degenerate.
// spatialData puts blob centres at N(0,9) per dim and points at centre+N(0,1) per dim, so
// for two points in the same blob d² = 2·χ²₂₀ with mean 40. eps=2 demands d² ≤ 4, i.e.
// χ²₂₀ ≤ 2, probability ~1e-7 — across ~5.3M same-blob pairs that is well under one
// expected pair. Measured: eps=2 yields 0 clusters, 0 core points and all 4000 points
// noise, so every neighbourhood is the singleton {i} and the cluster-expansion flood fill
// at dbscan.go:213 never runs a single iteration. That arm times a ball-tree build plus
// 4000 empty radius queries and NOTHING else — it cannot A/B the labeling phase, and it
// cannot A/B neighbour-list handling either, because there are no neighbours.
//
// So the arms are explicit about which regime they cover:
//
//	eps2 — kept for continuity with previously recorded numbers, and as the empty-result
//	       edge case. Labeled degenerate so nobody reads it as covering expansion.
//	eps4 — the load-bearing arm: 3 clusters, 2339 core points, 675 noise, so core, border
//	       and noise points all occur, the flood fill runs, and neighbour lists are long
//	       enough for their allocation to be visible.
//
// Both hold minSamples=5 and the fixed spatialData seed, so the arms differ only in eps.
func BenchmarkDBSCANFit(b *testing.B) {
	X, _ := spatialData(4000, 20)
	for _, eps := range []float64{2, 4} {
		b.Run(dbscanBenchName(eps), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := NewDBSCAN(WithDBSCANEps(eps), WithDBSCANMinSamples(5)).Fit(X); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkDBSCANFitManhattan drives the ball tree's L1 leaf test, which no benchmark reached.
//
// That gap had a consequence: when the L2 arm's per-dimension bail-out was moved to every fourth
// dimension, the L1 arm next to it was left alone precisely because a change there could not have
// been validated. PS3008 reported it as a candidate; this is what lets the candidate be measured
// instead of argued.
//
// eps is larger than the Euclidean sweep's because a Manhattan radius sums coordinates rather than
// squaring them, so the same geometry needs a bigger radius to produce comparable neighbourhoods.
func BenchmarkDBSCANFitManhattan(b *testing.B) {
	X, _ := spatialData(4000, 20)
	for _, eps := range []float64{8, 16} {
		b.Run(fmt.Sprintf("eps%g", eps), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := NewDBSCAN(WithDBSCANEps(eps), WithDBSCANMinSamples(5),
					WithDBSCANMetric(DBSCANManhattan)).Fit(X)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func dbscanBenchName(eps float64) string {
	switch eps {
	case 2:
		return "eps2_degenerate_all_noise"
	case 4:
		return "eps4"
	}
	return "eps"
}

// TestDBSCANBenchRegimes pins the two benchmark arms' cluster structure. A benchmark whose
// input silently drifts into a degenerate regime measures the wrong thing while still
// looking healthy — which is exactly what happened to the original eps=2 arm — so the
// regime each arm claims to cover is asserted rather than described in a comment.
func TestDBSCANBenchRegimes(t *testing.T) {
	X, _ := spatialData(4000, 20)
	for _, tc := range []struct {
		eps                   float64
		clusters, core, noise int
	}{
		{2, 0, 0, 4000},
		{4, 3, 2339, 675},
	} {
		m := NewDBSCAN(WithDBSCANEps(tc.eps), WithDBSCANMinSamples(5))
		labels, err := m.Fit(X)
		if err != nil {
			t.Fatal(err)
		}
		noise := 0
		for _, l := range labels {
			if l == DBSCANLabelNoise {
				noise++
			}
		}
		if got := m.NumClusters(); got != tc.clusters {
			t.Errorf("eps=%g: %d clusters, want %d", tc.eps, got, tc.clusters)
		}
		if got := len(m.CoreSampleIndices()); got != tc.core {
			t.Errorf("eps=%g: %d core points, want %d", tc.eps, got, tc.core)
		}
		if noise != tc.noise {
			t.Errorf("eps=%g: %d noise points, want %d", tc.eps, noise, tc.noise)
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

// BenchmarkKNNFit times the FIT, which is where the ball-tree index is built — the
// half of KNN that has historically lost to sklearn, and which BenchmarkKNNPredict
// deliberately excludes by fitting before ResetTimer.
func BenchmarkKNNFit(b *testing.B) {
	X, y := spatialData(4000, 20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewKNNClassifier(WithKNNK(5))
		if err := m.Fit(X, y); err != nil {
			b.Fatal(err)
		}
	}
}
