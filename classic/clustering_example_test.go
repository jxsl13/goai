package classic_test

import (
	"fmt"

	"github.com/jxsl13/goai/classic"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// GaussianMixture is soft, probabilistic clustering: it models the data as a
// blend of Gaussian "blobs" and, unlike k-means, reports how strongly each point
// belongs to each component (PredictProba) and a proper density per sample
// (ScoreSamples). Two well-separated groups are recovered as two components; the
// full covariance lets each blob be stretched or tilted.
func ExampleGaussianMixture() {
	x := [][]float64{{0, 0}, {0.3, -0.2}, {10, 10}, {9.7, 10.2}}
	m := classic.NewGaussianMixture(
		classic.WithGMMComponents(2),
		classic.WithGMMCovariance(classic.GMMFull),
		classic.WithGMMSeed(1),
	)
	if err := m.Fit(x); err != nil {
		panic(err)
	}
	labels, _ := m.Predict(x)
	proba, _ := m.PredictProba(x)
	ss, _ := m.ScoreSamples(x)
	score, _ := m.Score(x)
	_ = m.Covariance(0) // per-component covariance is available too

	fmt.Println("covariance:", classic.GMMFull)
	// Labels are permutation-invariant, so report the grouping, not raw ids.
	fmt.Println("two blobs separated:", labels[0] == labels[1] && labels[2] == labels[3] && labels[0] != labels[2])
	fmt.Printf("responsibilities sum to %.1f\n", proba[0][0]+proba[0][1])
	fmt.Printf("scored %d samples, mean log-likelihood finite: %v\n", len(ss), score > -1e9)
	// Output:
	// covariance: full
	// two blobs separated: true
	// responsibilities sum to 1.0
	// scored 4 samples, mean log-likelihood finite: true
}

// DBSCAN groups points by density and needs no K: it finds however many dense
// regions exist and marks sparse points as noise (label -1). Here two tight
// triples form two clusters and the lone far-away point is noise.
func ExampleDBSCAN() {
	x := [][]float64{
		{0, 0}, {0.1, 0.1}, {0.2, 0}, // cluster A
		{10, 10}, {10.1, 9.9}, {10, 10.2}, // cluster B
		{5, -5}, // noise
	}
	m := classic.NewDBSCAN(
		classic.WithDBSCANEps(0.5),
		classic.WithDBSCANMinSamples(3),
		classic.WithDBSCANMetric(classic.DBSCANEuclidean),
	)
	labels, err := m.Fit(x)
	if err != nil {
		panic(err)
	}
	_ = labels // same slice as m.Labels()

	fmt.Println("metric:", classic.DBSCANEuclidean)
	fmt.Println("clusters:", m.NumClusters())
	fmt.Println("core points:", len(m.CoreSampleIndices()))
	fmt.Println("last point is noise:", m.Labels()[6] == classic.DBSCANLabelNoise)
	// Output:
	// metric: euclidean
	// clusters: 2
	// core points: 6
	// last point is noise: true
}
