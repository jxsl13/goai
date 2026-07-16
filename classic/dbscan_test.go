package classic

import (
	"encoding/json"
	"os"
	"testing"
)

// --- golden loading -----------------------------------------------------------

type dbscanGolden struct {
	X          []float64 `json:"X"`
	N          int       `json:"n"`
	D          int       `json:"d"`
	Eps        float64   `json:"eps"`
	MinSamples int       `json:"min_samples"`
	Metric     string    `json:"metric"`
	Labels     []int     `json:"labels"`
	Core       []bool    `json:"core"`
	NClusters  int       `json:"n_clusters"`
	NNoise     int       `json:"n_noise"`
}

type dbscanData struct {
	Sklearn   string       `json:"sklearn"`
	Blobs     dbscanGolden `json:"blobs"`
	EpsSmall  dbscanGolden `json:"eps_small"`
	EpsMid    dbscanGolden `json:"eps_mid"`
	EpsLarge  dbscanGolden `json:"eps_large"`
	Manhattan dbscanGolden `json:"manhattan"`
	Moons     dbscanGolden `json:"moons"`
}

func loadDBSCAN(t *testing.T) dbscanData {
	t.Helper()
	raw, err := os.ReadFile("testdata/dbscan.json")
	if err != nil {
		t.Fatalf("read golden (run classic/testdata/gen_dbscan.py): %v", err)
	}
	var g dbscanData
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func dbscanMetric(s string) DBSCANMetric {
	if s == "manhattan" {
		return DBSCANManhattan
	}
	return DBSCANEuclidean
}

// sameLabelingUpToRenumber reports whether two label vectors describe the same
// partition: noise (−1) must line up exactly, and there must be a consistent
// bijection between the non-noise cluster ids of got and want.
func sameLabelingUpToRenumber(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	g2w := map[int]int{}
	w2g := map[int]int{}
	for i := range got {
		if (got[i] == DBSCANLabelNoise) != (want[i] == DBSCANLabelNoise) {
			return false
		}
		if got[i] == DBSCANLabelNoise {
			continue
		}
		if m, ok := g2w[got[i]]; ok {
			if m != want[i] {
				return false
			}
		} else {
			g2w[got[i]] = want[i]
		}
		if m, ok := w2g[want[i]]; ok {
			if m != got[i] {
				return false
			}
		} else {
			w2g[want[i]] = got[i]
		}
	}
	return true
}

// §V1 (sklearn-golden parity anchor): DBSCAN labels match scikit-learn EXACTLY
// up to a renumbering of cluster ids, and core-point classification matches
// exactly, across metrics and datasets.
func TestDBSCANParity(t *testing.T) {
	data := loadDBSCAN(t)
	cases := []struct {
		name string
		g    dbscanGolden
	}{
		{"blobs", data.Blobs},
		{"eps_small", data.EpsSmall},
		{"eps_mid", data.EpsMid},
		{"eps_large", data.EpsLarge},
		{"manhattan", data.Manhattan},
		{"moons", data.Moons},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := tc.g
			x := rows(g.X, g.N, g.D)
			m := NewDBSCAN(
				WithDBSCANEps(g.Eps),
				WithDBSCANMinSamples(g.MinSamples),
				WithDBSCANMetric(dbscanMetric(g.Metric)),
			)
			labels, err := m.Fit(x)
			if err != nil {
				t.Fatal(err)
			}
			if !sameLabelingUpToRenumber(labels, g.Labels) {
				t.Errorf("%s: labels differ from sklearn (up to renumbering)", tc.name)
			}
			if got := m.NumClusters(); got != g.NClusters {
				t.Errorf("%s: NumClusters got %d want %d", tc.name, got, g.NClusters)
			}
			// core-point classification matches exactly.
			core := make([]bool, g.N)
			for _, i := range m.CoreSampleIndices() {
				core[i] = true
			}
			for i := range core {
				if core[i] != g.Core[i] {
					t.Errorf("%s: core[%d] got %v want %v", tc.name, i, core[i], g.Core[i])
					break
				}
			}
			t.Logf("%s (sklearn %s): %d clusters, matched", tc.name, data.Sklearn, g.NClusters)
		})
	}
}

// §V (two-blobs + noise property anchor): DBSCAN finds exactly the 2 dense
// blobs and labels the injected outliers as noise (−1).
func TestDBSCANTwoBlobsNoise(t *testing.T) {
	g := loadDBSCAN(t).Blobs
	x := rows(g.X, g.N, g.D)
	m := NewDBSCAN(WithDBSCANEps(g.Eps), WithDBSCANMinSamples(g.MinSamples))
	labels, err := m.Fit(x)
	if err != nil {
		t.Fatal(err)
	}
	if m.NumClusters() != 2 {
		t.Fatalf("expected 2 clusters, got %d", m.NumClusters())
	}
	noise := 0
	for _, l := range labels {
		if l == DBSCANLabelNoise {
			noise++
		}
	}
	if noise != g.NNoise {
		t.Errorf("noise count got %d want %d", noise, g.NNoise)
	}
	// the 4 explicitly-injected far-away points (last 4 rows) must be noise.
	for i := g.N - 4; i < g.N; i++ {
		if labels[i] != DBSCANLabelNoise {
			t.Errorf("injected outlier %d not labeled noise (got %d)", i, labels[i])
		}
	}
}

// §V (core/border/noise hand case): a hand-built line of points exercises the
// three roles — dense interior points are core, the fringe is border, an
// isolated point is noise.
func TestDBSCANCoreBorderNoise(t *testing.T) {
	// Five collinear points spaced 1 apart, plus a far outlier.
	x := [][]float64{{0}, {1}, {2}, {3}, {4}, {20}}
	m := NewDBSCAN(WithDBSCANEps(1.0), WithDBSCANMinSamples(3))
	labels, err := m.Fit(x)
	if err != nil {
		t.Fatal(err)
	}
	// With eps=1, minSamples=3 (self included): interior points 1,2,3 have 3
	// neighbors -> core; endpoints 0,4 have 2 -> border; point 5 is isolated
	// -> noise.
	core := map[int]bool{}
	for _, i := range m.CoreSampleIndices() {
		core[i] = true
	}
	for _, i := range []int{1, 2, 3} {
		if !core[i] {
			t.Errorf("point %d should be core", i)
		}
	}
	for _, i := range []int{0, 4} {
		if core[i] {
			t.Errorf("point %d should be border, not core", i)
		}
	}
	if labels[0] == DBSCANLabelNoise || labels[4] == DBSCANLabelNoise {
		t.Error("border points should join the cluster, not be noise")
	}
	if labels[5] != DBSCANLabelNoise {
		t.Errorf("isolated point should be noise, got %d", labels[5])
	}
	if m.NumClusters() != 1 {
		t.Errorf("expected 1 cluster, got %d", m.NumClusters())
	}
}

// §V (eps monotonicity): larger eps yields fewer, bigger clusters (fewer noise
// points), confirmed against the three-eps goldens.
func TestDBSCANEpsMonotonicity(t *testing.T) {
	data := loadDBSCAN(t)
	small, mid, large := data.EpsSmall, data.EpsMid, data.EpsLarge
	x := rows(small.X, small.N, small.D)
	noiseOf := func(eps float64, ms int) (clusters, noise int) {
		m := NewDBSCAN(WithDBSCANEps(eps), WithDBSCANMinSamples(ms))
		labels, err := m.Fit(x)
		if err != nil {
			t.Fatal(err)
		}
		for _, l := range labels {
			if l == DBSCANLabelNoise {
				noise++
			}
		}
		return m.NumClusters(), noise
	}
	cS, nS := noiseOf(small.Eps, small.MinSamples)
	cM, nM := noiseOf(mid.Eps, mid.MinSamples)
	cL, nL := noiseOf(large.Eps, large.MinSamples)
	if !(nS >= nM && nM >= nL) {
		t.Errorf("noise not monotone decreasing in eps: %d,%d,%d", nS, nM, nL)
	}
	if !(cL <= cM) {
		t.Errorf("cluster count should not grow at very large eps: mid=%d large=%d", cM, cL)
	}
	if cL != 1 {
		t.Errorf("very large eps should merge to 1 cluster, got %d", cL)
	}
	t.Logf("eps monotonicity: clusters %d/%d/%d noise %d/%d/%d", cS, cM, cL, nS, nM, nL)
}

// §V (determinism): DBSCAN is fully deterministic given the inputs.
func TestDBSCANDeterminism(t *testing.T) {
	g := loadDBSCAN(t).Moons
	x := rows(g.X, g.N, g.D)
	run := func() []int {
		m := NewDBSCAN(WithDBSCANEps(g.Eps), WithDBSCANMinSamples(g.MinSamples))
		l, err := m.Fit(x)
		if err != nil {
			t.Fatal(err)
		}
		return l
	}
	a, b := run(), run()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic label at %d: %d vs %d", i, a[i], b[i])
		}
	}
}

// §V (degenerate): all-identical points collapse into a single cluster (or all
// noise when minSamples exceeds n).
func TestDBSCANDegenerate(t *testing.T) {
	x := [][]float64{{1, 1}, {1, 1}, {1, 1}, {1, 1}}
	m := NewDBSCAN(WithDBSCANEps(0.5), WithDBSCANMinSamples(2))
	labels, err := m.Fit(x)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range labels {
		if l != 0 {
			t.Errorf("identical points should all be cluster 0, got %d", l)
		}
	}
	// minSamples > n: no core points -> all noise.
	m2 := NewDBSCAN(WithDBSCANEps(0.5), WithDBSCANMinSamples(10))
	labels2, err := m2.Fit(x)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range labels2 {
		if l != DBSCANLabelNoise {
			t.Errorf("no core points -> all noise, got %d", l)
		}
	}
	if m2.NumClusters() != 0 {
		t.Errorf("expected 0 clusters, got %d", m2.NumClusters())
	}
}

// §V (sanity/errors): guard the documented failure modes.
func TestDBSCANErrors(t *testing.T) {
	if _, err := NewDBSCAN().Fit(nil); err == nil {
		t.Error("empty fit must error")
	}
	if _, err := NewDBSCAN(WithDBSCANEps(0)).Fit([][]float64{{1}}); err == nil {
		t.Error("eps ≤ 0 must error")
	}
	if _, err := NewDBSCAN(WithDBSCANMinSamples(0)).Fit([][]float64{{1}}); err == nil {
		t.Error("minSamples < 1 must error")
	}
	if _, err := NewDBSCAN(WithDBSCANEps(1)).Fit([][]float64{{1, 2}, {3}}); err == nil {
		t.Error("ragged X must error")
	}
}

// String coverage for the metric enum.
func TestDBSCANMetricString(t *testing.T) {
	if DBSCANEuclidean.String() != "euclidean" || DBSCANManhattan.String() != "manhattan" {
		t.Errorf("unexpected strings: %q %q", DBSCANEuclidean, DBSCANManhattan)
	}
	if DBSCANMetric(9).String() == "" {
		t.Error("unknown metric should still render")
	}
}
