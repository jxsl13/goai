package classic

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// --- golden loading -----------------------------------------------------------

type gmmGolden struct {
	X            []float64 `json:"X"`
	N            int       `json:"n"`
	D            int       `json:"d"`
	K            int       `json:"k"`
	Covariance   string    `json:"covariance"`
	RegCovar     float64   `json:"reg_covar"`
	Tol          float64   `json:"tol"`
	MaxIter      int       `json:"max_iter"`
	Weights      []float64 `json:"weights"`
	Means        []float64 `json:"means"`
	Covariances  []float64 `json:"covariances"`
	Predict      []int     `json:"predict"`
	ScoreSamples []float64 `json:"score_samples"`
	LowerBound   float64   `json:"lower_bound"`
	NIter        int       `json:"n_iter"`
	Converged    bool      `json:"converged"`
	Truth        []int     `json:"truth"`
	TrueMeans    []float64 `json:"true_means"`
	DataMean     []float64 `json:"data_mean"`
	DataCov      []float64 `json:"data_cov"`
}

type gmmData struct {
	Sklearn string    `json:"sklearn"`
	Diag    gmmGolden `json:"diag"`
	Full    gmmGolden `json:"full"`
	K1      gmmGolden `json:"k1"`
}

func loadGMM(t *testing.T) gmmData {
	t.Helper()
	raw, err := os.ReadFile("testdata/gmm.json")
	if err != nil {
		t.Fatalf("read golden (run classic/testdata/gen_gmm.py): %v", err)
	}
	var g gmmData
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func gmmCovType(s string) GMMCovariance {
	if s == "full" {
		return GMMFull
	}
	return GMMDiag
}

// matchByMeans builds a permutation mapping each fitted component to the sklearn
// component with the nearest mean, so assignments can be compared up to a label
// permutation (EM is invariant to component ordering).
func matchByMeans(gotMeans, skMeans [][]float64) []int {
	k := len(gotMeans)
	perm := make([]int, k)
	used := make([]bool, k)
	for g := range k {
		best, bd := -1, math.Inf(1)
		for s := range k {
			if used[s] {
				continue
			}
			var dd float64
			for j := range gotMeans[g] {
				dv := gotMeans[g][j] - skMeans[s][j]
				dd += dv * dv
			}
			if dd < bd {
				bd, best = dd, s
			}
		}
		perm[g] = best
		used[best] = true
	}
	return perm
}

// §V1 (sklearn-golden parity anchor): on well-separated data EM has a unique
// optimum, so GoAI's fit matches sklearn's cluster assignments up to a label
// permutation and its per-sample log-likelihood to rtol 1e-3.
func TestGMMParity(t *testing.T) {
	data := loadGMM(t)
	for _, tc := range []struct {
		name string
		g    gmmGolden
	}{
		{"diag", data.Diag}, {"full", data.Full},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := tc.g
			x := rows(g.X, g.N, g.D)
			m := NewGaussianMixture(
				WithGMMComponents(g.K),
				WithGMMCovariance(gmmCovType(g.Covariance)),
				WithGMMRegCovar(g.RegCovar),
				WithGMMTol(g.Tol),
				WithGMMMaxIter(g.MaxIter),
				WithGMMSeed(0),
			)
			if err := m.Fit(x); err != nil {
				t.Fatal(err)
			}
			skMeans := rows(g.Means, g.K, g.D)
			perm := matchByMeans(m.Means, skMeans)

			// (a) assignments match up to the permutation.
			pred, err := m.Predict(x)
			if err != nil {
				t.Fatal(err)
			}
			mismatch := 0
			for i := range pred {
				if perm[pred[i]] != g.Predict[i] {
					mismatch++
				}
			}
			if mismatch != 0 {
				t.Errorf("%s: %d/%d assignments differ from sklearn (up to permutation)", tc.name, mismatch, g.N)
			}

			// (b) matched means are close to sklearn's.
			for gk := range g.K {
				for j := range g.D {
					if math.Abs(m.Means[gk][j]-skMeans[perm[gk]][j]) > 1e-2 {
						t.Errorf("%s: mean[%d][%d] got %.5g want %.5g", tc.name, gk, j, m.Means[gk][j], skMeans[perm[gk]][j])
					}
				}
			}

			// (c) per-sample log-likelihood matches sklearn to rtol 1e-3.
			ss, err := m.ScoreSamples(x)
			if err != nil {
				t.Fatal(err)
			}
			var maxRel float64
			for i := range ss {
				rel := math.Abs(ss[i]-g.ScoreSamples[i]) / math.Max(1, math.Abs(g.ScoreSamples[i]))
				if rel > maxRel {
					maxRel = rel
				}
			}
			if maxRel > 1e-3 {
				t.Errorf("%s: max rel score_samples diff %.2g > 1e-3", tc.name, maxRel)
			}
			// (d) mean log-likelihood (lower bound) matches.
			score, _ := m.Score(x)
			if math.Abs(score-g.LowerBound) > 1e-3*math.Max(1, math.Abs(g.LowerBound)) {
				t.Errorf("%s: mean LL got %.6g want %.6g", tc.name, score, g.LowerBound)
			}
			t.Logf("%s (sklearn %s): maxRelLL=%.2g meanLL=%.6g nIter=%d", tc.name, data.Sklearn, maxRel, score, m.nIter)
		})
	}
}

// §V (LL-monotone mechanism): the EM mean log-likelihood is non-decreasing at
// every iteration.
func TestGMMLogLikelihoodMonotone(t *testing.T) {
	data := loadGMM(t)
	g := data.Full
	x := rows(g.X, g.N, g.D)
	m := NewGaussianMixture(
		WithGMMComponents(g.K),
		WithGMMCovariance(GMMFull),
		WithGMMRegCovar(g.RegCovar),
		WithGMMTol(0), // disable early stop: run the full history
		WithGMMMaxIter(50),
	)
	if err := m.Fit(x); err != nil {
		t.Fatal(err)
	}
	if len(m.llHistory) < 2 {
		t.Fatalf("need ≥2 iterations to test monotonicity, got %d", len(m.llHistory))
	}
	for i := 1; i < len(m.llHistory); i++ {
		if m.llHistory[i] < m.llHistory[i-1]-1e-9 {
			t.Errorf("LL decreased at iter %d: %.10g -> %.10g", i, m.llHistory[i-1], m.llHistory[i])
		}
	}
	t.Logf("LL history: %.6g -> %.6g over %d iters", m.llHistory[0], m.llHistory[len(m.llHistory)-1], len(m.llHistory))
}

// §V (K=1 collapse anchor): a one-component GMM is exactly the single
// maximum-likelihood Gaussian — its mean/covariance equal the data's population
// mean/covariance, and its per-sample score matches sklearn — to 1e-6.
func TestGMMK1Collapse(t *testing.T) {
	g := loadGMM(t).K1
	x := rows(g.X, g.N, g.D)
	m := NewGaussianMixture(
		WithGMMComponents(1),
		WithGMMCovariance(GMMFull),
		WithGMMRegCovar(0),
		WithGMMTol(1e-12),
		WithGMMMaxIter(200),
	)
	if err := m.Fit(x); err != nil {
		t.Fatal(err)
	}
	if math.Abs(m.Weights[0]-1) > 1e-12 {
		t.Errorf("K=1 weight must be 1, got %.12g", m.Weights[0])
	}
	for j := range g.D {
		if math.Abs(m.Means[0][j]-g.DataMean[j]) > 1e-6 {
			t.Errorf("mean[%d] got %.10g want data mean %.10g", j, m.Means[0][j], g.DataMean[j])
		}
	}
	cov := m.Covariance(0)
	for a := range g.D {
		for b := range g.D {
			want := g.DataCov[a*g.D+b]
			if math.Abs(cov[a][b]-want) > 1e-6 {
				t.Errorf("cov[%d][%d] got %.10g want population cov %.10g", a, b, cov[a][b], want)
			}
		}
	}
	// score_samples must also match sklearn's single-Gaussian density.
	ss, err := m.ScoreSamples(x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range ss {
		if math.Abs(ss[i]-g.ScoreSamples[i]) > 1e-6 {
			t.Errorf("score[%d] got %.10g want %.10g", i, ss[i], g.ScoreSamples[i])
		}
	}
}

// §V (recovers-known-clusters anchor): on data from well-separated Gaussians,
// GMM recovers the ground-truth components — assignments match the truth up to a
// permutation, and the fitted means land near the true generating means.
func TestGMMRecoversClusters(t *testing.T) {
	g := loadGMM(t).Diag
	x := rows(g.X, g.N, g.D)
	m := NewGaussianMixture(
		WithGMMComponents(g.K),
		WithGMMCovariance(GMMDiag),
		WithGMMRegCovar(g.RegCovar),
		WithGMMMaxIter(g.MaxIter),
	)
	if err := m.Fit(x); err != nil {
		t.Fatal(err)
	}
	trueMeans := rows(g.TrueMeans, g.K, g.D)
	perm := matchByMeans(m.Means, trueMeans)
	// means near the true generating centers (blobs, scale 0.6).
	for gk := range g.K {
		for j := range g.D {
			if math.Abs(m.Means[gk][j]-trueMeans[perm[gk]][j]) > 0.3 {
				t.Errorf("mean[%d][%d] got %.4g want true %.4g", gk, j, m.Means[gk][j], trueMeans[perm[gk]][j])
			}
		}
	}
	pred, err := m.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	correct := 0
	for i := range pred {
		if perm[pred[i]] == g.Truth[i] {
			correct++
		}
	}
	acc := float64(correct) / float64(g.N)
	if acc < 0.99 {
		t.Errorf("cluster-recovery accuracy %.3f < 0.99", acc)
	}
	t.Logf("recovered %d well-separated clusters at accuracy %.3f", g.K, acc)
}

// §V (determinism): the same seed yields identical fits.
func TestGMMDeterminism(t *testing.T) {
	g := loadGMM(t).Diag
	x := rows(g.X, g.N, g.D)
	fit := func() *GaussianMixture {
		m := NewGaussianMixture(WithGMMComponents(g.K), WithGMMSeed(7), WithGMMMaxIter(50))
		if err := m.Fit(x); err != nil {
			t.Fatal(err)
		}
		return m
	}
	a, b := fit(), fit()
	for k := range g.K {
		for j := range g.D {
			if a.Means[k][j] != b.Means[k][j] {
				t.Fatalf("non-deterministic mean[%d][%d]: %v vs %v", k, j, a.Means[k][j], b.Means[k][j])
			}
		}
	}
}

// §V (sanity/errors): guard the documented failure modes.
func TestGMMErrors(t *testing.T) {
	if err := NewGaussianMixture(WithGMMComponents(2)).Fit(nil); err == nil {
		t.Error("empty fit must error")
	}
	if err := NewGaussianMixture(WithGMMComponents(3)).Fit([][]float64{{1, 2}, {3, 4}}); err == nil {
		t.Error("K>n must error")
	}
	if err := NewGaussianMixture(WithGMMComponents(1), WithGMMRegCovar(-1)).Fit([][]float64{{1}}); err == nil {
		t.Error("negative regCovar must error")
	}
	if err := NewGaussianMixture(WithGMMComponents(1)).Fit([][]float64{{1, 2}, {3}}); err == nil {
		t.Error("ragged X must error")
	}
	var m GaussianMixture
	m.cfg = defaultGMMConfig()
	if _, err := m.Predict([][]float64{{1}}); err == nil {
		t.Error("predict before fit must error")
	}
	if _, err := m.ScoreSamples([][]float64{{1}}); err == nil {
		t.Error("score before fit must error")
	}
}

// String coverage for the covariance enum.
func TestGMMCovarianceString(t *testing.T) {
	if GMMDiag.String() != "diag" || GMMFull.String() != "full" {
		t.Errorf("unexpected strings: %q %q", GMMDiag, GMMFull)
	}
	if GMMCovariance(9).String() == "" {
		t.Error("unknown covariance should still render")
	}
}
