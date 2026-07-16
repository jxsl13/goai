package classic

import (
	"fmt"
	"math"
)

// DBSCANMetric selects the distance function DBSCAN uses to decide whether two
// samples lie within eps of each other.
type DBSCANMetric int

const (
	// DBSCANEuclidean is the straight-line (L2) distance ‖x−y‖₂ — the default,
	// matching scikit-learn's default metric.
	DBSCANEuclidean DBSCANMetric = iota
	// DBSCANManhattan is the city-block (L1) distance Σ|xᵢ−yᵢ|.
	DBSCANManhattan
)

// String renders the metric using scikit-learn's names ("euclidean",
// "manhattan").
func (m DBSCANMetric) String() string {
	switch m {
	case DBSCANEuclidean:
		return "euclidean"
	case DBSCANManhattan:
		return "manhattan"
	default:
		return fmt.Sprintf("DBSCANMetric(%d)", int(m))
	}
}

// DBSCANLabelNoise is the label assigned to points that belong to no cluster
// (density-unreachable "noise" points), matching scikit-learn's −1 convention.
const DBSCANLabelNoise = -1

// DBSCANOption configures a [DBSCAN] run. Options are prefixed WithDBSCAN* to
// keep the estimator's configuration namespace distinct from the other classic
// models.
type DBSCANOption func(*dbscanConfig)

type dbscanConfig struct {
	eps        float64      // neighborhood radius
	minSamples int          // core-point neighbor threshold (self included)
	metric     DBSCANMetric // distance function
}

func defaultDBSCANConfig() dbscanConfig {
	return dbscanConfig{eps: 0.5, minSamples: 5, metric: DBSCANEuclidean}
}

// ballMetricOfDBSCAN maps a DBSCANMetric onto the ball tree's metric enum.
func ballMetricOfDBSCAN(m DBSCANMetric) ballMetric {
	if m == DBSCANManhattan {
		return ballL1
	}
	return ballL2
}

// WithDBSCANEps sets the neighborhood radius eps (default 0.5): two samples are
// neighbors when their distance is ≤ eps. It is the most important knob — larger
// eps merges points into fewer, bigger clusters. eps must be > 0.
func WithDBSCANEps(eps float64) DBSCANOption { return func(c *dbscanConfig) { c.eps = eps } }

// WithDBSCANMinSamples sets the minimum neighborhood size (the point itself
// included) for a sample to be a core point (default 5, matching sklearn).
// Larger values demand denser regions and label more points as noise.
// minSamples must be ≥ 1.
func WithDBSCANMinSamples(n int) DBSCANOption { return func(c *dbscanConfig) { c.minSamples = n } }

// WithDBSCANMetric selects the distance metric (default [DBSCANEuclidean]).
func WithDBSCANMetric(m DBSCANMetric) DBSCANOption { return func(c *dbscanConfig) { c.metric = m } }

// DBSCAN is Density-Based Spatial Clustering of Applications with Noise (Ester,
// Kriegel, Sander & Xu, "A Density-Based Algorithm for Discovering Clusters in
// Large Spatial Databases with Noise", KDD 1996). It groups points that are
// packed closely together and marks points in low-density regions as noise,
// discovering the number of clusters on its own — no K is required.
//
// Two hyper-parameters define density: a radius eps and a count minSamples. A
// point is a *core point* when at least minSamples points (itself included) lie
// within eps of it. Clusters are the connected components of the core points
// under eps-reachability; a non-core point within eps of a core point is a
// *border point* and joins that core's cluster; everything else is *noise*
// (label [DBSCANLabelNoise]).
//
// # For AI practitioners
//
// DBSCAN is the go-to when clusters are non-convex or of unequal size and you
// don't know how many there are — cases where [KMeans] and [GaussianMixture]
// (which assume roughly blob-shaped clusters and a fixed K) struggle. It also
// isolates outliers as noise. Construct with [NewDBSCAN] and WithDBSCAN*
// options, then call [DBSCAN.Fit] to get integer labels (contiguous 0,1,2,… for
// clusters, −1 for noise) and [DBSCAN.CoreSampleIndices] for the core points.
// The result is deterministic given (eps, minSamples, metric, point order),
// matching scikit-learn's labels up to a renumbering of cluster ids. Cost is
// O(n²) distance evaluations (no spatial index). Tuning is mostly eps; a common
// heuristic sets minSamples ≥ d+1 for d-dimensional data.
//
// # For everyone else
//
// DBSCAN finds clusters by looking for crowds: wherever many points sit close
// together it grows a cluster outward through the crowd, and points stranded in
// sparse space are called noise instead of being forced into a group. Unlike
// methods that need you to say "make 3 clusters", it figures out how many
// crowds there are, and it happily finds long, curvy clusters — not just round
// blobs.
type DBSCAN struct {
	cfg dbscanConfig

	labels []int  // per-sample cluster id (−1 = noise)
	core   []bool // per-sample core-point flag
	fitted bool
}

// NewDBSCAN constructs an unfitted DBSCAN estimator with the given options
// (defaults: eps=0.5, minSamples=5, euclidean metric).
func NewDBSCAN(opts ...DBSCANOption) *DBSCAN {
	cfg := defaultDBSCANConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return &DBSCAN{cfg: cfg}
}

// Fit clusters X[n][d] and returns per-sample labels: contiguous cluster ids
// 0,1,2,… assigned in order of discovery, and [DBSCANLabelNoise] (−1) for noise
// points. The returned slice aliases the estimator's internal state; the same
// labels are available afterwards via [DBSCAN.Labels]. Fit errors on empty or
// ragged input, eps ≤ 0, or minSamples < 1.
func (m *DBSCAN) Fit(x [][]float64) ([]int, error) {
	n := len(x)
	if n == 0 {
		return nil, fmt.Errorf("classic: dbscan needs at least one sample")
	}
	if m.cfg.eps <= 0 {
		return nil, fmt.Errorf("classic: dbscan eps must be > 0, got %g", m.cfg.eps)
	}
	if m.cfg.minSamples < 1 {
		return nil, fmt.Errorf("classic: dbscan minSamples must be ≥ 1, got %d", m.cfg.minSamples)
	}
	d := len(x[0])
	for _, row := range x {
		if len(row) != d {
			return nil, fmt.Errorf("classic: dbscan ragged X")
		}
	}

	// Precompute eps-neighborhoods (self included) and core flags. A ball-tree
	// radius query replaces the brute-force O(n²) scan; it returns the identical
	// neighbour set (ascending index order, self included) so labels are
	// unchanged. For tiny n the tree is not built and we fall back to the scan.
	eps2 := m.cfg.eps * m.cfg.eps
	tree := buildBallTree(x, ballMetricOfDBSCAN(m.cfg.metric))
	neighbors := make([][]int, n)
	m.core = make([]bool, n)
	for i := range n {
		var nb []int
		if tree != nil {
			nb = tree.radius(x[i], m.cfg.eps, nil) // nil ⇒ a fresh slice per query
		} else {
			for j := range n {
				if m.dist(x[i], x[j], eps2) {
					nb = append(nb, j)
				}
			}
		}
		neighbors[i] = nb
		m.core[i] = len(nb) >= m.cfg.minSamples
	}

	// Cluster expansion — a faithful port of scikit-learn's dbscan_inner: scan
	// points in index order; each unlabeled core point seeds a new cluster and a
	// depth-first flood over eps-reachable core points. Border points are
	// labeled when first reached but do not expand further.
	labels := make([]int, n)
	for i := range labels {
		labels[i] = DBSCANLabelNoise
	}
	label := 0
	var stack []int
	for i := range n {
		if labels[i] != DBSCANLabelNoise || !m.core[i] {
			continue
		}
		cur := i
		for {
			if labels[cur] == DBSCANLabelNoise {
				labels[cur] = label
				if m.core[cur] {
					for _, j := range neighbors[cur] {
						if labels[j] == DBSCANLabelNoise {
							stack = append(stack, j)
						}
					}
				}
			}
			if len(stack) == 0 {
				break
			}
			cur = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		}
		label++
	}
	m.labels = labels
	m.fitted = true
	return labels, nil
}

// dist reports whether the distance between a and b is within eps, i.e. ≤ eps
// (comparing against eps² for the euclidean metric to avoid a square root).
func (m *DBSCAN) dist(a, b []float64, eps2 float64) bool {
	switch m.cfg.metric {
	case DBSCANManhattan:
		var s float64
		for i := range a {
			s += math.Abs(a[i] - b[i])
		}
		return s <= m.cfg.eps
	default: // euclidean
		var s float64
		for i := range a {
			dv := a[i] - b[i]
			s += dv * dv
		}
		return s <= eps2
	}
}

// Labels returns the fitted per-sample cluster labels (−1 = noise). It panics
// if called before [DBSCAN.Fit].
func (m *DBSCAN) Labels() []int {
	if !m.fitted {
		panic("classic: DBSCAN.Labels before Fit")
	}
	return m.labels
}

// CoreSampleIndices returns the indices of the core points (samples with at
// least minSamples neighbors within eps, themselves included), in ascending
// order — the analogue of scikit-learn's core_sample_indices_. It panics if
// called before [DBSCAN.Fit].
func (m *DBSCAN) CoreSampleIndices() []int {
	if !m.fitted {
		panic("classic: DBSCAN.CoreSampleIndices before Fit")
	}
	var out []int
	for i, c := range m.core {
		if c {
			out = append(out, i)
		}
	}
	return out
}

// NumClusters returns the number of clusters found (excluding noise). It panics
// if called before [DBSCAN.Fit].
func (m *DBSCAN) NumClusters() int {
	if !m.fitted {
		panic("classic: DBSCAN.NumClusters before Fit")
	}
	mx := -1
	for _, l := range m.labels {
		if l > mx {
			mx = l
		}
	}
	return mx + 1
}
