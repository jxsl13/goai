package classic

import (
	"fmt"
	"math"
	"runtime"
	"sort"
)

// KNNMetric selects the distance function a k-nearest-neighbours estimator uses
// to measure how far apart two feature vectors are (Cover & Hart, "Nearest
// Neighbor Pattern Classification", IEEE Trans. Information Theory 1967, §C18).
type KNNMetric int

const (
	// KNNEuclidean is the straight-line (L2) distance √Σ(aᵢ−bᵢ)² — the default,
	// matching scikit-learn's metric="minkowski" with p=2.
	KNNEuclidean KNNMetric = iota
	// KNNManhattan is the city-block (L1) distance Σ|aᵢ−bᵢ| (scikit-learn p=1).
	KNNManhattan
)

// KNNWeights selects how the k neighbours are combined into a prediction.
type KNNWeights int

const (
	// KNNUniform gives every one of the k neighbours an equal vote (the default,
	// matching scikit-learn weights="uniform").
	KNNUniform KNNWeights = iota
	// KNNDistance weights each neighbour by the inverse of its distance, so closer
	// neighbours count more (scikit-learn weights="distance"). A query that
	// coincides exactly with training points (distance 0) is decided by those
	// exact matches alone.
	KNNDistance
)

// KNNOption configures a [KNNClassifier] or [KNNRegressor]. Options follow the
// functional-options idiom (§C12) and are prefixed WithKNN* so they never
// collide with the other classic estimators' option namespaces.
type KNNOption func(*knnConfig)

// knnConfig holds the shared hyper-parameters of a k-NN estimator.
type knnConfig struct {
	k       int
	weights KNNWeights
	metric  KNNMetric
}

func defaultKNNConfig() knnConfig {
	return knnConfig{k: 5, weights: KNNUniform, metric: KNNEuclidean}
}

// WithKNNK sets the number of nearest neighbours k consulted per prediction
// (default 5, matching scikit-learn's n_neighbors). k must be ≥1 and ≤ the
// number of training samples; values <1 are clamped to 1.
func WithKNNK(k int) KNNOption {
	return func(c *knnConfig) {
		if k < 1 {
			k = 1
		}
		c.k = k
	}
}

// WithKNNWeights selects uniform (default) or inverse-distance neighbour
// weighting. See [KNNWeights].
func WithKNNWeights(w KNNWeights) KNNOption { return func(c *knnConfig) { c.weights = w } }

// WithKNNMetric selects the distance function ([KNNEuclidean] default or
// [KNNManhattan]). See [KNNMetric].
func WithKNNMetric(m KNNMetric) KNNOption { return func(c *knnConfig) { c.metric = m } }

// ballMetricOf maps a KNNMetric onto the ball tree's metric enum.
func ballMetricOf(m KNNMetric) ballMetric {
	if m == KNNManhattan {
		return ballL1
	}
	return ballL2
}

// knnDist returns the distance between a and b under the given metric.
func knnDist(a, b []float64, metric KNNMetric) float64 {
	switch metric {
	case KNNManhattan:
		var s float64
		//perfscan:ignore PS3010 knnDist: tiny-n brute-force fallback (n<=leafSize); memory-streaming distance
		for i := range a {
			s += math.Abs(a[i] - b[i])
		}
		return s
	default: // KNNEuclidean
		var s float64
		for i := range a {
			d := a[i] - b[i]
			s += d * d
		}
		return math.Sqrt(s)
	}
}

// neighbour is a (distance, training-index) pair used to rank candidates.
type neighbour struct {
	dist float64
	idx  int
}

// nearest returns the cfg.k nearest training rows to row, ranked by ascending
// distance with ties broken by ascending training index ("first-seen", the
// deterministic reduction of scikit-learn's stable neighbour order). It assumes
// Fit has validated that k ≤ len(x).
func nearest(x [][]float64, row []float64, cfg knnConfig) []neighbour {
	cand := make([]neighbour, len(x))
	for i := range x {
		//perfscan:ignore PS3073 nearest() only runs on tiny-n fallback; ball-tree is prod path
		cand[i] = neighbour{dist: knnDist(x[i], row, cfg.metric), idx: i}
	}
	//perfscan:ignore PS3002,PS3006,PS6009 sort in nearest(): tiny-n fallback, dim-sized slice not vocab | sort-all-take-k but tiny-n (n<=leafSize) fallb
	sort.Slice(cand, func(a, b int) bool {
		if cand[a].dist != cand[b].dist {
			return cand[a].dist < cand[b].dist
		}
		return cand[a].idx < cand[b].idx
	})
	return cand[:cfg.k]
}

// knnWeights turns the chosen neighbours' distances into combination weights,
// mirroring scikit-learn's _get_weights: uniform ⇒ all 1; distance ⇒ 1/dist,
// except that when any neighbour sits at distance 0 those exact matches take
// weight 1 and all others weight 0 (avoids a division by zero and reproduces
// sklearn's exact-match rule).
func knnWeightsInto(nb []neighbour, w KNNWeights, s *knnScratch) []float64 {
	if cap(s.w) < len(nb) {
		s.w = make([]float64, len(nb))
	}
	out := s.w[:len(nb)]
	if w == KNNUniform {
		for i := range out {
			out[i] = 1
		}
		return out
	}
	exact := false
	for _, n := range nb {
		if n.dist == 0 {
			exact = true
			break
		}
	}
	for i, n := range nb {
		switch {
		case exact && n.dist == 0:
			out[i] = 1
		case exact:
			out[i] = 0
		default:
			out[i] = 1 / n.dist
		}
	}
	return out
}

// --- classifier ---------------------------------------------------------------

// KNNClassifier is a k-nearest-neighbours classifier (Cover & Hart 1967, §C18).
// Fit simply stores the training set (k-NN is a lazy learner — there is no
// model to fit); Predict finds the k closest training rows to each query,
// tallies their class votes and returns the winner.
//
// Ties in the majority vote break to the lowest class label — the deterministic
// reduction of numpy.argmax over scikit-learn's sorted classes_, so predictions
// match scikit-learn's KNeighborsClassifier exactly on tie-free data.
//
// # For AI practitioners
//
// Construct with [NewKNNClassifier] and any WithKNN* options, then
// [KNNClassifier.Fit] then [KNNClassifier.Predict] (or [KNNClassifier.PredictProba]
// for the vote-share distribution). k trades bias against variance: k=1 memorises
// the training set (perfect train recall, high variance), while a larger k
// smooths the decision boundary. Pick the distance with [WithKNNMetric] and
// inverse-distance voting with [WithKNNWeights].
//
// # For everyone else
//
// To label a new point, k-NN looks at the handful of training points nearest to
// it and lets them vote: the most common label among those neighbours wins.
// There is no training beyond remembering the examples — all the work happens at
// prediction time.
type KNNClassifier struct {
	cfg     knnConfig
	x       [][]float64
	yi      []int // class index per training row (into classes)
	classes []int // sorted distinct labels
	nFeat   int
	tree    *ballTree // spatial index over x (nil ⇒ brute-force fallback for tiny n)
	fitted  bool
}

// NewKNNClassifier constructs an unfitted classifier. Defaults match
// scikit-learn: k=5, uniform weights, Euclidean distance.
func NewKNNClassifier(opts ...KNNOption) *KNNClassifier {
	cfg := defaultKNNConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return &KNNClassifier{cfg: cfg}
}

// Fit stores the training set X[n][d] with integer labels y[n]. The sorted
// distinct labels define the classes and Predict returns labels from that set.
// It errors on empty, ragged or mismatched input, or when k exceeds n.
func (m *KNNClassifier) Fit(x [][]float64, y []int) error {
	d, err := validateXY(x, len(x), len(y))
	if err != nil {
		return err
	}
	if m.cfg.k > len(x) {
		return fmt.Errorf("classic: knn k=%d exceeds n=%d training samples", m.cfg.k, len(x))
	}
	classes, yi := encodeLabels(y)
	m.x = x
	m.yi = yi
	m.classes = classes
	m.nFeat = d
	m.tree = buildBallTree(x, ballMetricOf(m.cfg.metric))
	m.fitted = true
	return nil
}

// knn returns the k nearest training rows to row using the ball-tree index when
// one was built, falling back to the brute-force scan for tiny training sets.
// Both paths produce the identical (dist, idx)-ordered neighbour set.
func (m *KNNClassifier) knnInto(row []float64, s *knnScratch) []neighbour {
	if m.tree != nil {
		return m.tree.kNNInto(row, m.cfg.k, &s.h)
	}
	return nearest(m.x, row, m.cfg)
}

// vote returns the per-class summed neighbour weights for one query row.
// knnParallelRows runs body over the n query rows chunk-parallel across GOMAXPROCS. Each
// query reads the immutable fitted ball tree / training data and allocates its own result
// (the tree is concurrency-safe — the same property DBSCAN.Fit relies on), and body writes
// only its own out index, so the result is bit-identical to the serial loop. Serial below a
// small work threshold. Callers pre-validate row widths.
//
//perfscan:ignore PS6021 classic-ML Predict, per-row small alloc; not DL hot path
func knnParallelRows(n int, body func(i int)) {
	knnParallelChunks(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			body(i)
		}
	})
}

// knnParallelChunks is the chunking knnParallelRows and knnParallelScratch share. It hands
// each worker its whole [lo, hi) range rather than one index at a time, which is what lets a
// query loop allocate its buffers ONCE per chunk instead of once per row.
//
//perfscan:ignore PS3048,PS3061 Predict per-row vote/argmax over few classes; ball-tree routed | Predict per-row small work; not inference hot
func knnParallelChunks(n int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw <= 1 || n < 64 {
		body(0, n)
		return
	}
	if nw > n {
		nw = n
	}
	csz := (n + nw - 1) / nw
	_ = parallelBuild(nw, func(c int) error {
		lo := c * csz
		hi := lo + csz
		if hi > n {
			hi = n
		}
		body(lo, hi)
		return nil
	})
}

// knnScratch is the buffer set one worker reuses across every query in its chunk: the k-best
// heap's backing array, the neighbour weights, and the per-class vote sums.
//
// A query needs all three and none of them outlive it, so allocating them per row put nine
// allocations on the path of every single prediction. Reusing them is invisible to the result
// because each worker owns its scratch and every buffer is fully overwritten before it is read
// — the heap is truncated to zero length, the weights are written for all len(nb) entries, and
// the scores are explicitly cleared.
type knnScratch struct {
	h      knnHeap
	w      []float64
	scores []float64
}

// classScores returns a cleared per-class accumulator of width n.
func (s *knnScratch) classScores(n int) []float64 {
	if cap(s.scores) < n {
		s.scores = make([]float64, n)
	}
	s.scores = s.scores[:n]
	clear(s.scores)
	return s.scores
}

// knnParallelScratch runs body over the n query rows chunk-parallel, giving each worker its own
// scratch. Bit-identical to the serial loop: the scratch is worker-private and body writes only
// its own out index.
func knnParallelScratch(n int, body func(i int, s *knnScratch)) {
	knnParallelChunks(n, func(lo, hi int) {
		var s knnScratch
		for i := lo; i < hi; i++ {
			body(i, &s)
		}
	})
}

// voteInto returns the per-class summed neighbour weights for one query row, using s for every
// intermediate. The returned slice is s's own buffer and is valid only until the next call.
func (m *KNNClassifier) voteInto(row []float64, s *knnScratch) []float64 {
	nb := m.knnInto(row, s)
	w := knnWeightsInto(nb, m.cfg.weights, s)
	scores := s.classScores(len(m.classes))
	for i, n := range nb {
		scores[m.yi[n.idx]] += w[i]
	}
	return scores
}

// Predict returns the predicted label for each row of X. It errors if called
// before Fit or if a row's width does not match the training data.
func (m *KNNClassifier) Predict(x [][]float64) ([]int, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: KNNClassifier.Predict before Fit")
	}
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeat)
		}
	}
	out := make([]int, len(x))
	knnParallelScratch(len(x), func(i int, s *knnScratch) {
		scores := m.voteInto(x[i], s)
		best, bestScore := 0, math.Inf(-1)
		for c, s := range scores {
			if s > bestScore { // strict ⇒ lowest label wins ties
				bestScore, best = s, c
			}
		}
		out[i] = m.classes[best]
	})
	return out, nil
}

// PredictProba returns the vote-share distribution over classes for each row of
// X: the summed neighbour weights per class, normalised to sum to 1 (matching
// scikit-learn's predict_proba). It errors if called before Fit or on a width
// mismatch.
func (m *KNNClassifier) PredictProba(x [][]float64) ([][]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: KNNClassifier.PredictProba before Fit")
	}
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeat)
		}
	}
	out := make([][]float64, len(x))
	knnParallelScratch(len(x), func(i int, s *knnScratch) {
		scores := m.voteInto(x[i], s)
		var sum float64
		for _, s := range scores {
			sum += s
		}
		//perfscan:ignore PS6008 Regressor.Predict guard/alloc; classic-ML small per-row
		p := make([]float64, len(scores))
		if sum > 0 {
			for c, s := range scores {
				p[c] = s / sum
			}
		}
		out[i] = p
	})
	return out, nil
}

// Classes returns a copy of the sorted distinct labels seen during Fit; column j
// of [KNNClassifier.PredictProba] corresponds to Classes()[j].
func (m *KNNClassifier) Classes() []int { return append([]int(nil), m.classes...) }

// --- regressor ----------------------------------------------------------------

// KNNRegressor is a k-nearest-neighbours regressor (Cover & Hart 1967, §C18).
// Fit stores the training set; Predict averages the target values of the k
// closest training rows — a plain mean under [KNNUniform], or an
// inverse-distance-weighted mean under [KNNDistance]. Matches scikit-learn's
// KNeighborsRegressor.
//
// # For AI practitioners
//
// Construct with [NewKNNRegressor] and any WithKNN* options, then
// [KNNRegressor.Fit] then [KNNRegressor.Predict]. As with the classifier, k=1
// interpolates the training targets exactly while larger k smooths the surface.
//
// # For everyone else
//
// To predict a number for a new point, k-NN averages the values of the nearest
// known points. Nothing is learned in advance — the neighbours are looked up
// fresh for each query.
type KNNRegressor struct {
	cfg    knnConfig
	x      [][]float64
	y      []float64
	nFeat  int
	tree   *ballTree // spatial index over x (nil ⇒ brute-force fallback for tiny n)
	fitted bool
}

// NewKNNRegressor constructs an unfitted regressor. Defaults match scikit-learn:
// k=5, uniform weights, Euclidean distance.
func NewKNNRegressor(opts ...KNNOption) *KNNRegressor {
	cfg := defaultKNNConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return &KNNRegressor{cfg: cfg}
}

// Fit stores the training set X[n][d] with real targets y[n]. It errors on
// empty, ragged or mismatched input, or when k exceeds n.
func (m *KNNRegressor) Fit(x [][]float64, y []float64) error {
	d, err := validateXY(x, len(x), len(y))
	if err != nil {
		return err
	}
	if m.cfg.k > len(x) {
		return fmt.Errorf("classic: knn k=%d exceeds n=%d training samples", m.cfg.k, len(x))
	}
	m.x = x
	m.y = y
	m.nFeat = d
	m.tree = buildBallTree(x, ballMetricOf(m.cfg.metric))
	m.fitted = true
	return nil
}

// knnReg returns the k nearest training rows to row using the ball-tree index
// when built, else the brute-force scan — identical neighbour sets either way.
func (m *KNNRegressor) knnReg(row []float64, s *knnScratch) []neighbour {
	if m.tree != nil {
		return m.tree.kNNInto(row, m.cfg.k, &s.h)
	}
	return nearest(m.x, row, m.cfg)
}

// Predict returns the predicted value for each row of X. It errors if called
// before Fit or if a row's width does not match the training data.
func (m *KNNRegressor) Predict(x [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: KNNRegressor.Predict before Fit")
	}
	for i, row := range x {
		if len(row) != m.nFeat {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeat)
		}
	}
	out := make([]float64, len(x))
	knnParallelScratch(len(x), func(i int, s *knnScratch) {
		nb := m.knnReg(x[i], s)
		w := knnWeightsInto(nb, m.cfg.weights, s)
		var num, den float64
		//perfscan:ignore PS3010 line out of range (file 387 lines); stale finding
		for j, n := range nb {
			num += w[j] * m.y[n.idx]
			den += w[j]
		}
		out[i] = num / den
	})
	return out, nil
}
