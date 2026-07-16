package classic

import (
	"fmt"
	"math"
)

// ForestOption configures a [RandomForestClassifier] or [RandomForestRegressor]
// (functional-options idiom, §C12).
type ForestOption func(*forestConfig)

// forestConfig holds the ensemble hyper-parameters plus the per-tree settings
// forwarded to each member.
type forestConfig struct {
	nTrees      int
	seed        int64
	maxFeatures int // features per split; 0 ⇒ √d (classify) or d/3 (regress)
	tree        cartConfig
}

func defaultForestConfig(regression bool) forestConfig {
	return forestConfig{nTrees: 100, seed: 0, maxFeatures: 0, tree: defaultTreeConfig(regression)}
}

// WithNumTrees sets the number of trees in the ensemble (default 100). Values <1
// are clamped to 1.
func WithNumTrees(n int) ForestOption {
	return func(c *forestConfig) {
		if n < 1 {
			n = 1
		}
		c.nTrees = n
	}
}

// WithSeed sets the random seed controlling bootstrap sampling and per-split
// feature subsampling, making Fit fully deterministic.
func WithSeed(s int64) ForestOption { return func(c *forestConfig) { c.seed = s } }

// WithMaxFeatures sets how many features are considered at each split. 0 (the
// default) selects √d for classification and d/3 for regression — the classic
// Breiman 2001 defaults (§C18, §C21). Values are clamped to [1, d] at Fit time.
func WithMaxFeatures(m int) ForestOption { return func(c *forestConfig) { c.maxFeatures = m } }

// WithForestMaxDepth caps the depth of every tree in the forest (see
// [WithMaxDepth]). Default unlimited.
func WithForestMaxDepth(d int) ForestOption { return func(c *forestConfig) { c.tree.maxDepth = d } }

// WithForestMinSamplesLeaf sets the per-leaf minimum for every tree (see
// [WithMinSamplesLeaf]). Default 1.
func WithForestMinSamplesLeaf(n int) ForestOption {
	return func(c *forestConfig) {
		if n < 1 {
			n = 1
		}
		c.tree.minSamplesLeaf = n
	}
}

// WithForestCriterion selects the classification impurity for every tree
// ([Gini] or [Entropy]); ignored by the regressor. See [WithCriterion].
func WithForestCriterion(cr Criterion) ForestOption {
	return func(c *forestConfig) { c.tree.criterion = cr }
}

// resolveMaxFeatures returns the per-split feature count, applying the default
// (√d classify / d/3 regress) and clamping to [1, d].
func resolveMaxFeatures(m, d int, regression bool) int {
	if m <= 0 {
		if regression {
			m = d / 3
		} else {
			m = int(math.Sqrt(float64(d)))
		}
	}
	if m < 1 {
		m = 1
	}
	if m > d {
		m = d
	}
	return m
}

// bootstrap draws n indices in [0,n) with replacement using the given LCG.
func bootstrap(n int, rng *lcg) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = rng.intn(n)
	}
	return out
}

// --- classifier ---------------------------------------------------------------

// RandomForestClassifier is an ensemble of CART classification trees (Breiman,
// "Random Forests", 2001, §C18). Each tree is fit on a bootstrap sample of the
// rows (sampled with replacement) and considers only a random √d-sized subset of
// features at each split; Predict takes a majority vote across the trees.
//
// Bagging plus feature subsampling decorrelates the trees, so the averaged
// prediction has lower variance than a single deep tree on noisy data. Fitting
// is deterministic given the seed ([WithSeed]).
type RandomForestClassifier struct {
	trees    []*DecisionTreeClassifier
	classes  []int
	nFeature int
	cfg      forestConfig
}

// NewRandomForestClassifier constructs an unfitted forest. Defaults: 100 trees,
// √d features per split, [Gini] impurity, unlimited depth. Configure with
// [WithNumTrees], [WithSeed], [WithMaxFeatures], [WithForestMaxDepth],
// [WithForestMinSamplesLeaf] and [WithForestCriterion].
func NewRandomForestClassifier(opts ...ForestOption) *RandomForestClassifier {
	cfg := defaultForestConfig(false)
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.tree.criterion == MSE {
		cfg.tree.criterion = Gini
	}
	return &RandomForestClassifier{cfg: cfg}
}

// Fit trains the ensemble on X[n][d] and integer labels y[n]. Errors on empty or
// mismatched input.
func (m *RandomForestClassifier) Fit(x [][]float64, y []int) error {
	d, err := validateXY(x, len(x), len(y))
	if err != nil {
		return err
	}
	classes, _ := encodeLabels(y)
	mf := resolveMaxFeatures(m.cfg.maxFeatures, d, false)
	rng := &lcg{state: seedState(m.cfg.seed)}
	n := len(x)
	m.trees = make([]*DecisionTreeClassifier, m.cfg.nTrees)
	for t := range m.trees {
		sample := bootstrap(n, rng)
		bx, by := gatherInt(x, y, sample)
		tree := NewDecisionTreeClassifier(
			WithMaxDepth(m.cfg.tree.maxDepth),
			WithMinSamplesSplit(m.cfg.tree.minSamplesSplit),
			WithMinSamplesLeaf(m.cfg.tree.minSamplesLeaf),
			WithCriterion(m.cfg.tree.criterion),
			withMaxFeatures(mf),
		)
		// each tree gets its own deterministic split RNG stream
		if err := tree.fitWithSeed(bx, by, classes, rng.next()); err != nil {
			return err
		}
		m.trees[t] = tree
	}
	m.classes = classes
	m.nFeature = d
	return nil
}

// Predict returns the majority-vote label for each row of X. Ties break to the
// lowest label. Errors before Fit or on a width mismatch.
func (m *RandomForestClassifier) Predict(x [][]float64) ([]int, error) {
	if m.trees == nil {
		return nil, fmt.Errorf("classic: RandomForestClassifier.Predict before Fit")
	}
	out := make([]int, len(x))
	votes := make([]int, len(m.classes))
	pos := make(map[int]int, len(m.classes))
	for i, c := range m.classes {
		pos[c] = i
	}
	for i, row := range x {
		if len(row) != m.nFeature {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeature)
		}
		for k := range votes {
			votes[k] = 0
		}
		for _, tree := range m.trees {
			lab, err := tree.Predict([][]float64{row})
			if err != nil {
				return nil, err
			}
			votes[pos[lab[0]]]++
		}
		best, bc := 0, -1
		for k, v := range votes {
			if v > bc {
				bc, best = v, k
			}
		}
		out[i] = m.classes[best]
	}
	return out, nil
}

// --- regressor ----------------------------------------------------------------

// RandomForestRegressor is an ensemble of CART regression trees (Breiman 2001,
// §C18). Each tree is fit on a bootstrap sample and considers a random d/3-sized
// feature subset at each split; Predict averages the tree outputs.
type RandomForestRegressor struct {
	trees    []*DecisionTreeRegressor
	nFeature int
	cfg      forestConfig
}

// NewRandomForestRegressor constructs an unfitted forest. Defaults: 100 trees,
// d/3 features per split, [MSE] impurity, unlimited depth. Configure with
// [WithNumTrees], [WithSeed], [WithMaxFeatures], [WithForestMaxDepth] and
// [WithForestMinSamplesLeaf].
func NewRandomForestRegressor(opts ...ForestOption) *RandomForestRegressor {
	cfg := defaultForestConfig(true)
	for _, o := range opts {
		o(&cfg)
	}
	cfg.tree.criterion = MSE
	return &RandomForestRegressor{cfg: cfg}
}

// Fit trains the ensemble on X[n][d] and real targets y[n]. Errors on empty or
// mismatched input.
func (m *RandomForestRegressor) Fit(x [][]float64, y []float64) error {
	d, err := validateXY(x, len(x), len(y))
	if err != nil {
		return err
	}
	mf := resolveMaxFeatures(m.cfg.maxFeatures, d, true)
	rng := &lcg{state: seedState(m.cfg.seed)}
	n := len(x)
	m.trees = make([]*DecisionTreeRegressor, m.cfg.nTrees)
	for t := range m.trees {
		sample := bootstrap(n, rng)
		bx, by := gatherFloat(x, y, sample)
		tree := NewDecisionTreeRegressor(
			WithMaxDepth(m.cfg.tree.maxDepth),
			WithMinSamplesSplit(m.cfg.tree.minSamplesSplit),
			WithMinSamplesLeaf(m.cfg.tree.minSamplesLeaf),
			withMaxFeatures(mf),
		)
		if err := tree.fitWithSeed(bx, by, rng.next()); err != nil {
			return err
		}
		m.trees[t] = tree
	}
	m.nFeature = d
	return nil
}

// Predict returns the averaged prediction across the trees for each row of X.
// Errors before Fit or on a width mismatch.
func (m *RandomForestRegressor) Predict(x [][]float64) ([]float64, error) {
	if m.trees == nil {
		return nil, fmt.Errorf("classic: RandomForestRegressor.Predict before Fit")
	}
	out := make([]float64, len(x))
	for i, row := range x {
		if len(row) != m.nFeature {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeature)
		}
		var s float64
		for _, tree := range m.trees {
			v, err := tree.Predict([][]float64{row})
			if err != nil {
				return nil, err
			}
			s += v[0]
		}
		out[i] = s / float64(len(m.trees))
	}
	return out, nil
}

// --- helpers ------------------------------------------------------------------

// seedState maps a user seed to a nonzero LCG state (xorshift needs nonzero).
func seedState(seed int64) uint64 {
	s := uint64(seed)*2862933555777941757 + 3037000493
	if s == 0 {
		s = 0x9e3779b97f4a7c15
	}
	return s
}

func gatherInt(x [][]float64, y []int, idx []int) ([][]float64, []int) {
	bx := make([][]float64, len(idx))
	by := make([]int, len(idx))
	for i, j := range idx {
		bx[i] = x[j]
		by[i] = y[j]
	}
	return bx, by
}

func gatherFloat(x [][]float64, y []float64, idx []int) ([][]float64, []float64) {
	bx := make([][]float64, len(idx))
	by := make([]float64, len(idx))
	for i, j := range idx {
		bx[i] = x[j]
		by[i] = y[j]
	}
	return bx, by
}
