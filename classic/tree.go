package classic

import (
	"fmt"
	"math"
	"sort"
)

// Criterion selects the impurity measure a decision tree greedily minimises at
// each split (Breiman et al., "Classification and Regression Trees", 1984, §C18).
//
//   - [Gini] and [Entropy] are for classification. Gini impurity is
//     1−Σₖ pₖ² and entropy is −Σₖ pₖ·ln pₖ over the class proportions pₖ; both are
//     minimised at a pure node. Gini is scikit-learn's default because it avoids
//     the logarithm; the two rarely disagree on the chosen split.
//   - [MSE] (mean-squared-error / variance) is for regression: the node impurity
//     is the variance of the targets, so a split is scored by how much it reduces
//     within-child variance.
type Criterion int

const (
	// Gini is the default classification criterion: impurity 1−Σₖ pₖ².
	Gini Criterion = iota
	// Entropy is the alternative classification criterion: impurity −Σₖ pₖ·ln pₖ.
	Entropy
	// MSE is the regression criterion: node impurity is the target variance.
	MSE
)

// featureThreshold mirrors scikit-learn's FEATURE_THRESHOLD: two consecutive
// sorted feature values closer than this are treated as equal, so no split
// candidate is placed between them (avoids splitting on float noise).
const featureThreshold = 1e-7

// TreeOption configures a [DecisionTreeClassifier] or [DecisionTreeRegressor].
// Options follow the functional-options idiom (§C12); pass any combination to
// the constructor.
type TreeOption func(*cartConfig)

// cartConfig holds the shared hyper-parameters of a CART tree.
type cartConfig struct {
	maxDepth        int       // maximum depth; <0 means unlimited (default)
	minSamplesSplit int       // minimum samples required to split an internal node
	minSamplesLeaf  int       // minimum samples required at each leaf
	criterion       Criterion // impurity measure
	maxFeatures     int       // features considered per split; <=0 means all
}

func defaultTreeConfig(regression bool) cartConfig {
	c := cartConfig{maxDepth: -1, minSamplesSplit: 2, minSamplesLeaf: 1, maxFeatures: 0}
	if regression {
		c.criterion = MSE
	} else {
		c.criterion = Gini
	}
	return c
}

// WithMaxDepth caps the tree depth. The root is depth 0, so d=0 yields a single
// leaf (a stump predicting the global majority/mean) and d<0 (the default)
// leaves depth unlimited — an unlimited tree grows until every leaf is pure
// (§C21). Matches scikit-learn's max_depth (None ⇒ unlimited).
func WithMaxDepth(d int) TreeOption { return func(c *cartConfig) { c.maxDepth = d } }

// WithMinSamplesSplit sets the minimum number of samples an internal node must
// hold to be eligible for splitting (scikit-learn default 2). Values <2 are
// clamped to 2.
func WithMinSamplesSplit(n int) TreeOption {
	return func(c *cartConfig) {
		if n < 2 {
			n = 2
		}
		c.minSamplesSplit = n
	}
}

// WithMinSamplesLeaf sets the minimum number of samples required at each leaf; a
// split is rejected unless both children meet it (scikit-learn default 1).
// Larger values regularise the tree. Values <1 are clamped to 1.
func WithMinSamplesLeaf(n int) TreeOption {
	return func(c *cartConfig) {
		if n < 1 {
			n = 1
		}
		c.minSamplesLeaf = n
	}
}

// WithCriterion selects the impurity measure. For classifiers use [Gini]
// (default) or [Entropy]; regressors always use [MSE] and ignore this option.
func WithCriterion(cr Criterion) TreeOption { return func(c *cartConfig) { c.criterion = cr } }

// withMaxFeatures (unexported) restricts each split to a random subset of
// features — used by the random forest to decorrelate trees. 0 means all.
func withMaxFeatures(m int) TreeOption { return func(c *cartConfig) { c.maxFeatures = m } }

// cartNode is one node of a fitted tree. Internal nodes carry a (feature,
// threshold) test; leaves carry a prediction.
type cartNode struct {
	leaf      bool
	feature   int     // split feature index (internal nodes)
	threshold float64 // samples with X[feature] <= threshold go left
	left      *cartNode
	right     *cartNode
	predClass int     // majority class index (classifier leaves)
	value     float64 // mean target (regressor leaves)
}

// splitFinder is a seeded random source used only when maxFeatures>0. Kept as a
// tiny LCG so the tree has no external dependency and stays deterministic.
type lcg struct{ state uint64 }

func (r *lcg) next() uint64 {
	// xorshift64* — small, fast, good enough for feature subsampling.
	r.state ^= r.state >> 12
	r.state ^= r.state << 25
	r.state ^= r.state >> 27
	return r.state * 2685821657736338717
}

func (r *lcg) intn(n int) int { return int(r.next() % uint64(n)) }

// nonzero returns s, or a fixed nonzero constant if s is 0 (xorshift requires a
// nonzero state).
func nonzero(s uint64) uint64 {
	if s == 0 {
		return 0x9e3779b97f4a7c15
	}
	return s
}

// cartBuilder carries the immutable inputs while a tree is being grown.
type cartBuilder struct {
	x          [][]float64
	yi         []int     // class indices (classification)
	yf         []float64 // targets (regression)
	nClasses   int
	regression bool
	cfg        cartConfig
	rng        *lcg
}

// checkX validates X shape against the builder's feature count and returns d.
func validateXY(x [][]float64, n, ylen int) (int, error) {
	if n == 0 || ylen != n {
		return 0, fmt.Errorf("classic: bad shapes n=%d len(y)=%d", n, ylen)
	}
	d := len(x[0])
	if d == 0 {
		return 0, fmt.Errorf("classic: X has zero features")
	}
	for _, row := range x {
		if len(row) != d {
			return 0, fmt.Errorf("classic: ragged X")
		}
	}
	return d, nil
}

// build grows the tree from the given sample indices at the given depth.
func (b *cartBuilder) build(idx []int, depth int) *cartNode {
	n := len(idx)
	node := &cartNode{leaf: true}
	if b.regression {
		node.value = b.mean(idx)
	} else {
		node.predClass = b.majority(idx)
	}

	// Stopping rules mirror scikit-learn's DepthFirstTreeBuilder (§C21).
	pure := b.impurity(idx) <= 1e-12
	if pure ||
		(b.cfg.maxDepth >= 0 && depth >= b.cfg.maxDepth) ||
		n < b.cfg.minSamplesSplit ||
		n < 2*b.cfg.minSamplesLeaf {
		return node
	}

	feat, thr, ok := b.bestSplit(idx)
	if !ok {
		return node // no admissible split (e.g. constant features)
	}

	left := make([]int, 0, n)
	right := make([]int, 0, n)
	for _, i := range idx {
		if b.x[i][feat] <= thr {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	node.leaf = false
	node.feature = feat
	node.threshold = thr
	node.left = b.build(left, depth+1)
	node.right = b.build(right, depth+1)
	return node
}

// candidateFeatures returns the feature indices considered at a split: all of
// them, or a random maxFeatures-sized subset (ascending, so the split tie-break
// still favours the lowest feature index among the sampled set).
func (b *cartBuilder) candidateFeatures(d int) []int {
	if b.cfg.maxFeatures <= 0 || b.cfg.maxFeatures >= d {
		feats := make([]int, d)
		for i := range feats {
			feats[i] = i
		}
		return feats
	}
	// partial Fisher–Yates draw of maxFeatures distinct indices
	pool := make([]int, d)
	for i := range pool {
		pool[i] = i
	}
	m := b.cfg.maxFeatures
	for i := 0; i < m; i++ {
		j := i + b.rng.intn(d-i)
		pool[i], pool[j] = pool[j], pool[i]
	}
	feats := append([]int(nil), pool[:m]...)
	sort.Ints(feats)
	return feats
}

// bestSplit finds the (feature, threshold) minimising the weighted child
// impurity. Ties break to the lowest feature index then lowest threshold — the
// deterministic reduction of scikit-learn's tie-break on unique-optimum data.
func (b *cartBuilder) bestSplit(idx []int) (feat int, thr float64, ok bool) {
	d := len(b.x[0])
	bestCost := math.Inf(1)
	minLeaf := b.cfg.minSamplesLeaf
	for _, f := range b.candidateFeatures(d) {
		order := append([]int(nil), idx...)
		sort.Slice(order, func(a, c int) bool { return b.x[order[a]][f] < b.x[order[c]][f] })
		cost, cut, found := b.sweep(order, f, minLeaf)
		if found && cost < bestCost {
			bestCost = cost
			feat = f
			a := b.x[order[cut-1]][f]
			c := b.x[order[cut]][f]
			thr = (a + c) / 2
			// guard against the midpoint rounding up to the right value
			if thr == c || math.IsInf(thr, 0) {
				thr = a
			}
			ok = true
		}
	}
	return feat, thr, ok
}

// sweep scans the sorted-by-feature order and returns the minimal weighted child
// impurity together with the left-child size (cut) achieving it.
func (b *cartBuilder) sweep(order []int, f, minLeaf int) (bestCost float64, cut int, found bool) {
	n := len(order)
	bestCost = math.Inf(1)
	if b.regression {
		var totSum, totSq float64
		for _, i := range order {
			v := b.yf[i]
			totSum += v
			totSq += v * v
		}
		var lSum, lSq float64
		for p := 1; p < n; p++ {
			v := b.yf[order[p-1]]
			lSum += v
			lSq += v * v
			if p < minLeaf || n-p < minLeaf {
				continue
			}
			if b.x[order[p]][f]-b.x[order[p-1]][f] <= featureThreshold {
				continue
			}
			nl := float64(p)
			nr := float64(n - p)
			costL := lSq - lSum*lSum/nl
			costR := (totSq - lSq) - (totSum-lSum)*(totSum-lSum)/nr
			if c := costL + costR; c < bestCost {
				bestCost, cut, found = c, p, true
			}
		}
		return bestCost, cut, found
	}
	// classification
	total := make([]int, b.nClasses)
	for _, i := range order {
		total[b.yi[i]]++
	}
	left := make([]int, b.nClasses)
	for p := 1; p < n; p++ {
		left[b.yi[order[p-1]]]++
		if p < minLeaf || n-p < minLeaf {
			continue
		}
		if b.x[order[p]][f]-b.x[order[p-1]][f] <= featureThreshold {
			continue
		}
		cost := b.weightedImpurityClf(left, p) + b.weightedImpurityClfComp(left, total, n-p)
		if cost < bestCost {
			bestCost, cut, found = cost, p, true
		}
	}
	return bestCost, cut, found
}

// weightedImpurityClf returns nL·impurity(left counts).
func (b *cartBuilder) weightedImpurityClf(counts []int, n int) float64 {
	return float64(n) * impurityCounts(counts, n, b.cfg.criterion)
}

// weightedImpurityClfComp returns nR·impurity(total−left counts) without
// allocating the right-count slice.
func (b *cartBuilder) weightedImpurityClfComp(left, total []int, n int) float64 {
	if n == 0 {
		return 0
	}
	var acc float64
	nf := float64(n)
	switch b.cfg.criterion {
	case Entropy:
		for k := range total {
			c := total[k] - left[k]
			if c > 0 {
				p := float64(c) / nf
				acc -= p * math.Log(p)
			}
		}
	default: // Gini
		var sq float64
		for k := range total {
			c := float64(total[k] - left[k])
			sq += c * c
		}
		acc = 1 - sq/(nf*nf)
	}
	return nf * acc
}

// impurityCounts computes the impurity of a set given its class counts.
func impurityCounts(counts []int, n int, cr Criterion) float64 {
	if n == 0 {
		return 0
	}
	nf := float64(n)
	if cr == Entropy {
		var e float64
		for _, c := range counts {
			if c > 0 {
				p := float64(c) / nf
				e -= p * math.Log(p)
			}
		}
		return e
	}
	var sq float64
	for _, c := range counts {
		cf := float64(c)
		sq += cf * cf
	}
	return 1 - sq/(nf*nf)
}

// impurity returns the node impurity over sample indices (used for the pure-node
// stopping rule).
func (b *cartBuilder) impurity(idx []int) float64 {
	if b.regression {
		m := b.mean(idx)
		var v float64
		for _, i := range idx {
			d := b.yf[i] - m
			v += d * d
		}
		return v / float64(len(idx))
	}
	counts := make([]int, b.nClasses)
	for _, i := range idx {
		counts[b.yi[i]]++
	}
	return impurityCounts(counts, len(idx), b.cfg.criterion)
}

func (b *cartBuilder) mean(idx []int) float64 {
	var s float64
	for _, i := range idx {
		s += b.yf[i]
	}
	return s / float64(len(idx))
}

// majority returns the argmax class index, ties broken to the lowest index
// (matching numpy.argmax / scikit-learn).
func (b *cartBuilder) majority(idx []int) int {
	counts := make([]int, b.nClasses)
	for _, i := range idx {
		counts[b.yi[i]]++
	}
	best, bc := 0, -1
	for k, c := range counts {
		if c > bc {
			bc, best = c, k
		}
	}
	return best
}

// predict routes a single row down the fitted tree.
func (n *cartNode) predict(row []float64) *cartNode {
	for !n.leaf {
		if row[n.feature] <= n.threshold {
			n = n.left
		} else {
			n = n.right
		}
	}
	return n
}

// --- public classifier --------------------------------------------------------

// DecisionTreeClassifier is a CART decision tree for classification (Breiman
// 1984, §C18). Fit greedily splits on the (feature, threshold) that most reduces
// [Gini] (default) or [Entropy] impurity until a stopping rule fires; Predict
// routes each row to a leaf and returns its majority class.
//
// Splits use the rule X[feature] ≤ threshold ⇒ left child, with the threshold
// at the midpoint of adjacent sorted feature values — matching scikit-learn.
type DecisionTreeClassifier struct {
	root     *cartNode
	classes  []int // sorted distinct labels; leaf.predClass indexes into this
	nFeature int
	cfg      cartConfig
}

// NewDecisionTreeClassifier constructs an unfitted classifier. Defaults match
// scikit-learn: [Gini] impurity, unlimited depth, minSamplesSplit=2,
// minSamplesLeaf=1. Configure with [WithMaxDepth], [WithMinSamplesLeaf],
// [WithMinSamplesSplit] and [WithCriterion].
func NewDecisionTreeClassifier(opts ...TreeOption) *DecisionTreeClassifier {
	cfg := defaultTreeConfig(false)
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.criterion == MSE {
		cfg.criterion = Gini // MSE is regression-only; fall back to the default
	}
	return &DecisionTreeClassifier{cfg: cfg}
}

// Fit grows the tree from X[n][d] and integer labels y[n]. Labels may be any
// non-negative-or-negative integers; the sorted distinct set defines the classes
// and Predict returns labels from that set. Errors on empty or mismatched input.
func (m *DecisionTreeClassifier) Fit(x [][]float64, y []int) error {
	classes, _ := encodeLabels(y)
	return m.fitWithSeed(x, y, classes, 0x9e3779b97f4a7c15)
}

// fitWithSeed grows the classifier using a caller-supplied class set (so every
// tree in a forest shares one label→index space even when a bootstrap sample
// omits a class) and split RNG seed. Used by RandomForestClassifier.
func (m *DecisionTreeClassifier) fitWithSeed(x [][]float64, y, classes []int, seed uint64) error {
	d, err := validateXY(x, len(x), len(y))
	if err != nil {
		return err
	}
	pos := make(map[int]int, len(classes))
	for i, v := range classes {
		pos[v] = i
	}
	yi := make([]int, len(y))
	for i, v := range y {
		p, ok := pos[v]
		if !ok {
			return fmt.Errorf("classic: label %d not in class set", v)
		}
		yi[i] = p
	}
	b := &cartBuilder{x: x, yi: yi, nClasses: len(classes), cfg: m.cfg, rng: &lcg{state: nonzero(seed)}}
	idx := make([]int, len(x))
	for i := range idx {
		idx[i] = i
	}
	m.root = b.build(idx, 0)
	m.classes = classes
	m.nFeature = d
	return nil
}

// Predict returns the predicted label for each row of X. It errors if called
// before Fit or if a row's width does not match the training data.
func (m *DecisionTreeClassifier) Predict(x [][]float64) ([]int, error) {
	if m.root == nil {
		return nil, fmt.Errorf("classic: DecisionTreeClassifier.Predict before Fit")
	}
	out := make([]int, len(x))
	for i, row := range x {
		if len(row) != m.nFeature {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeature)
		}
		out[i] = m.classes[m.root.predict(row).predClass]
	}
	return out, nil
}

// --- public regressor ---------------------------------------------------------

// DecisionTreeRegressor is a CART decision tree for regression (Breiman 1984,
// §C18). Fit greedily splits on the (feature, threshold) that most reduces
// within-child target variance ([MSE]); Predict returns each leaf's mean target.
type DecisionTreeRegressor struct {
	root     *cartNode
	nFeature int
	cfg      cartConfig
}

// NewDecisionTreeRegressor constructs an unfitted regressor. Defaults match
// scikit-learn: [MSE] impurity, unlimited depth, minSamplesSplit=2,
// minSamplesLeaf=1. Configure with [WithMaxDepth], [WithMinSamplesLeaf] and
// [WithMinSamplesSplit].
func NewDecisionTreeRegressor(opts ...TreeOption) *DecisionTreeRegressor {
	cfg := defaultTreeConfig(true)
	for _, o := range opts {
		o(&cfg)
	}
	cfg.criterion = MSE // regression is always variance-based
	return &DecisionTreeRegressor{cfg: cfg}
}

// Fit grows the tree from X[n][d] and real targets y[n]. Errors on empty or
// mismatched input.
func (m *DecisionTreeRegressor) Fit(x [][]float64, y []float64) error {
	return m.fitWithSeed(x, y, 0x9e3779b97f4a7c15)
}

// fitWithSeed grows the regressor with a caller-supplied split RNG seed. Used by
// RandomForestRegressor so each tree gets an independent, deterministic stream.
func (m *DecisionTreeRegressor) fitWithSeed(x [][]float64, y []float64, seed uint64) error {
	d, err := validateXY(x, len(x), len(y))
	if err != nil {
		return err
	}
	b := &cartBuilder{x: x, yf: y, regression: true, cfg: m.cfg, rng: &lcg{state: nonzero(seed)}}
	idx := make([]int, len(x))
	for i := range idx {
		idx[i] = i
	}
	m.root = b.build(idx, 0)
	m.nFeature = d
	return nil
}

// Predict returns the predicted value for each row of X. It errors if called
// before Fit or if a row's width does not match the training data.
func (m *DecisionTreeRegressor) Predict(x [][]float64) ([]float64, error) {
	if m.root == nil {
		return nil, fmt.Errorf("classic: DecisionTreeRegressor.Predict before Fit")
	}
	out := make([]float64, len(x))
	for i, row := range x {
		if len(row) != m.nFeature {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeature)
		}
		out[i] = m.root.predict(row).value
	}
	return out, nil
}

// encodeLabels returns the sorted distinct labels and, for each sample, its
// index into that slice.
func encodeLabels(y []int) (classes []int, idx []int) {
	seen := map[int]struct{}{}
	for _, v := range y {
		seen[v] = struct{}{}
	}
	classes = make([]int, 0, len(seen))
	for v := range seen {
		classes = append(classes, v)
	}
	sort.Ints(classes)
	pos := make(map[int]int, len(classes))
	for i, v := range classes {
		pos[v] = i
	}
	idx = make([]int, len(y))
	for i, v := range y {
		idx[i] = pos[v]
	}
	return classes, idx
}
