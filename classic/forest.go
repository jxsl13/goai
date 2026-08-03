package classic

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
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
	n := len(x)
	nt := m.cfg.nTrees

	// Pre-derive every tree's bootstrap sample and split-RNG seed sequentially
	// from the single forest RNG, reproducing the exact draw order of the old
	// sequential loop (per tree: n bootstrap draws, then one rng.next() seed).
	// The trees are then built concurrently, but because each already owns a
	// fixed sample+seed the result is bit-identical regardless of completion
	// order — determinism is decoupled from goroutine scheduling.
	samples, seeds := deriveTreeInputs(n, nt, m.cfg.seed)

	m.trees = make([]*DecisionTreeClassifier, nt)
	if err := parallelBuild(nt, func(t int) error {
		// RECYCLE THE GATHER BUFFERS ACROSS TREES. Every tree materialized its own row-pointer
		// slice and label copy — two allocations the size of the training set, once per tree,
		// and together they were the largest single source of bytes in a forest fit.
		//
		// SAFE BECAUSE THE TREE KEEPS NEITHER: fitWithSeed stores only the fitted root, the
		// class set and the feature count, and the builder that holds x and y dies with the
		// call. Both buffers are fully overwritten before they are read, so a recycled one
		// carries nothing forward.
		gb := gatherPool.Get().(*gatherBuf)
		defer gatherPool.Put(gb)
		bx, by := gb.take(x, y, samples[t])
		tree := NewDecisionTreeClassifier(
			WithMaxDepth(m.cfg.tree.maxDepth),
			WithMinSamplesSplit(m.cfg.tree.minSamplesSplit),
			WithMinSamplesLeaf(m.cfg.tree.minSamplesLeaf),
			WithCriterion(m.cfg.tree.criterion),
			withMaxFeatures(mf),
		)
		// each tree gets its own deterministic split RNG stream
		if err := tree.fitWithSeed(bx, by, classes, seeds[t]); err != nil {
			return err
		}
		m.trees[t] = tree // goroutine writes only its own slot — race-free
		return nil
	}); err != nil {
		m.trees = nil
		return err
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
	pos := make(map[int]int, len(m.classes))
	for i, c := range m.classes {
		pos[c] = i
	}
	// Precompute each tree's class-index → forest vote-index once, so the hot per-(sample,
	// tree) loop does a plain slice lookup instead of the pos[label] map access (measured
	// ~14% of Predict). A tree's classes are a subset of the forest's, so every lookup hits.
	// One flat backing array (not a slice per tree) keeps this at two allocations.
	total := 0
	for _, tree := range m.trees {
		total += len(tree.classes)
	}
	flat := make([]int, total)
	treeVote := make([][]int, len(m.trees))
	off := 0
	for t, tree := range m.trees {
		tv := flat[off : off+len(tree.classes)]
		for ci, label := range tree.classes {
			tv[ci] = pos[label]
		}
		treeVote[t] = tv
		off += len(tree.classes)
	}
	for i, row := range x {
		if len(row) != m.nFeature {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeature)
		}
	}
	// Single-row tree walk (tree.root.predict) avoids the per-(sample,tree) wrapper +
	// result-slice allocation. Chunk-parallel across GOMAXPROCS with a PER-WORKER votes
	// scratch: rows are independent (each writes only out[i]; treeVote/flat are read-only),
	// so this is bit-identical to the serial loop. NOTE: an earlier per-ROW parallelBuild
	// channel dispatch measured slower (497µs seq vs 832µs pooled), but COARSE chunking (nw
	// goroutines, ~len(x)/nw rows each — the shape gbmPredictSum already uses) amortizes the
	// spawn and wins. Serial below a small work threshold.
	predictRow := func(votes []int, i int) {
		for k := range votes {
			votes[k] = 0
		}
		for t, tree := range m.trees {
			votes[treeVote[t][tree.root.predict(x[i]).predClass]]++
		}
		best, bc := 0, -1
		for k, v := range votes {
			if v > bc {
				bc, best = v, k
			}
		}
		out[i] = m.classes[best]
	}
	nw := runtime.GOMAXPROCS(0)
	if nw > len(x) {
		nw = len(x)
	}
	if nw <= 1 || len(x)*len(m.trees) < 1<<13 {
		votes := make([]int, len(m.classes))
		for i := range x {
			predictRow(votes, i)
		}
		return out, nil
	}
	// Rows are CLAIMED in blocks, not dealt one equal chunk per worker. The GOMAXPROCS screen this
	// change was made under is what selected it: this benchmark got SLOWER from 8 to 12 cores
	// (139273 -> 165529 ns/op), the signature PS3011 names for efficiency-core imbalance, where a
	// chunk landing on an E core sets the barrier. The scratch stays PER WORKER, allocated once
	// outside the claim loop, so claiming costs nothing per claim.
	const grain = 64
	var next atomic.Int64
	_ = parallelBuild(nw, func(int) error {
		votes := make([]int, len(m.classes)) // per-worker scratch
		for {
			lo := int(next.Add(grain)) - grain
			if lo >= len(x) {
				return nil
			}
			hi := min(lo+grain, len(x))
			for i := lo; i < hi; i++ {
				predictRow(votes, i)
			}
		}
	})
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
	n := len(x)
	nt := m.cfg.nTrees

	// Same deterministic pre-derivation as the classifier: fix each tree's
	// bootstrap sample and split seed in RNG-draw order, then build in parallel.
	samples, seeds := deriveTreeInputs(n, nt, m.cfg.seed)

	m.trees = make([]*DecisionTreeRegressor, nt)
	if err := parallelBuild(nt, func(t int) error {
		// The regressor's trees recycle the same way, for the same reason and under the same
		// retention argument as the classifier's above.
		gb := gatherPool.Get().(*gatherBuf)
		defer gatherPool.Put(gb)
		bx, by := gb.takeFloat(x, y, samples[t])
		tree := NewDecisionTreeRegressor(
			WithMaxDepth(m.cfg.tree.maxDepth),
			WithMinSamplesSplit(m.cfg.tree.minSamplesSplit),
			WithMinSamplesLeaf(m.cfg.tree.minSamplesLeaf),
			withMaxFeatures(mf),
		)
		if err := tree.fitWithSeed(bx, by, seeds[t]); err != nil {
			return err
		}
		m.trees[t] = tree // goroutine writes only its own slot — race-free
		return nil
	}); err != nil {
		m.trees = nil
		return err
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
	for i, row := range x {
		if len(row) != m.nFeature {
			return nil, fmt.Errorf("classic: row %d width %d, want %d", i, len(row), m.nFeature)
		}
	}
	out := make([]float64, len(x))
	// Single-row tree walk avoids the per-(sample,tree) wrapper/result allocation; each row
	// sums in tree order then divides — no shared scratch (s is local, cartNode.predict is
	// pure), so chunk-parallel across GOMAXPROCS is bit-identical (each out[i] disjoint).
	// Serial below a small work threshold.
	predictRow := func(i int) {
		var s float64
		for _, tree := range m.trees {
			s += tree.root.predict(x[i]).value
		}
		out[i] = s / float64(len(m.trees))
	}
	nw := runtime.GOMAXPROCS(0)
	if nw > len(x) {
		nw = len(x)
	}
	if nw <= 1 || len(x)*len(m.trees) < 1<<13 {
		for i := range x {
			predictRow(i)
		}
		return out, nil
	}
	csz := (len(x) + nw - 1) / nw
	_ = parallelBuild(nw, func(c int) error {
		lo := c * csz
		hi := lo + csz
		if hi > len(x) {
			hi = len(x)
		}
		for i := lo; i < hi; i++ {
			predictRow(i)
		}
		return nil
	})
	return out, nil
}

// --- helpers ------------------------------------------------------------------

// deriveTreeInputs pre-computes, sequentially from a single forest RNG, each
// tree's bootstrap sample and per-tree split seed. It reproduces the exact draw
// order of the original sequential Fit loop — for every tree, n bootstrap draws
// followed by one rng.next() seed — so the parallel build produces bit-identical
// trees to the sequential forest for the same seed. Doing all RNG consumption up
// front (not inside the goroutines) is what keeps determinism independent of
// goroutine scheduling.
func deriveTreeInputs(n, nTrees int, seed int64) (samples [][]int, seeds []uint64) {
	rng := &lcg{state: seedState(seed)}
	samples = make([][]int, nTrees)
	seeds = make([]uint64, nTrees)
	for t := 0; t < nTrees; t++ {
		samples[t] = bootstrap(n, rng)
		seeds[t] = rng.next()
	}
	return samples, seeds
}

// parallelBuild runs work(t) for t in [0,n) across a bounded pool of
// GOMAXPROCS workers. Each invocation is independent and must write only its own
// data (e.g. trees[t]); parallelBuild adds no shared mutable state beyond the
// first-error latch. It returns the first error reported by any worker (after
// all workers drain), or nil.
func parallelBuild(n int, work func(t int) error) error {
	if n <= 0 {
		return nil
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers < 1 {
		workers = 1
	}
	var (
		wg       sync.WaitGroup
		mu       sync.Mutex
		firstErr error
	)
	run := func(t int) {
		if err := work(t); err != nil {
			mu.Lock()
			if firstErr == nil {
				firstErr = err
			}
			mu.Unlock()
		}
	}
	// NO QUEUE WHEN THERE IS NOTHING TO QUEUE. The channel below exists so more jobs than
	// workers can be handed out one at a time, and it is pure cost when n <= workers — which is
	// how the tree builders use this helper: they call it with exactly one job per chunk, once
	// per NODE, thousands of times per fit. An unbuffered send is a rendezvous, so that path
	// paid two handoffs per job on top of the goroutine. A GBM fit spent 95.6% of its profile
	// samples in pthread_cond_wait, pthread_cond_signal and usleep against 1.75% in the split
	// scan itself.
	if n <= workers {
		for t := 0; t < n; t++ {
			wg.Add(1)
			go func(t int) {
				defer wg.Done()
				run(t)
			}(t)
		}
		wg.Wait()
		return firstErr
	}
	jobs := make(chan int)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range jobs {
				run(t)
			}
		}()
	}
	for t := 0; t < n; t++ {
		jobs <- t
	}
	close(jobs)
	wg.Wait()
	return firstErr
}

// seedState maps a user seed to a nonzero LCG state (xorshift needs nonzero).
func seedState(seed int64) uint64 {
	s := uint64(seed)*2862933555777941757 + 3037000493
	if s == 0 {
		s = 0x9e3779b97f4a7c15
	}
	return s
}

// gatherBuf is one tree's row-pointer and label scratch, recycled across trees.
type gatherBuf struct {
	bx  [][]float64
	by  []int
	byf []float64
}

var gatherPool = sync.Pool{New: func() any { return &gatherBuf{} }}

// take fills the buffer with the rows idx selects and returns views of exactly that length.
// Both slices are written at every position, so nothing survives from the previous tree.
func (g *gatherBuf) take(x [][]float64, y []int, idx []int) ([][]float64, []int) {
	if cap(g.bx) < len(idx) {
		g.bx = make([][]float64, len(idx))
	}
	if cap(g.by) < len(idx) {
		g.by = make([]int, len(idx))
	}
	bx, by := g.bx[:len(idx)], g.by[:len(idx)]
	for i, j := range idx {
		bx[i] = x[j]
		by[i] = y[j]
	}
	return bx, by
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

// takeFloat is take for a float target. It keeps its own label buffer rather than reinterpreting
// the integer one, so a buffer recycled between a classifier and a regressor cannot alias.
func (g *gatherBuf) takeFloat(x [][]float64, y []float64, idx []int) ([][]float64, []float64) {
	if cap(g.bx) < len(idx) {
		g.bx = make([][]float64, len(idx))
	}
	if cap(g.byf) < len(idx) {
		g.byf = make([]float64, len(idx))
	}
	bx, by := g.bx[:len(idx)], g.byf[:len(idx)]
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
