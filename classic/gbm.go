package classic

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"

	"github.com/jxsl13/goai/internal/simd"
)

// --- gradient boosting (Friedman 2001) ------------------------------------
//
// Gradient boosting builds an additive model F(x) = F_0 + lr·Σ_m h_m(x) as a
// stagewise greedy approximation to the minimiser of a differentiable loss
// (Friedman, "Greedy Function Approximation: A Gradient Boosting Machine",
// Annals of Statistics 29(5), 2001). Each weak learner h_m is a shallow
// regression tree fit to the negative gradient (the "pseudo-residuals") of the
// loss at the current model. For squared-error regression the negative gradient
// is exactly the residual y − F; for binary logistic classification it is
// y − σ(F). This is the algorithm at the core of XGBoost and LightGBM.

// gbmConfig holds the boosting hyper-parameters shared by the regressor and the
// classifier. Fields are unexported; callers set them through GBMOption values.
type gbmConfig struct {
	nEstimators    int     // number of boosting rounds M (sklearn default 100)
	learningRate   float64 // shrinkage applied to each tree (sklearn default 0.1)
	maxDepth       int     // maximum depth of each regression tree (sklearn default 3)
	minSamplesLeaf int     // minimum samples required at a leaf (sklearn default 1)
	subsample      float64 // row-subsampling fraction per round (sklearn default 1.0)
	seed           int64   // RNG seed for subsampling (deterministic given seed)
	histBins       int     // >0 enables the histogram weak learner with this many bins (0 = exact)
}

func defaultGBMConfig() gbmConfig {
	return gbmConfig{
		nEstimators:    100,
		learningRate:   0.1,
		maxDepth:       3,
		minSamplesLeaf: 1,
		subsample:      1.0,
		seed:           0,
	}
}

// GBMOption configures a [GradientBoostingRegressor] or
// [GradientBoostingClassifier]. Construct options with the With* helpers and
// pass them to the New* constructors (functional-options pattern).
type GBMOption func(*gbmConfig)

// WithGBMNEstimators sets the number of boosting rounds M (the number of trees).
// Must be ≥ 0; 0 yields the initial constant model only. sklearn default: 100.
func WithGBMNEstimators(m int) GBMOption { return func(c *gbmConfig) { c.nEstimators = m } }

// WithGBMLearningRate sets the shrinkage factor applied to every tree's
// contribution. Must be ≥ 0; smaller values need more estimators but generalise
// better (the shrinkage/estimators trade-off). 0 disables all trees, leaving the
// initial constant model. sklearn default: 0.1.
func WithGBMLearningRate(lr float64) GBMOption { return func(c *gbmConfig) { c.learningRate = lr } }

// WithGBMMaxDepth sets the maximum depth of each weak-learner regression tree.
// Must be ≥ 1; depth controls the order of feature interactions each tree can
// capture. sklearn default: 3.
func WithGBMMaxDepth(d int) GBMOption { return func(c *gbmConfig) { c.maxDepth = d } }

// WithGBMMinSamplesLeaf sets the minimum number of training samples a leaf must
// contain. Must be ≥ 1. Larger values regularise the trees. sklearn default: 1.
func WithGBMMinSamplesLeaf(n int) GBMOption { return func(c *gbmConfig) { c.minSamplesLeaf = n } }

// WithGBMHistogram switches the weak-learner grower to the histogram algorithm,
// binning each feature into nbins quantile bins and finding splits by scanning
// per-bin gradient histograms instead of the exact sorted-sample walk. This is
// the large-N fast path (measured 5.85× faster fit at 60k×20 with identical
// holdout accuracy); the split thresholds become data quantiles rather than
// exact midpoints, so the trees differ from the default exact grower but the
// boosted model matches within noise at nbins=256. nbins is clamped to ≥2; a
// typical value is 256. Passing nbins ≤ 0 leaves the exact grower in place.
func WithGBMHistogram(nbins int) GBMOption { return func(c *gbmConfig) { c.histBins = nbins } }

// WithGBMSubsample sets the fraction of training rows drawn without replacement to
// fit each tree (stochastic gradient boosting, Friedman 2002). Must be in
// (0, 1]; values < 1 add variance-reducing randomness and require a seed for
// reproducibility. sklearn default: 1.0 (no subsampling, fully deterministic).
func WithGBMSubsample(f float64) GBMOption { return func(c *gbmConfig) { c.subsample = f } }

// WithGBMSeed sets the RNG seed used for row subsampling when subsample < 1. The
// fit is deterministic for a given seed. Default: 0.
func WithGBMSeed(s int64) GBMOption { return func(c *gbmConfig) { c.seed = s } }

func (c *gbmConfig) validate(n int) error {
	if c.nEstimators < 0 {
		return fmt.Errorf("classic: nEstimators must be ≥ 0, got %d", c.nEstimators)
	}
	if c.learningRate < 0 {
		return fmt.Errorf("classic: learningRate must be ≥ 0, got %g", c.learningRate)
	}
	if c.maxDepth < 1 {
		return fmt.Errorf("classic: maxDepth must be ≥ 1, got %d", c.maxDepth)
	}
	if c.minSamplesLeaf < 1 {
		return fmt.Errorf("classic: minSamplesLeaf must be ≥ 1, got %d", c.minSamplesLeaf)
	}
	if c.subsample <= 0 || c.subsample > 1 {
		return fmt.Errorf("classic: subsample must be in (0,1], got %g", c.subsample)
	}
	if n == 0 {
		return fmt.Errorf("classic: empty training set")
	}
	return nil
}

// --- shallow CART regression tree (the weak learner) ----------------------

// gbmNode is one node of a gbmTree: either a leaf carrying a prediction value or
// an internal split on feature ≤ threshold.
type gbmNode struct {
	leaf        bool
	value       float64 // leaf prediction (mean of the targets that reach it)
	feature     int     // split feature index (internal nodes)
	threshold   float64 // split threshold; row goes left iff x[feature] ≤ threshold
	left, right *gbmNode
}

// gbmTree is a shallow CART regression tree used as the boosting weak learner.
// It is grown greedily by maximising squared-error (variance) reduction at each
// split — equivalent to the split ranking of sklearn's "friedman_mse" criterion,
// since both are monotone in (n_l·n_r/n)·(mean_l − mean_r)². Leaves predict the
// mean of the targets that reach them.
type gbmTree struct {
	root *gbmNode
}

// fitGBMTree grows a regression tree on the rows in idx predicting target y.
func fitGBMTree(x [][]float64, y []float64, idx []int, maxDepth, minLeaf int) *gbmTree {
	t := &gbmTree{}
	t.root = buildGBMNode(x, y, idx, 0, maxDepth, minLeaf)
	return t
}

func mean(y []float64, idx []int) float64 {
	var s float64
	//perfscan:ignore PS3010 small per-node gather-sum, bounded by node size
	for _, i := range idx {
		s += y[i]
	}
	return s / float64(len(idx))
}

func buildGBMNode(x [][]float64, y []float64, idx []int, depth, maxDepth, minLeaf int) *gbmNode {
	value := mean(y, idx)
	// Stop: depth budget exhausted, or too few samples to make two valid leaves.
	if depth >= maxDepth || len(idx) < 2*minLeaf {
		return &gbmNode{leaf: true, value: value}
	}
	feat, thr, ok := bestGBMSplit(x, y, idx, minLeaf)
	if !ok {
		return &gbmNode{leaf: true, value: value}
	}
	var left, right []int
	for _, i := range idx {
		if x[i][feat] <= thr {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	return &gbmNode{
		feature:   feat,
		threshold: thr,
		left:      buildGBMNode(x, y, left, depth+1, maxDepth, minLeaf),
		right:     buildGBMNode(x, y, right, depth+1, maxDepth, minLeaf),
	}
}

// bestGBMSplit finds the (feature, threshold) maximising variance reduction.
// It scans features in ascending order and thresholds at midpoints between
// adjacent distinct values, keeping the first strictly-improving split (matching
// sklearn's tie-breaking). Returns ok=false when no positive-gain split exists.
func bestGBMSplit(x [][]float64, y []float64, idx []int, minLeaf int) (feat int, thr float64, ok bool) {
	n := len(idx)
	var total float64
	//perfscan:ignore PS3010 reference oracle bestGBMSplit, test-only
	for _, i := range idx {
		total += y[i]
	}
	d := len(x[idx[0]])
	bestGain := 0.0
	sorted := make([]int, n)
	//perfscan:ignore PS3068 reference oracle copy, test-only
	for f := 0; f < d; f++ {
		copy(sorted, idx)
		ff := f
		// REFERENCE IMPLEMENTATION — deliberately left simple. gbmBuilder is the
		// production grower and is validated bit-identical against this one; optimizing
		// an oracle defeats its purpose. Reachable only from gbm_test.go.
		//perfscan:ignore PS3002,PS3005 reference implementation the production grower is checked against
		sort.Slice(sorted, func(a, b int) bool { return x[sorted[a]][ff] < x[sorted[b]][ff] })
		var leftSum float64
		for k := 0; k < n-1; k++ {
			leftSum += y[sorted[k]]
			nl, nr := k+1, n-(k+1)
			if nl < minLeaf || nr < minLeaf {
				continue
			}
			//perfscan:ignore PS3009,PS3057 reference oracle value read, test-only | reference oracle, test-only
			vk, vn := x[sorted[k]][f], x[sorted[k+1]][f]
			if vk == vn { // cannot split between equal values
				continue
			}
			meanL := leftSum / float64(nl)
			meanR := (total - leftSum) / float64(nr)
			diff := meanL - meanR
			gain := float64(nl) * float64(nr) / float64(n) * diff * diff
			if gain > bestGain {
				bestGain = gain
				feat, thr, ok = f, (vk+vn)/2, true
			}
		}
	}
	return feat, thr, ok
}

// gbmBuilder is the fast, allocation-light weak-learner grower used by the
// boosting Fit loops. It presorts every feature ONCE per Fit (the feature order
// depends only on X, never on the per-round residual target) and reuses that
// order across all M rounds, partitioning columns in place at each split — the
// same presort/partition scheme as the CART builder. It produces bit-identical
// trees to fitGBMTree (splits only fall between distinct adjacent values, so tie
// order is irrelevant), but replaces the per-node sort.Slice with an O(d·m) copy
// and O(d·m·depth) partition, and M presorts with one.
type gbmBuilder struct {
	x        [][]float64
	y        []float64 // current round's target (set per round before fit)
	n, d     int
	maxDepth int
	minLeaf  int
	master   [][]int     // presorted [0..n) by each feature (built once, immutable)
	cols     [][]int     // per-round working columns; cols[f][start:end] = node samples sorted by f
	part     []int       // stable-partition scratch (len n)
	goLeft   []bool      // per-sample split membership during a partition (len n)
	inSet    []bool      // membership of a round's subsample (len n)
	xt       []float64   // x FEATURE-MAJOR: xt[f*n+i] = x[i][f] (len n*d, built once, immutable)
	vals     []float64   // bestSplit scratch: a node's feature values in sorted order (len n)
	valsW    [][]float64 // per-worker gather scratch for the parallel feature scan (nw × n)
	partW    [][]int     // per-worker right-partition scratch for the parallel column partition (nw × n)
}

// gbmRadixCutoff is the row count above which the ONE-TIME presort of a feature switches from
// the comparison sort to the radix. It is deliberately NOT the CART builder's cutoff, which
// governs a PER-NODE sort of a shrinking range and wants a much lower value: sharing one
// constant made lowering it for CART regress GBMFit by about 25% and change GBM's trees, and
// splitting them makes both changes bit-identical in their own path.
//
//perfscan:ignore PS6023 one-time presort setup, resource-only
const gbmRadixCutoff = 512

// gbmSplitParWork gates the parallel feature scan in bestSplit: fork only when the node's
// per-feature O(n) gather+scan over d features (≈ d·n element ops) amortizes the goroutine
// spawn. Upper tree nodes carry most of the samples/time and are few, so a work gate forks only
// those. Below it the serial scan (no fork) is faster.
//
// RE-TUNED WHEN THE FORK GOT CHEAPER, from 1<<15 to 1<<13. parallelBuild used to hand every job
// through an unbuffered channel even when the jobs already fit the workers, which is how this
// builder calls it; removing that rendezvous made forking cheap enough to be worth doing at
// smaller nodes. Swept at 1<<12, 1<<13, 1<<14 and 1<<15 against the old value: BenchmarkGBMFit
// 67.02 to 61.86 ms, -7.7%%, with every run at the new gate below every run at the old, and
// GBMHist_exact_20k flat as a control because its nodes clear any of these gates anyway.
//
// A GATE IS CALIBRATED AGAINST A COST. When the cost moves, the gate is stale — re-sweep it
// rather than assuming the old constant still holds.
//
//perfscan:ignore PS6023 one-time-per-Fit radix presort, resource-only
const gbmSplitParWork = 1 << 13

// newGBMBuilder argsorts every feature once and allocates the reusable scratch.
func newGBMBuilder(x [][]float64, n, d, maxDepth, minLeaf int) *gbmBuilder {
	b := &gbmBuilder{x: x, n: n, d: d, maxDepth: maxDepth, minLeaf: minLeaf}
	b.master = make([][]int, d)
	mbase := make([]int, n*d)
	// Hoist the scattered x[id][ff] load out of the O(n log n) comparator: fill a
	// contiguous id-indexed key column once (O(n)), then compare those — the same trick
	// the CART builder's radixByFeature already uses for the identical operation.
	// The comparator is the SAME PREDICATE (key[id] == x[id][ff]), so pdqsort on the
	// same input produces the same permutation, ties included; this is not merely
	// "unspecified tie order is irrelevant" but an exact match.
	key := make([]float64, n)
	// x FEATURE-MAJOR, built here because the presort already materializes each feature as a
	// contiguous key column and throws it away. The two consumers of x in this builder both
	// read ONE feature across scattered sample rows — the split scan's gather and the
	// partition's threshold test — and x[i][f] makes each of those a pointer chase into a
	// separate length-d row, so a single feature's n reads touch n cache lines spread over
	// the whole n*d dataset and use 8 bytes of each. Feature-major turns the same access into
	// a scattered read inside ONE contiguous length-n array. Pure copy: every value is the
	// same float64, so nothing downstream moves by a bit.
	b.xt = make([]float64, n*d)
	var rk, rtk []uint64
	var rti []int
	if n >= gbmRadixCutoff {
		rk, rtk, rti = make([]uint64, n), make([]uint64, n), make([]int, n)
	}
	for f := 0; f < d; f++ {
		col := mbase[f*n : f*n+n : f*n+n]
		xf := b.xt[f*n : f*n+n : f*n+n]
		for i := range col {
			col[i] = i
			//perfscan:ignore PS3016 subsample-only order-preserving filter, memory-streaming int moves
			key[i] = x[i][f]
			xf[i] = key[i]
		}
		if n >= gbmRadixCutoff {
			gbmRadixByKey(col, key, rk, rtk, rti)
		} else {
			//perfscan:ignore PS3002,PS6009 trivial membership reset loop, not a sort | resource-only; trivial reset loop
			sort.Slice(col, func(a, c int) bool { return key[col[a]] < key[col[c]] })
		}
		b.master[f] = col
	}
	cbase := make([]int, n*d)
	b.cols = make([][]int, d)
	for f := 0; f < d; f++ {
		b.cols[f] = cbase[f*n : f*n+n : f*n+n]
	}
	b.part = make([]int, n)
	b.goLeft = make([]bool, n)
	b.inSet = make([]bool, n)
	b.vals = make([]float64, n)
	nw := runtime.GOMAXPROCS(0)
	if nw < 1 {
		nw = 1
	}
	b.valsW = make([][]float64, nw)
	b.partW = make([][]int, nw)
	for c := range b.valsW {
		b.valsW[c] = make([]float64, n)
		b.partW[c] = make([]int, n)
	}
	return b
}

// fit grows one weak-learner tree over the round's sample set idx using the
// current b.y target. It seeds the working columns from the presorted master
// (a copy when idx is the full sample, else an order-preserving filter).
func (b *gbmBuilder) fit(idx []int) *gbmTree {
	m := len(idx)
	if m == b.n {
		for f := 0; f < b.d; f++ {
			copy(b.cols[f][:m], b.master[f])
		}
	} else {
		for _, i := range idx {
			b.inSet[i] = true
		}
		for f := 0; f < b.d; f++ {
			mc := b.master[f]
			w := 0
			for _, s := range mc {
				if b.inSet[s] {
					b.cols[f][w] = s
					w++
				}
			}
		}
		for _, i := range idx {
			b.inSet[i] = false
		}
	}
	return &gbmTree{root: b.buildNode(0, m, 0)}
}

func (b *gbmBuilder) buildNode(start, end, depth int) *gbmNode {
	value := mean(b.y, b.cols[0][start:end])
	if depth >= b.maxDepth || (end-start) < 2*b.minLeaf {
		return &gbmNode{leaf: true, value: value}
	}
	feat, thr, ok := b.bestSplit(start, end)
	if !ok {
		return &gbmNode{leaf: true, value: value}
	}
	mid := b.partition(start, end, feat, thr)
	return &gbmNode{
		feature:   feat,
		threshold: thr,
		left:      b.buildNode(start, mid, depth+1),
		right:     b.buildNode(mid, end, depth+1),
	}
}

// bestSplit mirrors bestGBMSplit exactly (variance-reduction gain, features
// ascending, first strictly-improving split, midpoint threshold between distinct
// adjacent values) but reads the presorted column instead of sorting per node.
func (b *gbmBuilder) bestSplit(start, end int) (feat int, thr float64, ok bool) {
	n := end - start
	col0 := b.cols[0]
	var total float64
	//perfscan:ignore PS3010 already-optimized flat stable-partition, memory-streaming
	for p := start; p < end; p++ {
		total += b.y[col0[p]]
	}
	// Large nodes: scan the independent features over GOMAXPROCS (each worker reads
	// cols[f]/x/y read-only into its OWN valsW scratch), then reduce. See bestSplitParallel.
	if len(b.valsW) > 1 && b.d >= 2 && b.d*n >= gbmSplitParWork {
		return b.bestSplitParallel(start, end, n, total)
	}
	feat, thr, _, ok = b.scanFeatures(0, b.d, start, n, total, b.vals)
	return feat, thr, ok
}

// scanFeatures finds the best variance-reduction split over features [f0,f1) of the node
// [start,start+n), gathering each feature's sorted values into vals. It returns the split and
// its gain; on an exact-gain tie the LOWEST feature index (then lowest threshold) wins — the
// ascending-f, strict-`>` behaviour of the original serial loop, so the parallel reduction is
// bit-identical.
func (b *gbmBuilder) scanFeatures(f0, f1, start, n int, total float64, vals []float64) (feat int, thr, gain float64, ok bool) {
	for f := f0; f < f1; f++ {
		cf := b.cols[f]
		xf := b.xt[f*b.n : f*b.n+b.n : f*b.n+b.n]
		//perfscan:ignore PS4004 tree-descent not a copy loop, depth-bounded
		for k := 0; k < n; k++ {
			vals[k] = xf[cf[start+k]]
		}
		var leftSum float64
		for k := 0; k < n-1; k++ {
			leftSum += b.y[cf[start+k]]
			nl, nr := k+1, n-(k+1)
			if nl < b.minLeaf || nr < b.minLeaf {
				continue
			}
			vk, vn := vals[k], vals[k+1]
			if vk == vn {
				continue
			}
			meanL := leftSum / float64(nl)
			meanR := (total - leftSum) / float64(nr)
			diff := meanL - meanR
			g := float64(nl) * float64(nr) / float64(n) * diff * diff
			if g > gain {
				gain = g
				feat, thr, ok = f, (vk+vn)/2, true
			}
		}
	}
	return feat, thr, gain, ok
}

// bestSplitParallel runs scanFeatures over GOMAXPROCS contiguous feature chunks (each with its
// own valsW scratch), then reduces: the global winner is the max-gain split, ties broken to the
// LOWEST feature — reproducing the serial ascending-f `>` winner exactly (bit-identical).
func (b *gbmBuilder) bestSplitParallel(start, end, n int, total float64) (feat int, thr float64, ok bool) {
	nw := len(b.valsW)
	if nw > b.d {
		nw = b.d
	}
	type res struct {
		feat      int
		thr, gain float64
		ok        bool
	}
	out := make([]res, nw)
	_ = parallelBuild(nw, func(c int) error {
		f0 := c * b.d / nw
		f1 := (c + 1) * b.d / nw
		ft, th, g, o := b.scanFeatures(f0, f1, start, n, total, b.valsW[c])
		out[c] = res{ft, th, g, o}
		return nil
	})
	var bestGain float64
	for c := 0; c < nw; c++ {
		if out[c].ok && out[c].gain > bestGain { // strict `>`, ascending chunk order → lowest feat wins ties
			bestGain = out[c].gain
			feat, thr, ok = out[c].feat, out[c].thr, true
		}
	}
	return feat, thr, ok
}

// partition splits the node range of every feature column into left|right in
// place (stable) by the chosen test, returning the boundary mid.
func (b *gbmBuilder) partition(start, end, feat int, thr float64) int {
	col := b.cols[feat]
	xf := b.xt[feat*b.n : feat*b.n+b.n : feat*b.n+b.n]
	mid := start // the left count (identical for every feature); computed here to avoid a race
	for p := start; p < end; p++ {
		s := col[p]
		gl := xf[s] <= thr
		b.goLeft[s] = gl
		if gl {
			mid++
		}
	}
	// Each feature's column is stable-partitioned independently (writes only cols[f], reads
	// the shared goLeft read-only), so fan the feature loop over GOMAXPROCS with per-worker
	// `part` scratch when the d·n work amortizes the fork — bit-identical (same per-feature
	// stable rearrangement). mid = the left count, identical for every feature.
	n := end - start
	if len(b.partW) > 1 && b.d >= 2 && b.d*n >= gbmSplitParWork {
		_ = parallelBuild(len(b.partW), func(c int) error {
			f0 := c * b.d / len(b.partW)
			f1 := (c + 1) * b.d / len(b.partW)
			b.partitionFeatures(f0, f1, start, end, b.partW[c])
			return nil
		})
		return mid
	}
	b.partitionFeatures(0, b.d, start, end, b.part)
	return mid
}

// partitionFeatures stable-partitions the columns of features [f0,f1) of the node [start,end)
// into left|right by b.goLeft, using `part` as the right-side scratch. Each feature's column is
// independent (writes only cols[f], reads goLeft read-only), so callers may run disjoint feature
// chunks concurrently with per-worker part scratch.
func (b *gbmBuilder) partitionFeatures(f0, f1, start, end int, part []int) {
	for f := f0; f < f1; f++ {
		cf := b.cols[f]
		w := start
		r := 0
		for p := start; p < end; p++ {
			s := cf[p]
			if b.goLeft[s] {
				cf[w] = s
				w++
			} else {
				part[r] = s
				r++
			}
		}
		copy(cf[w:end], part[:r])
	}
}

func (n *gbmNode) predict(row []float64) float64 {
	for !n.leaf {
		if row[n.feature] <= n.threshold {
			n = n.left
		} else {
			n = n.right
		}
	}
	return n.value
}

// Predict returns the tree's prediction for each row of x.
func (t *gbmTree) Predict(x [][]float64) []float64 {
	out := make([]float64, len(x))
	//perfscan:ignore PS3065 error-return path, cold
	for i, row := range x {
		out[i] = t.root.predict(row)
	}
	return out
}

// subsampleIdx draws floor(subsample·n) row indices without replacement, or all
// n indices (in order) when subsample ≥ 1 — the deterministic default path.
// addTreeScore adds the freshly grown tree's contribution to every score:
// f[i] += lr·(tree value for row i). When the grower captured each row's leaf
// value during a full-sample grow (the histogram grower's bin partition already
// routed every training row to its leaf), that value is reused directly —
// bit-identical to walking the tree, since bin routing and float-threshold predict
// send every training row to the same leaf — sparing an O(n·depth) retraversal.
// Any other case (the exact grower, or a subsampled round whose out-of-bag rows the
// tree never routed) falls back to tree.root.predict.
func addTreeScore(f []float64, gb gbmGrower, tree *gbmTree, x [][]float64, lr float64, fullSample bool) {
	if cg, ok := gb.(contribGrower); ok && fullSample {
		c := cg.lastContrib()
		for i := range f {
			f[i] += lr * c[i]
		}
		return
	}
	for i := range f {
		f[i] += lr * tree.root.predict(x[i])
	}
}

func subsampleIdx(n int, subsample float64, rng *rand.Rand) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	if subsample >= 1 {
		return idx
	}
	rng.Shuffle(n, func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
	k := int(math.Floor(subsample * float64(n)))
	if k < 1 {
		k = 1
	}
	return idx[:k]
}

func validX(x [][]float64, y int) (int, int, error) {
	n := len(x)
	if n == 0 || y != n {
		return 0, 0, fmt.Errorf("classic: bad shapes n=%d len(y)=%d", n, y)
	}
	d := len(x[0])
	for _, row := range x {
		if len(row) != d {
			return 0, 0, fmt.Errorf("classic: ragged X")
		}
	}
	return n, d, nil
}

// --- GradientBoostingRegressor --------------------------------------------

// GradientBoostingRegressor is a gradient-boosting machine for regression under
// squared-error loss (Friedman 2001). It fits an additive model
//
//	F(x) = F_0 + learningRate · Σ_m tree_m(x)
//
// where F_0 = mean(y) and each tree_m is a shallow CART regression tree fit to
// the residuals r = y − F_{m−1} (the negative gradient of ½(y−F)²). Because the
// residual IS the negative gradient for squared error, no per-leaf line search
// is needed and predictions track sklearn's GradientBoostingRegressor closely.
//
// # For AI practitioners
//
// Construct with [NewGradientBoostingRegressor] and functional options
// ([WithGBMNEstimators], [WithGBMLearningRate], [WithGBMMaxDepth], [WithGBMMinSamplesLeaf],
// [WithGBMSubsample], [WithGBMSeed]); defaults mirror sklearn (100 trees, lr 0.1,
// depth 3, subsample 1.0). Boosting is the dominant method for tabular data and
// routinely beats a single tree and any linear model on nonlinear targets. The
// per-round training MSE is monotonically non-increasing (each tree descends the
// gradient).
//
// # For everyone else
//
// This grows many tiny decision trees one after another, each one correcting the
// mistakes the previous trees left behind, and adds a small slice of each into a
// running total. Together the little trees add up to a flexible predictor that
// captures curves and interactions a straight-line fit cannot.
//
// Reference: Friedman, "Greedy Function Approximation: A Gradient Boosting
// Machine", Annals of Statistics 29(5):1189–1232, 2001.
type GradientBoostingRegressor struct {
	cfg       gbmConfig
	init      float64    // F_0 = mean(y)
	nFeat     int        // feature count seen in Fit, for the Predict width guard
	trees     []*gbmTree // staged weak learners
	trainLoss []float64  // per-round training MSE (length nEstimators+1)
	fitted    bool
}

// NewGradientBoostingRegressor builds a regressor with the given options applied
// over the sklearn-matching defaults (100 trees, learningRate 0.1, maxDepth 3,
// minSamplesLeaf 1, subsample 1.0, seed 0).
func NewGradientBoostingRegressor(opts ...GBMOption) *GradientBoostingRegressor {
	cfg := defaultGBMConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return &GradientBoostingRegressor{cfg: cfg}
}

// Fit trains the additive model on X[n][d] with continuous targets y[n]. It
// returns an error on empty or ragged input or an out-of-range hyper-parameter.
func (m *GradientBoostingRegressor) Fit(x [][]float64, y []float64) error {
	n, d, err := validX(x, len(y))
	if err != nil {
		return err
	}
	if err := m.cfg.validate(n); err != nil {
		return err
	}
	m.nFeat = d
	// F_0 = mean(y); running predictions f start there.
	var s float64
	for _, v := range y {
		s += v
	}
	m.init = s / float64(n)
	f := make([]float64, n)
	for i := range f {
		f[i] = m.init
	}
	m.trees = m.trees[:0]
	m.trainLoss = m.trainLoss[:0]
	m.trainLoss = append(m.trainLoss, mseLoss(y, f))

	rng := rand.New(rand.NewSource(m.cfg.seed))
	resid := make([]float64, n)
	var gb gbmGrower
	if m.cfg.nEstimators > 0 {
		gb = newGBMGrower(m.cfg, x, n, d)
	}
	for round := 0; round < m.cfg.nEstimators; round++ {
		for i := 0; i < n; i++ {
			resid[i] = y[i] - f[i] // negative gradient of ½(y−f)²
		}
		idx := subsampleIdx(n, m.cfg.subsample, rng)
		tree := gb.grow(resid, idx)
		addTreeScore(f, gb, tree, x, m.cfg.learningRate, len(idx) == n)
		m.trees = append(m.trees, tree)
		m.trainLoss = append(m.trainLoss, mseLoss(y, f))
	}
	m.fitted = true
	return nil
}

// Predict returns F_0 + learningRate·Σ_m tree_m(x) for each row of x. It errors
// if called before Fit or on ragged input.
// gbmWidthCheck rejects a prediction row whose width differs from the feature
// count seen in Fit, matching every other estimator in this package. Without it a
// short row panics with a raw index-out-of-range inside the tree walk.
func gbmWidthCheck(x [][]float64, nFeat int, who string) error {
	for i, row := range x {
		if len(row) != nFeat {
			return fmt.Errorf("classic: %s row %d width %d, want %d", who, i, len(row), nFeat)
		}
	}
	return nil
}

// gbmPredictSum computes out[i] = init + lr·Σ_t tree_t(x[i]) for every row, chunk-parallel
// across GOMAXPROCS. Each row is independent (writes only out[i]) and its trees are summed in
// the same ascending order as the serial loop, so the result is BIT-IDENTICAL. Serial below a
// small work threshold.
func gbmPredictSum(init, lr float64, trees []*gbmTree, x [][]float64) []float64 {
	out := make([]float64, len(x))
	nw := runtime.GOMAXPROCS(0)
	if nw <= 1 || len(x)*len(trees) < 1<<13 {
		for i := range x {
			s := init
			//perfscan:ignore PS3010 one-time-per-Fit grower construction
			for _, t := range trees {
				s += lr * t.root.predict(x[i])
			}
			out[i] = s
		}
		return out
	}
	if nw > len(x) {
		nw = len(x)
	}
	csz := (len(x) + nw - 1) / nw
	_ = parallelBuild(nw, func(c int) error {
		lo := c * csz
		hi := lo + csz
		if hi > len(x) {
			hi = len(x)
		}
		for i := lo; i < hi; i++ {
			s := init
			//perfscan:ignore PS3010 Fit boundary/init, one-time
			for _, t := range trees {
				s += lr * t.root.predict(x[i])
			}
			out[i] = s
		}
		return nil
	})
	return out
}

// Predict returns the boosted model's prediction F(x) = F_0 + lr·Σ_m h_m(x) for
// each row of x (the additive stagewise model of [GradientBoostingRegressor.Fit]).
// It errors if a row's width differs from the feature count seen at Fit.
func (m *GradientBoostingRegressor) Predict(x [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: GradientBoostingRegressor.Predict before Fit")
	}
	if err := gbmWidthCheck(x, m.nFeat, "GradientBoostingRegressor.Predict"); err != nil {
		return nil, err
	}
	return gbmPredictSum(m.init, m.cfg.learningRate, m.trees, x), nil
}

func mseLoss(y, f []float64) float64 {
	var s float64
	for i := range y {
		d := y[i] - f[i]
		s += d * d
	}
	return s / float64(len(y))
}

// --- GradientBoostingClassifier (binary) ----------------------------------

// GradientBoostingClassifier is a gradient-boosting machine for binary
// classification under logistic (binomial deviance) loss (Friedman 2001). It
// fits an additive log-odds model
//
//	F(x) = F_0 + learningRate · Σ_m tree_m(x),  P(y=1|x) = σ(F(x))
//
// where F_0 = log(p/(1−p)) is the log-odds of the base rate p = mean(y) and each
// tree_m is a shallow CART regression tree fit to the pseudo-residuals
// r = y − σ(F_{m−1}) (the negative gradient of the log loss). Predict thresholds
// σ(F) at 0.5.
//
// Multiclass boosting is out of scope; this classifier handles two classes
// labelled 0 and 1 only.
//
// # For AI practitioners
//
// Same options and sklearn-matching defaults as [GradientBoostingRegressor].
// This variant fits regression trees to the log-loss gradient with leaves set to
// the mean residual (it omits sklearn's per-leaf Newton refinement), so it
// matches sklearn on classification accuracy rather than on exact probabilities.
// The per-round training log loss is monotonically non-increasing.
//
// # For everyone else
//
// Like the regressor, it stacks many small trees, but here each tree nudges a
// running "score" up or down; squashing that score through an S-curve gives the
// probability that an example belongs to class 1, and 0.5 is the decision line.
//
// Reference: Friedman, "Greedy Function Approximation: A Gradient Boosting
// Machine", Annals of Statistics 29(5):1189–1232, 2001.
type GradientBoostingClassifier struct {
	cfg       gbmConfig
	init      float64 // F_0 = log-odds of the base rate
	nFeat     int     // feature count seen in Fit, for the predict width guard
	trees     []*gbmTree
	trainLoss []float64 // per-round training log loss (length nEstimators+1)
	fitted    bool
}

// NewGradientBoostingClassifier builds a binary classifier with the given
// options applied over the sklearn-matching defaults (100 trees,
// learningRate 0.1, maxDepth 3, minSamplesLeaf 1, subsample 1.0, seed 0).
func NewGradientBoostingClassifier(opts ...GBMOption) *GradientBoostingClassifier {
	cfg := defaultGBMConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return &GradientBoostingClassifier{cfg: cfg}
}

// Fit trains the additive log-odds model on X[n][d] with binary labels y[n],
// each 0 or 1. It errors on empty/ragged input, a bad label, a degenerate
// single-class target (log-odds undefined), or an out-of-range hyper-parameter.
func (m *GradientBoostingClassifier) Fit(x [][]float64, y []int) error {
	n, d, err := validX(x, len(y))
	if err != nil {
		return err
	}
	if err := m.cfg.validate(n); err != nil {
		return err
	}
	m.nFeat = d
	yf := make([]float64, n)
	var pos float64
	for i, v := range y {
		switch v {
		case 0:
			yf[i] = 0
		case 1:
			yf[i] = 1
			pos++
		default:
			return fmt.Errorf("classic: binary classifier needs labels 0/1, got %d", v)
		}
	}
	p := pos / float64(n)
	if p <= 0 || p >= 1 {
		return fmt.Errorf("classic: classifier needs both classes present (base rate %g)", p)
	}
	m.init = math.Log(p / (1 - p))
	f := make([]float64, n)
	for i := range f {
		f[i] = m.init
	}
	m.trees = m.trees[:0]
	m.trainLoss = m.trainLoss[:0]
	m.trainLoss = append(m.trainLoss, logLoss(yf, f))

	rng := rand.New(rand.NewSource(m.cfg.seed))
	resid := make([]float64, n)
	var gb gbmGrower
	if m.cfg.nEstimators > 0 {
		gb = newGBMGrower(m.cfg, x, n, d)
	}
	for round := 0; round < m.cfg.nEstimators; round++ {
		// negative gradient of log loss: resid = y − σ(f). Vectorize σ over the whole
		// score vector (recomputed every boosting round — ~18% of GBM fit); σ writes
		// into resid, then the subtraction runs in place. ~1 ulp, well inside the GBM
		// goldens' 1e-2 R² tolerance.
		simd.SigmoidF64(resid, f)
		for i := 0; i < n; i++ {
			resid[i] = yf[i] - resid[i]
		}
		idx := subsampleIdx(n, m.cfg.subsample, rng)
		tree := gb.grow(resid, idx)
		addTreeScore(f, gb, tree, x, m.cfg.learningRate, len(idx) == n)
		m.trees = append(m.trees, tree)
		m.trainLoss = append(m.trainLoss, logLoss(yf, f))
	}
	m.fitted = true
	return nil
}

// decision returns the raw log-odds F(x) for each row.
func (m *GradientBoostingClassifier) decision(x [][]float64) []float64 {
	return gbmPredictSum(m.init, m.cfg.learningRate, m.trees, x)
}

// PredictProba returns P(y=1|x) = σ(F(x)) for each row. It errors before Fit.
func (m *GradientBoostingClassifier) PredictProba(x [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: GradientBoostingClassifier.PredictProba before Fit")
	}
	if err := gbmWidthCheck(x, m.nFeat, "GradientBoostingClassifier.PredictProba"); err != nil {
		return nil, err
	}
	f := m.decision(x)
	out := make([]float64, len(f))
	//perfscan:ignore PS4003 output sigmoid over n, dominated by decision() tree traversal
	for i, v := range f {
		out[i] = sigmoid(v)
	}
	return out, nil
}

// Predict returns the hard 0/1 label for each row (σ(F) ≥ 0.5 ⇒ 1). It errors
// before Fit.
func (m *GradientBoostingClassifier) Predict(x [][]float64) ([]int, error) {
	if !m.fitted {
		return nil, fmt.Errorf("classic: GradientBoostingClassifier.Predict before Fit")
	}
	if err := gbmWidthCheck(x, m.nFeat, "GradientBoostingClassifier.Predict"); err != nil {
		return nil, err
	}
	f := m.decision(x)
	out := make([]int, len(f))
	for i, v := range f {
		if v >= 0 { // σ(v) ≥ 0.5 ⇔ v ≥ 0
			out[i] = 1
		}
	}
	return out, nil
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

func logLoss(y, f []float64) float64 {
	// Mean binary cross-entropy. −(y·log σ(f) + (1−y)·log(1−σ(f))) = softplus((1−2y)·f)
	// exactly (y∈{0,1}), so one vectorized softplus per sample replaces the sigmoid +
	// two logs the old form recomputed every boosting round (~15% of GBM fit as scalar
	// math.Log). No eps clamp needed — softplus is stable at both tails. This is the
	// monitoring loss curve (predictions come from the trees), so the ~1-ulp SIMD shift
	// is invisible to the sklearn parity and keeps the curve monotonic.
	return simd.SoftplusNegLLSumF64(f, y) / float64(len(y))
}

// gbmRadixByKey sorts col ascending by key[col[·]] with an 8-pass LSD radix over the
// ORDER-PRESERVING bit image of the float64 — negatives become ^bits, non-negatives
// bits|signbit — which is monotonic in the float value, so an unsigned radix orders it.
// The same transform the CART builder's radixByFeature uses; GBM's presort never got it.
//
// O(n) per pass against pdqsort's O(n log n) comparisons, and the comparisons it
// replaces were the expensive kind: an indirection through col into a key column. Scratch
// is caller-owned so the d presorts share one allocation.
//
// Ties: equal keys keep their relative input order within a pass, and the input is the
// identity permutation, so equal values come out in ascending id order — at least as
// well-defined as the unstable comparison sort it replaces. Split thresholds sit between
// DISTINCT adjacent values, so tie order does not move any split either way.
func gbmRadixByKey(col []int, key []float64, k, tmpK []uint64, tmpI []int) {
	n := len(col)
	k = k[:n]
	for i, id := range col {
		u := math.Float64bits(key[id])
		if u&(1<<63) != 0 {
			u = ^u
		} else {
			u |= 1 << 63
		}
		k[i] = u
	}
	src, dst := col, tmpI[:n]
	srcK, dstK := k, tmpK[:n]
	var count [256]int
	//perfscan:ignore PS3078 stale line past EOF; radix presort already optimized
	for shift := uint(0); shift < 64; shift += 8 {
		count = [256]int{}
		for _, u := range srcK {
			count[(u>>shift)&0xff]++
		}
		sum := 0
		for i := range count {
			c := count[i]
			count[i] = sum
			sum += c
		}
		for i, u := range srcK {
			bkt := (u >> shift) & 0xff
			p := count[bkt]
			count[bkt]++
			dst[p] = src[i]
			dstK[p] = u
		}
		src, dst = dst, src
		srcK, dstK = dstK, srcK
	}
	// 8 (even) passes ⇒ src is the caller's col again, holding the sorted ids.
}
