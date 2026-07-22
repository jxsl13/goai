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
//
// Split finding uses a presort/partition scheme (the classic CART speed-up,
// sklearn's old PresortBestSplitter). Each feature is argsorted ONCE at fit time
// into cols[f]; a node owns the contiguous range cols[f][start:end] and, on
// splitting, each column is stably partitioned in place into left|right so both
// children inherit already-sorted ranges. This turns the per-node cost from
// O(d·n log n) (re-sorting every feature at every node) into O(d·n), and removes
// the sort.Slice closure/reflection overhead — while producing bit-identical
// splits, because thresholds only ever fall between DISTINCT adjacent values, so
// the ordering of tied values never affects a split decision.
type cartBuilder struct {
	x          [][]float64
	yi         []int     // class indices (classification)
	yf         []float64 // targets (regression)
	nClasses   int
	regression bool
	cfg        cartConfig
	rng        *lcg

	// presort/partition scratch, allocated once per fit and reused everywhere.
	cols    [][]int   // cols[f][start:end] = current node's samples, sorted asc by feature f
	part    []int     // scratch for the stable partition (len n)
	goLeft  []bool    // per-sample split membership during a partition (len n)
	totCnt  []int     // reused class-count buffer (len nClasses); classification only
	leftCnt []int     // reused running left-count buffer (len nClasses); classification only
	clogc   []float64 // Entropy only: clogc[c] = c·ln c cache (counts are integers), so a
	// node's weighted entropy is clogc[n] − Σ clogc[countₖ] with no per-split math.Log.
	sweepVals []float64 // sweep scratch: a node's feature values in sorted order (len n)
	allFeats  []int     // reused ascending [0..d) for the all-features split path
	featPool  []int     // reused pool for feature subsampling (maxFeatures>0)
	featSub   []int     // reused subsample result (maxFeatures>0)
	sortBuf   []int     // reused per-feature sort scratch for the subsampled path
	// radix-sort scratch (reused): keys = order-preserving u64 of the feature value,
	// tmpI/tmpK = ping-pong buffers for the 8-pass LSD radix (replaces the sort.Slice
	// closure sort — the split-search's dominant cost).
	radixKeys []uint64
	radixTmpI []int
	radixTmpK []uint64
	// keyByID[id] = the current feature's value for sample id, filled once before a
	// sub-cutoff comparison sort so its comparator reads one contiguous float instead
	// of chasing b.x[id] (a scattered [][]float64 row pointer) on every comparison.
	keyByID []float64
}

// subsampled reports whether per-split feature subsampling is active (the random
// forest case). When it is, only maxFeatures≈√d of the d columns are examined at
// each node, so the presort/partition scheme — which must keep ALL d columns
// validly partitioned at every node — costs more than it saves. The builder
// therefore falls back to the per-node sort-of-the-sampled-features path
// (buildIdx/bestSplitIdx) for forests, and uses presort/partition only when all
// features are considered (single trees, and every GBM weak learner).
func (b *cartBuilder) subsampled(d int) bool {
	return b.cfg.maxFeatures > 0 && b.cfg.maxFeatures < d
}

// initIdx allocates the reusable scratch the per-node (subsampled) split finder
// needs: the class-count buffers and a single sort buffer reused across features.
func (b *cartBuilder) initIdx(n int) {
	if !b.regression {
		b.totCnt = make([]int, b.nClasses)
		b.leftCnt = make([]int, b.nClasses)
	}
	b.sortBuf = make([]int, n)
	b.radixKeys = make([]uint64, n)
	b.radixTmpI = make([]int, n)
	b.radixTmpK = make([]uint64, n)
	b.sweepVals = make([]float64, n)
	b.keyByID = make([]float64, n)
	b.buildCLogC(n)
}

// buildCLogC caches clogc[c] = c·ln c for c∈[0,n] when the Entropy criterion is active
// (counts are integers ≤ n), so the impurity kernels evaluate no math.Log per candidate
// split — the log was ~52% of an entropy tree fit. No-op for Gini/regression.
func (b *cartBuilder) buildCLogC(n int) {
	if b.regression || b.cfg.criterion != Entropy {
		return
	}
	b.clogc = make([]float64, n+1)
	for c := 1; c <= n; c++ {
		b.clogc[c] = float64(c) * math.Log(float64(c))
	}
}

// initColumns argsorts every feature once and allocates the reusable scratch the
// presort/partition split finder needs. Called once per Fit before build.
// treeRadixCutoff is the index count above which sorting `order` ascending by a
// feature value switches from the comparison sort to the 8-pass LSD radix. Below it
// the O(n log n) comparison sort's lower constant wins.
const treeRadixCutoff = 512

// radixByFeature sorts order ascending by b.x[order[i]][ff]. Tie order is
// unspecified — irrelevant to split decisions (thresholds sit between DISTINCT
// values, per initColumns/bestSplitIdx), so the split chosen is identical to the
// comparison sort's. Closure-free O(n) radix on the order-preserving u64 transform
// of the float64 bits (negatives → ^bits, non-negatives → bits|sign): monotonic in
// the float value, so a plain unsigned radix orders it. Uses the reused scratch.
func (b *cartBuilder) radixByFeature(order []int, ff int) {
	n := len(order)
	if n < treeRadixCutoff {
		// Hoist the scattered b.x[id][ff] loads out of the O(n log n) comparator: fill a
		// contiguous id-indexed key once (O(n)), then compare those. Tie order stays
		// unspecified (irrelevant — thresholds sit between distinct values), so the
		// chosen split is unchanged.
		kb := b.keyByID
		for _, id := range order {
			kb[id] = b.x[id][ff]
		}
		sort.Slice(order, func(a, c int) bool { return kb[order[a]] < kb[order[c]] })
		return
	}
	k := b.radixKeys[:n]
	for i, id := range order {
		u := math.Float64bits(b.x[id][ff])
		if u&(1<<63) != 0 {
			u = ^u
		} else {
			u |= 1 << 63
		}
		k[i] = u
	}
	src, dst := order, b.radixTmpI[:n]
	srcK, dstK := k, b.radixTmpK[:n]
	var count [256]int
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
	// 8 (even) passes ⇒ src is again the caller's `order` slice, now holding the
	// sorted indices; no final copy needed.
}

func (b *cartBuilder) initColumns(n, d int) {
	b.cols = make([][]int, d)
	base := make([]int, n*d) // single backing allocation for all d columns
	b.radixKeys = make([]uint64, n)
	b.radixTmpI = make([]int, n)
	b.radixTmpK = make([]uint64, n)
	b.keyByID = make([]float64, n) // sub-cutoff comparison-sort key scratch (radixByFeature)
	for f := 0; f < d; f++ {
		col := base[f*n : f*n+n : f*n+n]
		for i := range col {
			col[i] = i
		}
		ff := f
		// Order by feature value only. Tie order is irrelevant to split
		// decisions (thresholds sit between distinct values), so an unstable
		// sort is safe and reproduces the previous per-node sort's choices.
		b.radixByFeature(col, ff)
		b.cols[f] = col
	}
	b.part = make([]int, n)
	b.goLeft = make([]bool, n)
	if !b.regression {
		b.totCnt = make([]int, b.nClasses)
		b.leftCnt = make([]int, b.nClasses)
	}
	b.sweepVals = make([]float64, n)
	b.buildCLogC(n)
	b.allFeats = make([]int, d)
	for i := range b.allFeats {
		b.allFeats[i] = i
	}
	if b.cfg.maxFeatures > 0 && b.cfg.maxFeatures < d {
		b.featPool = make([]int, d)
		b.featSub = make([]int, b.cfg.maxFeatures)
	}
}

// partition splits the node range [start,end) of every feature column into
// left|right in place (stable), by the chosen (feat, thr) test, and returns the
// left/right boundary mid. After it returns, cols[f][start:mid] holds the left
// child's samples sorted by f, and cols[f][mid:end] the right child's — for all
// f — so recursion just re-uses the sub-ranges.
func (b *cartBuilder) partition(start, end, feat int, thr float64) int {
	d := len(b.cols)
	col := b.cols[feat]
	for p := start; p < end; p++ {
		s := col[p]
		b.goLeft[s] = b.x[s][feat] <= thr
	}
	mid := start
	for f := 0; f < d; f++ {
		cf := b.cols[f]
		w := start
		r := 0
		for p := start; p < end; p++ {
			s := cf[p]
			if b.goLeft[s] {
				cf[w] = s
				w++
			} else {
				b.part[r] = s
				r++
			}
		}
		copy(cf[w:end], b.part[:r])
		mid = w
	}
	return mid
}

// allIndices returns the slice [0, 1, …, n-1].
func allIndices(n int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	return idx
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

// build grows the tree over the node's sample range cols[·][start:end] at the
// given depth. The three column-order-independent aggregates (mean/majority/
// impurity) read column 0's sub-range as the node's sample set.
func (b *cartBuilder) build(start, end, depth int) *cartNode {
	samples := b.cols[0][start:end]
	n := end - start
	node := &cartNode{leaf: true}
	if b.regression {
		node.value = b.mean(samples)
	} else {
		node.predClass = b.majority(samples)
	}

	// Stopping rules mirror scikit-learn's DepthFirstTreeBuilder (§C21).
	pure := b.impurity(samples) <= 1e-12
	if pure ||
		(b.cfg.maxDepth >= 0 && depth >= b.cfg.maxDepth) ||
		n < b.cfg.minSamplesSplit ||
		n < 2*b.cfg.minSamplesLeaf {
		return node
	}

	feat, thr, ok := b.bestSplit(start, end)
	if !ok {
		return node // no admissible split (e.g. constant features)
	}

	mid := b.partition(start, end, feat, thr)
	node.leaf = false
	node.feature = feat
	node.threshold = thr
	node.left = b.build(start, mid, depth+1)
	node.right = b.build(mid, end, depth+1)
	return node
}

// buildIdx is the per-node split-finding path used when feature subsampling is
// active (random forests). It carries an explicit sample-index slice and sorts
// only the sampled features at each node (cheaper than maintaining all d presort
// columns when only √d are examined per split). Split decisions and the RNG
// feature-subsample sequence are identical to the original builder, so forest
// goldens stay bit-exact; only the sort/count buffers are now reused.
func (b *cartBuilder) buildIdx(idx []int, depth int) *cartNode {
	n := len(idx)
	node := &cartNode{leaf: true}
	if b.regression {
		node.value = b.mean(idx)
	} else {
		node.predClass = b.majority(idx)
	}

	pure := b.impurity(idx) <= 1e-12
	if pure ||
		(b.cfg.maxDepth >= 0 && depth >= b.cfg.maxDepth) ||
		n < b.cfg.minSamplesSplit ||
		n < 2*b.cfg.minSamplesLeaf {
		return node
	}

	feat, thr, ok := b.bestSplitIdx(idx)
	if !ok {
		return node
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
	node.left = b.buildIdx(left, depth+1)
	node.right = b.buildIdx(right, depth+1)
	return node
}

// bestSplitIdx finds the best split over an explicit index slice, sorting each
// sampled feature into the reused sortBuf. Same tie-breaks as bestSplit.
func (b *cartBuilder) bestSplitIdx(idx []int) (feat int, thr float64, ok bool) {
	d := len(b.x[0])
	bestCost := math.Inf(1)
	minLeaf := b.cfg.minSamplesLeaf
	for _, f := range b.candidateFeatures(d) {
		order := b.sortBuf[:len(idx)]
		copy(order, idx)
		ff := f
		b.radixByFeature(order, ff)
		cost, cut, found := b.sweep(order, f, minLeaf)
		if found && cost < bestCost {
			bestCost = cost
			feat = f
			a := b.x[order[cut-1]][f]
			c := b.x[order[cut]][f]
			thr = (a + c) / 2
			if thr == c || math.IsInf(thr, 0) {
				thr = a
			}
			ok = true
		}
	}
	return feat, thr, ok
}

// candidateFeatures returns the feature indices considered at a split: all of
// them, or a random maxFeatures-sized subset (ascending, so the split tie-break
// still favours the lowest feature index among the sampled set).
func (b *cartBuilder) candidateFeatures(d int) []int {
	if b.cfg.maxFeatures <= 0 || b.cfg.maxFeatures >= d {
		if len(b.allFeats) != d {
			b.allFeats = make([]int, d)
			for i := range b.allFeats {
				b.allFeats[i] = i
			}
		}
		return b.allFeats // reused ascending [0..d)
	}
	// partial Fisher–Yates draw of maxFeatures distinct indices. The RNG call
	// sequence is identical to the original (same intn args in the same order),
	// so forest goldens are unaffected; only the buffers are now reused.
	m := b.cfg.maxFeatures
	if len(b.featPool) != d {
		b.featPool = make([]int, d)
	}
	if len(b.featSub) != m {
		b.featSub = make([]int, m)
	}
	pool := b.featPool
	for i := range pool {
		pool[i] = i
	}
	for i := 0; i < m; i++ {
		j := i + b.rng.intn(d-i)
		pool[i], pool[j] = pool[j], pool[i]
	}
	copy(b.featSub, pool[:m])
	sort.Ints(b.featSub)
	return b.featSub
}

// bestSplit finds the (feature, threshold) minimising the weighted child
// impurity over the node range [start,end). Ties break to the lowest feature
// index then lowest threshold — the deterministic reduction of scikit-learn's
// tie-break on unique-optimum data. The per-feature order comes from the presort
// (cols[f]); no per-node sort is performed.
func (b *cartBuilder) bestSplit(start, end int) (feat int, thr float64, ok bool) {
	d := len(b.x[0])
	bestCost := math.Inf(1)
	minLeaf := b.cfg.minSamplesLeaf
	for _, f := range b.candidateFeatures(d) {
		order := b.cols[f][start:end]
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
	// Hoist this node's sorted feature values once (n gathers into b.x), then the sweep
	// reads them sequentially. The old inline b.x[order[p]][f] / b.x[order[p-1]][f]
	// re-gathered every value TWICE across adjacent iterations — halved to one here.
	vals := b.sweepVals[:n]
	for k := 0; k < n; k++ {
		vals[k] = b.x[order[k]][f]
	}
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
			if vals[p]-vals[p-1] <= featureThreshold {
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
	// classification — reuse the builder's count buffers (no per-node alloc)
	total := b.totCnt
	for k := range total {
		total[k] = 0
	}
	for _, i := range order {
		total[b.yi[i]]++
	}
	left := b.leftCnt
	for k := range left {
		left[k] = 0
	}
	if b.cfg.criterion != Entropy {
		// Gini: maintain Σleft[k]² and Σright[k]² incrementally. Moving one sample across
		// the split increments exactly one class count, so each sum updates in O(1) — the
		// left gains 2v+1 (=(v+1)²−v²) and the right loses 2·rc−1 — instead of the O(classes)
		// rescan weightedImpurityClf/Comp did at every candidate. The counts are integers
		// whose squared sums stay well under 2^53, so the running float64 totals equal the
		// freshly-summed ones bit-for-bit, and the cost is formed with the same op order as
		// weightedImpurityClf/Comp (pf·(1−Σ/pf²)); the chosen split is therefore identical to
		// the per-point rescan.
		var sumLeftSq, sumRightSq float64
		for k := range total {
			tf := float64(total[k])
			sumRightSq += tf * tf
		}
		for p := 1; p < n; p++ {
			c := b.yi[order[p-1]]
			v := left[c]
			rc := total[c] - v
			sumLeftSq += float64(2*v + 1)
			sumRightSq += float64(1 - 2*rc)
			left[c] = v + 1
			if p < minLeaf || n-p < minLeaf {
				continue
			}
			if vals[p]-vals[p-1] <= featureThreshold {
				continue
			}
			pf := float64(p)
			nrf := float64(n - p)
			cost := pf*(1-sumLeftSq/(pf*pf)) + nrf*(1-sumRightSq/(nrf*nrf))
			if cost < bestCost {
				bestCost, cut, found = cost, p, true
			}
		}
		return bestCost, cut, found
	}
	// entropy — per-split (the clogc terms are floats, so an incremental running sum
	// would not be bit-identical to the fresh Σ; keep the exact per-candidate rescan).
	for p := 1; p < n; p++ {
		left[b.yi[order[p-1]]]++
		if p < minLeaf || n-p < minLeaf {
			continue
		}
		if vals[p]-vals[p-1] <= featureThreshold {
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
	if b.cfg.criterion == Entropy {
		// n·H = clogc[n] − Σ clogc[cₖ] directly (matches weightedImpurityClfComp).
		w := b.clogc[n]
		for _, c := range counts {
			w -= b.clogc[c]
		}
		return w
	}
	return float64(n) * b.impurityCounts(counts, n)
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
		// Weighted entropy n·H = n·ln n − Σₖ cₖ·ln cₖ = clogc[n] − Σ clogc[cₖ]; the cₖ
		// are integer class counts, so this needs no per-split math.Log. Returned directly
		// (this function's contract is the n-weighted impurity), skipping the ×nf below.
		w := b.clogc[n]
		for k := range total {
			w -= b.clogc[total[k]-left[k]]
		}
		return w
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

// impurityCounts computes the (unweighted) impurity of a set given its class counts.
func (b *cartBuilder) impurityCounts(counts []int, n int) float64 {
	if n == 0 {
		return 0
	}
	nf := float64(n)
	if b.cfg.criterion == Entropy {
		// H = (n·ln n − Σ cₖ·ln cₖ)/n = (clogc[n] − Σ clogc[cₖ]) / n (no per-node math.Log).
		w := b.clogc[n]
		for _, c := range counts {
			w -= b.clogc[c]
		}
		return w / nf
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
	counts := b.totCnt
	for k := range counts {
		counts[k] = 0
	}
	for _, i := range idx {
		counts[b.yi[i]]++
	}
	return b.impurityCounts(counts, len(idx))
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
	counts := b.totCnt
	for k := range counts {
		counts[k] = 0
	}
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
	if b.subsampled(d) {
		b.initIdx(len(x))
		m.root = b.buildIdx(allIndices(len(x)), 0)
	} else {
		b.initColumns(len(x), d)
		m.root = b.build(0, len(x), 0)
	}
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
	if b.subsampled(d) {
		b.initIdx(len(x))
		m.root = b.buildIdx(allIndices(len(x)), 0)
	} else {
		b.initColumns(len(x), d)
		m.root = b.build(0, len(x), 0)
	}
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
