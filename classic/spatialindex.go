package classic

import (
	"math"
	"slices"
	"sort"
)

// ballTree is an exact spatial index over a fixed set of training points,
// built once in O(n log n) and reused for every query. It answers two kinds of
// proximity question far faster than the brute-force O(n) scan the classic
// estimators used before:
//
//   - kNN(query, k): the k nearest training points (ranked by ascending
//     distance, ties broken by ascending training index);
//   - radius(query, eps): every training point within eps of the query.
//
// A ball tree recursively partitions the points into nested hyper-spheres
// ("balls"): each node stores the centroid and radius of the smallest sphere
// enclosing its points. A query prunes an entire subtree whenever the sphere's
// nearest possible point is farther than the current search bound (the
// triangle inequality: for any point p inside a ball of centre c and radius r,
// dist(q,p) ≥ dist(q,c) − r). Unlike a KD-tree, the balls adapt to the data's
// shape, so pruning stays effective at the d≈20 dimensionalities where axis-
// aligned KD-tree splits collapse to brute force (Omohundro, "Five Balltree
// Construction Algorithms", 1989).
//
// The index is EXACT: it returns exactly the same neighbours a brute-force scan
// would, including the same tie-breaks. Pruning is done conservatively (a tiny
// slack on the bound), so floating-point rounding can only ever make the search
// visit a few extra nodes — never skip one that could hold a qualifying point.
// Every candidate point is then scored with the identical distance/predicate
// the brute-force path uses, so the selected set is bit-identical.
type ballTree struct {
	pts    [][]float64
	metric ballMetric
	root   *ballNode

	// splitKey is point-id-indexed scratch for the median-split sort: filling it once
	// per node (O(m)) hoists the scattered pts[id][splitDim] load out of the O(m log m)
	// comparator, which otherwise pays a row-pointer load plus an index per comparison.
	// One allocation for the whole build, reused down the recursion.
	splitKey []float64
	// spreadLo/spreadHi are d-length builder scratch for the split-dimension choice, filled by the
	// SAME row-major pass that accumulates the centroid. The scan they replace made one pass per
	// DIMENSION over pts[i][j] with j fixed — a column walk over a slice-of-slices, so it
	// re-dereferenced every row header d times and touched a fresh cache line per row to read eight
	// bytes. enclose already loads each row contiguously; min/max ride along for free.
	spreadLo []float64
	spreadHi []float64
}

// ballMetric selects the distance the tree measures with. It mirrors the two
// metrics the classic estimators expose (L2 and L1); both obey the triangle
// inequality, which the pruning relies on.
type ballMetric int

const (
	ballL2 ballMetric = iota // Euclidean, √Σ(aᵢ−bᵢ)²
	ballL1                   // Manhattan, Σ|aᵢ−bᵢ|
)

// ballLeafSize caps the number of points held directly in a leaf. Below this
// the linear scan of the leaf is cheaper than deeper tree structure; it also
// matches scikit-learn's default leaf_size=40 in spirit (we use a smaller value
// so even the modest golden datasets exercise real internal nodes).
const ballLeafSize = 16

type ballNode struct {
	centroid []float64
	radius   float64
	idx      []int // point indices held in this node (leaf only)
	left     *ballNode
	right    *ballNode
}

// buildBallTree constructs the index over pts under the given metric. It returns
// nil when n ≤ ballLeafSize: for such small sets a single brute-force scan is
// already optimal and the callers fall back to it, which keeps behaviour
// identical while skipping needless structure (the "auto" fallback).
func buildBallTree(pts [][]float64, metric ballMetric) *ballTree {
	if len(pts) <= ballLeafSize {
		return nil
	}
	bt := &ballTree{pts: pts, metric: metric, splitKey: make([]float64, len(pts))}
	idx := make([]int, len(pts))
	for i := range idx {
		idx[i] = i
	}
	bt.root = bt.build(idx)
	return bt
}

// dist returns the point-to-point distance under the tree's metric. For L2 it
// is √Σ(aᵢ−bᵢ)², identical to knnDist(KNNEuclidean) and to the square root of
// DBSCAN's squared comparison — so neighbours the tree scores match the brute
// paths exactly.
func (bt *ballTree) dist(a, b []float64) float64 {
	b = b[:len(a)] // discharges the per-iteration bounds check on b[i]; see distSq
	switch bt.metric {
	case ballL1:
		var s float64
		for i := range a {
			s += math.Abs(a[i] - b[i])
		}
		return s
	default: // ballL2
		var s float64
		for i := range a {
			d := a[i] - b[i]
			s += d * d
		}
		return math.Sqrt(s)
	}
}

// distSq is the MONOTONE ranking distance — L1 as-is, L2 WITHOUT the sqrt (Σd²). The kNN
// search compares and heaps neighbours only by rank, so ordering by distSq is identical to
// ordering by dist, but skips the per-leaf-point sqrt (the profile's 61% hot path). toDist
// converts back to the real distance for the pruning bound (per node) and the final k
// results (per query) — sqrt(Σd²) is bit-identical to dist's, so the returned neighbours and
// distances are unchanged. For L1 both are identities.
func (bt *ballTree) distSq(a, b []float64) float64 {
	// Ranging over a while indexing b leaves prove no relation between i and len(b), so
	// every iteration carried a live bounds check on b[i] (confirmed with
	// -d=ssa/check_bce/debug=1). Clamping b to a's length once discharges it. Callers pass
	// two rows of the same width, so this cannot truncate; a shorter b panicked on the index
	// before and panics on this slice now, one iteration earlier.
	b = b[:len(a)]
	if bt.metric == ballL1 {
		var s float64
		for i := range a {
			s += math.Abs(a[i] - b[i])
		}
		return s
	}
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func (bt *ballTree) toDist(sq float64) float64 {
	if bt.metric == ballL1 {
		return sq
	}
	return math.Sqrt(sq)
}

// build recursively partitions idx into a ball-tree node. It splits on the
// dimension of largest spread at the median coordinate — a standard,
// deterministic construction that keeps the two children balanced.
func (bt *ballTree) build(idx []int) *ballNode {
	n := &ballNode{}
	willSplit := len(idx) > ballLeafSize
	n.centroid, n.radius = bt.enclose(idx, willSplit)
	if !willSplit {
		n.idx = idx
		return n
	}
	// Choose the split dimension: the one with the greatest coordinate spread. The per-dimension
	// lo/hi came out of enclose's own row-major pass above, so this is a d-length scan rather than d
	// full passes over the node's points. Bit-identical: c[j] still sums i ascending over the same
	// operands, min/max are order-insensitive, and the strict-> argmax still scans j ascending, so
	// the same split dimension is chosen.
	d := len(bt.pts[idx[0]])
	splitDim, bestSpread := 0, -1.0
	for j := 0; j < d; j++ {
		if s := bt.spreadHi[j] - bt.spreadLo[j]; s > bestSpread {
			bestSpread, splitDim = s, j
		}
	}
	if bestSpread == 0 { // all points coincide — nothing to split on
		n.idx = idx
		return n
	}
	// Hoist the scattered load out of the comparator (see ballTree.splitKey). The
	// comparator stays the SAME PREDICATE — key[id] == pts[id][splitDim] — so pdqsort
	// returns the same permutation on the same input, ties included.
	key := bt.splitKey
	for _, id := range idx {
		key[id] = bt.pts[id][splitDim]
	}
	// A median PARTITION, not a full sort. build only reads idx[:mid] and idx[mid:] and then
	// recurses, and each child immediately re-partitions its own half — so the ordering a full sort
	// establishes inside each half is thrown away before anything reads it. Sorting every node costs
	// sum over nodes of m*log(m) comparisons, each doing two scattered loads into key; selecting the
	// median costs about 2m per node.
	//
	// Three-way (Dutch national flag) partitioning rather than two-way: the split key is a raw
	// coordinate, so duplicate values are common — the equivalence fixture is about one-sixth exact
	// duplicates — and a two-way partition degrades toward quadratic on them.
	//
	// Same contract as the sort it replaces: every element before mid compares <= the element at
	// mid, so the same median split and the same child sizes. Ties may land in a different child than
	// pdqsort put them, which this file already licenses above — the kNN search is exact and orders
	// its result by (dist, idx), so the SAME k neighbours come back whatever shape the tree takes.
	mid := len(idx) / 2
	nthByKey(idx, key, mid)
	n.left = bt.build(idx[:mid])
	n.right = bt.build(idx[mid:])
	return n
}

// nthByKey rearranges idx in place so that idx[k] holds an element with the k-th smallest key, every
// element before k has a key <= it, and every element after has a key >= it. This is the partition
// half of a sort: ballTree.build needs the median boundary and nothing else about the order.
//
// Three-way partitioning keeps it linear when keys repeat, which they do here — the split key is one
// coordinate of the data, so exact duplicates are ordinary rather than exceptional.
func nthByKey(idx []int, key []float64, k int) {
	lo, hi := 0, len(idx)-1
	for lo < hi {
		pivot := key[idx[lo+(hi-lo)/2]]
		lt, gt, i := lo, hi, lo
		for i <= gt {
			switch v := key[idx[i]]; {
			case v < pivot:
				idx[lt], idx[i] = idx[i], idx[lt]
				lt++
				i++
			case v > pivot:
				idx[gt], idx[i] = idx[i], idx[gt]
				gt--
			default:
				i++
			}
		}
		switch {
		case k < lt:
			hi = lt - 1
		case k > gt:
			lo = gt + 1
		default:
			return // k lands inside the equal-to-pivot run: already in place
		}
	}
}

// enclose returns the centroid (componentwise mean) of the given points and the
// radius of the smallest enclosing ball about that centroid under the metric.
func (bt *ballTree) enclose(idx []int, spread bool) ([]float64, float64) {
	d := len(bt.pts[idx[0]])
	c := make([]float64, d)
	if spread {
		// Only for nodes that will actually split; a leaf never consults the spread, so it keeps the
		// plain accumulation below and pays nothing for this.
		if cap(bt.spreadLo) < d {
			bt.spreadLo, bt.spreadHi = make([]float64, d), make([]float64, d)
		}
		lo, hi := bt.spreadLo[:d], bt.spreadHi[:d]
		for j := 0; j < d; j++ {
			lo[j], hi[j] = math.Inf(1), math.Inf(-1)
		}
		for _, i := range idx {
			p := bt.pts[i]
			for j := 0; j < d; j++ {
				v := p[j]
				c[j] += v
				if v < lo[j] {
					lo[j] = v
				}
				if v > hi[j] {
					hi[j] = v
				}
			}
		}
	} else {
		for _, i := range idx {
			p := bt.pts[i]
			for j := 0; j < d; j++ {
				c[j] += p[j]
			}
		}
	}
	inv := 1.0 / float64(len(idx))
	for j := 0; j < d; j++ {
		c[j] *= inv
	}
	// sqrt is monotone + correctly-rounded, so max_i sqrt(distSq_i) == sqrt(max_i distSq_i)
	// bit-for-bit — rank by squared distance and take a single toDist, not one sqrt per point.
	var rsq float64
	for _, i := range idx {
		if dd := bt.distSq(c, bt.pts[i]); dd > rsq {
			rsq = dd
		}
	}
	return c, bt.toDist(rsq)
}

// pruneSlack is a tiny relative+absolute tolerance added to every pruning bound
// so floating-point rounding of the centroid distance and radius can never make
// the search prune a subtree that might still hold a qualifying point. Erring
// this way only ever visits a few extra nodes; the exact per-point test then
// discards any non-neighbour, keeping results bit-identical to brute force.
const pruneSlack = 1e-9

// --- k-nearest-neighbours -----------------------------------------------------

// kNN returns the k nearest points to query, ranked by ascending distance with
// ties broken by ascending index — identical to a full brute-force sort by
// (dist, idx) truncated to k. It assumes k ≤ n.
func (bt *ballTree) kNN(query []float64, k int) []neighbour {
	// items is sized to k up front. Left nil it grew by doubling inside consider — log2(k)
	// allocations per QUERY plus the overshoot of the last doubling — which an alloc_space
	// profile put at 47.6% of the bytes a KNN predict allocates, the largest single site.
	// The heap never exceeds k members by construction, so one allocation of exactly k both
	// removes the growth and wastes nothing.
	return bt.kNNInto(query, k, &knnHeap{k: k, items: make([]neighbour, 0, k)})
}

// kNNInto is kNN against a caller-owned heap, so a loop over many queries pays for the
// heap once rather than once per query. h is reset here; its items slice keeps whatever
// capacity it already had. The returned slice ALIASES h.items and is therefore only valid
// until the next kNNInto call on the same h — callers must consume it before reusing the
// scratch, which every caller in this package does within one iteration.
//
// The search itself is untouched: same reset state (empty, capacity ≥ k), same visit order,
// same comparator, same distance conversion, so the result is bit-identical to kNN.
func (bt *ballTree) kNNInto(query []float64, k int, h *knnHeap) []neighbour {
	h.k = k
	h.items = h.items[:0]
	if bt.root != nil {
		bt.searchKNN(bt.root, query, h, bt.distSq(query, bt.root.centroid))
	}
	out := h.items
	// out[].dist holds distSq (monotone in dist), so the (dist,idx) sort order is identical.
	// Same swap-allocation fix, once per QUERY here. This comparator is a TOTAL order —
	// (dist, idx) with idx unique — so stability is irrelevant and the permutation is
	// identical, not merely equivalent.
	slices.SortFunc(out, func(a, b neighbour) int {
		switch {
		case a.dist != b.dist:
			if a.dist < b.dist {
				return -1
			}
			return 1
		case a.idx < b.idx:
			return -1
		case a.idx > b.idx:
			return 1
		}
		return 0
	})
	for i := range out { // convert the k results back to real distances for the caller
		out[i].dist = bt.toDist(out[i].dist)
	}
	return out
}

// searchKNN takes dCent = bt.distSq(query, n.centroid) precomputed by the caller.
// Every internal node already computes both children's centroid distSq to order the
// visit, so threading that value in reuses it for the child's own prune test instead
// of recomputing a full d-dimensional distSq per node visit. Bit-identical: dCent is
// the exact value the recompute produced (same distSq call, same operands, same
// deterministic summation order).
func (bt *ballTree) searchKNN(n *ballNode, query []float64, h *knnHeap, dCent float64) {
	if n == nil {
		return
	}
	// Prune: nearest possible point in this ball is dist(query,centroid)−radius. The
	// heap holds distSq, so convert both sides to real distances for the radius bound.
	if h.full() {
		minDist := bt.toDist(dCent) - n.radius
		if minDist > bt.toDist(h.worst())*(1+pruneSlack)+pruneSlack {
			return
		}
	}
	if n.idx != nil { // leaf: score every point exactly (rank by distSq, no per-point sqrt)
		for _, i := range n.idx {
			h.consider(neighbour{dist: bt.distSq(query, bt.pts[i]), idx: i})
		}
		return
	}
	// Visit the child whose centroid is nearer first, tightening the bound
	// sooner and so pruning the far child more often. distSq orders identically.
	dl := bt.distSq(query, n.left.centroid)
	dr := bt.distSq(query, n.right.centroid)
	if dl <= dr {
		bt.searchKNN(n.left, query, h, dl)
		bt.searchKNN(n.right, query, h, dr)
	} else {
		bt.searchKNN(n.right, query, h, dr)
		bt.searchKNN(n.left, query, h, dl)
	}
}

// knnHeap keeps the k best neighbours seen so far as a binary max-heap keyed by
// (dist, idx): the root is the current worst member, so it is what a new
// candidate must beat and what bounds the search. Ordering by (dist, idx)
// reproduces the brute-force tie-break (nearest ties → lowest training index).
type knnHeap struct {
	k     int
	items []neighbour
}

func (h *knnHeap) full() bool     { return len(h.items) >= h.k }
func (h *knnHeap) worst() float64 { return h.items[0].dist }

// worseThan reports whether a ranks strictly after b under (dist, idx) order —
// i.e. a is a "larger" (worse) neighbour than b.
func worseThan(a, b neighbour) bool {
	if a.dist != b.dist {
		return a.dist > b.dist
	}
	return a.idx > b.idx
}

func (h *knnHeap) consider(c neighbour) {
	if len(h.items) < h.k {
		h.items = append(h.items, c)
		h.up(len(h.items) - 1)
		return
	}
	if worseThan(h.items[0], c) { // c is better than the current worst
		h.items[0] = c
		h.down(0)
	}
}

func (h *knnHeap) up(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !worseThan(h.items[i], h.items[parent]) {
			break
		}
		h.items[i], h.items[parent] = h.items[parent], h.items[i]
		i = parent
	}
}

func (h *knnHeap) down(i int) {
	n := len(h.items)
	for {
		l, r, largest := 2*i+1, 2*i+2, i
		if l < n && worseThan(h.items[l], h.items[largest]) {
			largest = l
		}
		if r < n && worseThan(h.items[r], h.items[largest]) {
			largest = r
		}
		if largest == i {
			break
		}
		h.items[i], h.items[largest] = h.items[largest], h.items[i]
		i = largest
	}
}

// --- radius / eps-neighbourhood ----------------------------------------------

// radius appends the indices of every point within eps of query to dst and
// returns the extended slice, in ascending index order. For L2 the membership
// test compares squared distances to eps² exactly as the brute-force DBSCAN
// scan does (avoiding a square root); for L1 it compares the summed absolute
// differences to eps. Pruning uses the true metric distance to the ball, so the
// returned set is bit-identical to the brute-force neighbourhood.
func (bt *ballTree) radius(query []float64, eps float64, dst []int) []int {
	dst = dst[:0]
	dst = bt.searchRadius(bt.root, query, eps, eps*eps, dst)
	sort.Ints(dst)
	return dst
}

func (bt *ballTree) searchRadius(n *ballNode, query []float64, eps, eps2 float64, dst []int) []int {
	if n == nil {
		return dst
	}
	// Prune when the ball's nearest possible point is already beyond eps.
	minDist := bt.dist(query, n.centroid) - n.radius
	if minDist > eps*(1+pruneSlack)+pruneSlack {
		return dst
	}
	if n.idx != nil { // leaf: test every point with DBSCAN's exact predicate
		for _, i := range n.idx {
			if bt.within(query, bt.pts[i], eps, eps2) {
				dst = append(dst, i)
			}
		}
		return dst
	}
	dst = bt.searchRadius(n.left, query, eps, eps2, dst)
	dst = bt.searchRadius(n.right, query, eps, eps2, dst)
	return dst
}

// within reports whether b lies within eps of a, using the same comparison the
// brute-force DBSCAN scan uses: squared distance ≤ eps² for L2, |·| sum ≤ eps
// for L1.
func (bt *ballTree) within(a, b []float64, eps, eps2 float64) bool {
	b = b[:len(a)] // discharges the per-iteration bounds check on b[i]; see distSq
	// Early-exit: the accumulator is monotonically non-decreasing (each term is ≥0),
	// so once it exceeds the threshold the point is out and the remaining dimensions
	// can only push it further out. Bailing returns the SAME boolean the full sum
	// would (a point that is within never trips the check, since its full sum ≤ thr),
	// so the neighbour set is bit-identical to the brute-force scan — it just skips
	// the tail dims of the far points that dominate a leaf test.
	switch bt.metric {
	case ballL1:
		var s float64
		for i := range a {
			s += math.Abs(a[i] - b[i])
			if s > eps {
				return false
			}
		}
		return true
	default: // ballL2
		// The bail-out is checked every FOUR dimensions rather than every one. A profile of the
		// DBSCAN fit put that single branch at 450ms against 30ms for the subtraction and square
		// it guards — it is the dominant cost of this function, not the arithmetic, because it is
		// a data-dependent branch the predictor cannot learn.
		//
		// Checking less often returns the SAME boolean: s never decreases, so if it exceeds the
		// threshold at some dimension it still exceeds it at the next checkpoint and at the end.
		// The accumulation itself is untouched — one accumulator, same order, same operands — so
		// s is bit-identical too, which is what the exact-label DBSCAN goldens require.
		var s float64
		i := 0
		for ; i+4 <= len(a); i += 4 {
			d0 := a[i] - b[i]
			s += d0 * d0
			d1 := a[i+1] - b[i+1]
			s += d1 * d1
			d2 := a[i+2] - b[i+2]
			s += d2 * d2
			d3 := a[i+3] - b[i+3]
			s += d3 * d3
			if s > eps2 {
				return false
			}
		}
		for ; i < len(a); i++ {
			d := a[i] - b[i]
			s += d * d
		}
		// NOT `s <= eps2`: with a NaN coordinate s is NaN, and the loop above never bailed
		// because NaN > eps2 is false, so the original returned true. !(s > eps2) reproduces
		// that; s <= eps2 would flip it.
		return !(s > eps2)
	}
}
