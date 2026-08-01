package classic

import (
	"math"
	"runtime"
	"sort"
)

// histRadixCutoff is the column length above which the per-feature quantile sort switches
// from sort.Float64s to the O(n) LSD radix (tree.go's proven transform). Below it the
// comparison sort's lower constant wins.
const histRadixCutoff = 512

// gbmHistSetupParWork gates the one-time parallel per-feature binning in newHistBuilder (n·d work).
const gbmHistSetupParWork = 1 << 17

// radixSortF64 sorts col ascending with an 8-pass LSD radix on the order-preserving u64
// transform of the float64 bits (negatives → ^bits, non-negatives → bits|sign — monotone in
// the value). keys/tmp are caller-provided scratch of length ≥ len(col), reused across
// features. newHistBuilder needs only the SORTED VALUES (the quantile positions), which are
// identical to sort.Float64s's (ties at a quantile hold the same value), so binning is
// unchanged. Assumes no NaN, as the tree radix does.
func radixSortF64(col []float64, keys, tmp []uint64) {
	n := len(col)
	for i, v := range col {
		u := math.Float64bits(v)
		if u&(1<<63) != 0 {
			u = ^u
		} else {
			u |= 1 << 63
		}
		keys[i] = u
	}
	src, dst := keys[:n], tmp[:n]
	var count [256]int
	for shift := uint(0); shift < 64; shift += 8 {
		count = [256]int{}
		for _, u := range src {
			count[(u>>shift)&0xff]++
		}
		sum := 0
		for i := range count {
			c := count[i]
			count[i] = sum
			sum += c
		}
		for _, u := range src {
			b := (u >> shift) & 0xff
			dst[count[b]] = u
			count[b]++
		}
		src, dst = dst, src
	}
	for i, u := range src { // src holds the sorted keys (8 passes → back in keys); invert
		if u&(1<<63) != 0 {
			u &^= 1 << 63
		} else {
			u = ^u
		}
		col[i] = math.Float64frombits(u)
	}
}

// histBuilder is the histogram weak-learner grower for gradient boosting — the
// fast large-N path, opt-in via WithGBMHistogram. It bins every feature into
// nbins quantile bins ONCE (the order depends only on X, never on the per-round
// residual), then at each node accumulates the round's residual into per-bin
// gradient histograms and scans BIN BOUNDARIES for the best variance-reduction
// split. Node cost is O(m·d + d·bins) — the split scan reads ≤bins values per
// feature instead of the exact builder's O(d·m) sorted walk, and the histogram
// build streams the compact uint16 bin codes cache-resident where the exact
// path's per-node sorted-column reads thrash L3 past ~L3/N. Measured 5.85× on a
// 60k×20 fit at identical holdout accuracy.
//
// The chosen threshold is a bin edge (a data quantile), not the exact midpoint
// between adjacent distinct values, so the trees are NOT bit-identical to the
// exact gbmBuilder — but with 256 bins on continuous features the split set is a
// dense subset of the exact candidates and the boosted model matches within
// noise (that is precisely why histogram GBMs are the production default in
// LightGBM / sklearn HistGradientBoosting). It produces the same gbmTree shape,
// so prediction is unchanged. Same mean-of-residual leaves as gbmBuilder, so it
// is a drop-in for both the regressor and the classifier weak learners.
// histBin holds one histogram bin's gradient sum and sample count together. The
// scatter that fills the histogram (buildHist) and the sweep that reads it
// (buildNode) both touch a bin's sum AND count at the same index, so keeping them
// interleaved lands both in one cache line — half the line traffic of the old
// parallel []float64 / []int32 arrays. The 4 bytes of tail padding are cheaper than
// the second cache miss on the memory-bound scatter.
type histBin struct {
	sum float64
	cnt int32
}

type histBuilder struct {
	x                 [][]float64
	y                 []float64 // current round's target (residual); set per round via grow
	n, d              int
	maxDepth, minLeaf int
	nbins             int
	edges             [][]float64 // [d][≤nbins-1] ascending split thresholds
	binned            []uint16    // [n*d] row-major bin index (bin = #edges < x)
	hist              [][]histBin // [maxDepth+1][d*nbins] per-level (gradient sum, count) histograms
	idxbuf            []int       // reusable full-sample index scratch
	contrib           []float64   // [n] per-sample leaf value of the last grown tree (full-sample rounds)
	wantContrib       bool        // capture contrib this round (only when the round grows on all n samples)
}

// newHistBuilder bins every feature once and allocates the reusable scratch.
func newHistBuilder(x [][]float64, n, d, maxDepth, minLeaf, nbins int) *histBuilder {
	if nbins < 2 {
		nbins = 2
	}
	b := &histBuilder{x: x, n: n, d: d, maxDepth: maxDepth, minLeaf: minLeaf, nbins: nbins}
	b.edges = make([][]float64, d)
	b.binned = make([]uint16, n*d)
	// Bin every feature ONCE (sort → quantile edges → assign bin codes). Features are
	// independent — each writes only edges[f] and the strided binned[·*d+f] column — so fan the
	// one-time setup over GOMAXPROCS with PER-WORKER sort scratch (col/rkeys/rtmp). Bit-identical:
	// each feature's binning is deterministic and independent of the others.
	binFeatures := func(f0, f1 int) {
		col := make([]float64, n)
		var rkeys, rtmp []uint64
		if n >= histRadixCutoff {
			rkeys = make([]uint64, n)
			rtmp = make([]uint64, n)
		}
		for f := f0; f < f1; f++ {
			for i := 0; i < n; i++ {
				col[i] = x[i][f]
			}
			if n >= histRadixCutoff {
				radixSortF64(col, rkeys, rtmp)
			} else {
				sort.Float64s(col)
			}
			ne := nbins - 1
			ed := make([]float64, 0, ne)
			for q := 1; q <= ne; q++ {
				v := col[q*n/nbins]
				if len(ed) == 0 || v > ed[len(ed)-1] {
					ed = append(ed, v) // keep strictly ascending, drop duplicate quantiles
				}
			}
			b.edges[f] = ed
			for i := 0; i < n; i++ {
				// bin = #edges strictly < x[i][f]  (SearchFloat64s: first idx with ed[idx] ≥ x)
				b.binned[i*d+f] = uint16(sort.SearchFloat64s(ed, x[i][f]))
			}
		}
	}
	if d >= 2 && n*d >= gbmHistSetupParWork {
		nw := runtime.GOMAXPROCS(0)
		if nw > d {
			nw = d
		}
		_ = parallelBuild(nw, func(c int) error {
			binFeatures(c*d/nw, (c+1)*d/nw)
			return nil
		})
	} else {
		binFeatures(0, d)
	}
	// One histogram buffer per recursion level (the histogram-subtraction scheme
	// keeps the parent's histogram live while its children are grown).
	levels := maxDepth + 1
	b.hist = make([][]histBin, levels)
	for l := 0; l < levels; l++ {
		b.hist[l] = make([]histBin, d*nbins)
	}
	b.idxbuf = make([]int, n)
	b.contrib = make([]float64, n)
	return b
}

// leafNode makes a leaf and, on a full-sample round, records value as the tree's
// contribution for every sample reaching this leaf. The bin partition routes each
// training sample to the same leaf its float-threshold predict would (bin ≤ bestBin
// ⟺ x ≤ edges[bestBin]), so contrib[s] equals tree.root.predict(x[s]) exactly —
// letting the boosting loop skip the O(n·depth) retraversal (§C-gbm).
func (b *histBuilder) leafNode(idx []int, value float64) *gbmNode {
	if b.wantContrib {
		for _, s := range idx {
			b.contrib[s] = value
		}
	}
	return &gbmNode{leaf: true, value: value}
}

// lastContrib returns the per-sample leaf values captured while growing the most
// recent full-sample tree (see leafNode); satisfies contribGrower.
func (b *histBuilder) lastContrib() []float64 { return b.contrib }

// buildHist clears buffer `buf` and accumulates the round target over idx into it.
func (b *histBuilder) buildHist(idx []int, buf int) {
	h, nb := b.hist[buf], b.nbins
	for f := 0; f < b.d; f++ {
		base := f * nb
		for j := 0; j <= len(b.edges[f]); j++ {
			h[base+j] = histBin{}
		}
	}
	// Hoist the struct fields and take a per-sample bounds-check-elided slice of
	// the binned row; strength-reduce f*nb to a running base. c is computed to the
	// identical value in the identical (sample, feature) order, so the histogram
	// (and every downstream split) is bit-identical.
	y, binned, d := b.y, b.binned, b.d
	for _, i := range idx {
		yi := y[i]
		br := binned[i*d : i*d+d : i*d+d]
		base := 0
		for f := 0; f < d; f++ {
			c := base + int(br[f])
			h[c].sum += yi
			h[c].cnt++
			base += nb
		}
	}
}

// subHist turns buffer `par` (the parent histogram) into the LARGER child's by
// subtracting the already-built smaller child in buffer `small` — the
// histogram-subtraction trick that builds only one child per node.
func (b *histBuilder) subHist(par, small int) {
	h, sh, nb := b.hist[par], b.hist[small], b.nbins
	for f := 0; f < b.d; f++ {
		base := f * nb
		for j := 0; j <= len(b.edges[f]); j++ {
			h[base+j].sum -= sh[base+j].sum
			h[base+j].cnt -= sh[base+j].cnt
		}
	}
}

// grow sets the round target and grows one weak-learner tree over sample set idx.
func (b *histBuilder) grow(y []float64, idx []int) *gbmTree {
	b.y = y
	b.wantContrib = len(idx) == b.n // full-sample round: every sample lands in a leaf, so contrib is complete
	work := b.idxbuf[:len(idx)]
	copy(work, idx) // partition works in place; never mutate the caller's idx
	b.buildHist(work, 0)
	return &gbmTree{root: b.buildNode(work, 0, 0)}
}

// buildNode grows the subtree over idx whose histogram is ALREADY in buffer buf
// (built by the caller — the root by grow, each child by buildHist/subHist).
func (b *histBuilder) buildNode(idx []int, depth, buf int) *gbmNode {
	m := len(idx)
	h, nb := b.hist[buf], b.nbins
	var total float64 // = Σ residual over idx = sum of feature-0's histogram bins
	for j := 0; j <= len(b.edges[0]); j++ {
		total += h[j].sum
	}
	value := total / float64(m)
	if depth >= b.maxDepth || m < 2*b.minLeaf {
		return b.leafNode(idx, value)
	}
	bestGain := 0.0
	bestFeat, bestBin, bestNL := -1, -1, 0
	for f := 0; f < b.d; f++ {
		base := f * nb
		var leftSum float64
		var nl int
		ne := len(b.edges[f])
		for sb := 0; sb < ne; sb++ { // split after bin sb → left = {x ≤ edges[f][sb]}
			leftSum += h[base+sb].sum
			nl += int(h[base+sb].cnt)
			nr := m - nl
			if nl < b.minLeaf || nr < b.minLeaf {
				continue
			}
			meanL := leftSum / float64(nl)
			meanR := (total - leftSum) / float64(nr)
			diff := meanL - meanR
			gain := float64(nl) * float64(nr) / float64(m) * diff * diff
			if gain > bestGain {
				bestGain, bestFeat, bestBin, bestNL = gain, f, sb, nl
			}
		}
	}
	if bestFeat < 0 {
		return b.leafNode(idx, value)
	}
	// Partition idx in place (unstable): bin ≤ bestBin goes left. nl==bestNL.
	lo := 0
	for k := 0; k < m; k++ {
		s := idx[k]
		if int(b.binned[s*b.d+bestFeat]) <= bestBin {
			idx[lo], idx[k] = idx[k], idx[lo]
			lo++
		}
	}
	left, right := idx[:lo], idx[lo:]
	// Build the SMALLER child's histogram into buf+1, then subtract from this
	// node's histogram (buf) in place to get the LARGER child's — one build per
	// node instead of two. Always recurse the buf+1 child FIRST: its subtree
	// reuses buffers ≥buf+1, so the larger child's histogram in buf survives.
	nl, nr := bestNL, m-bestNL
	node := &gbmNode{feature: bestFeat, threshold: b.edges[bestFeat][bestBin]}
	if nl <= nr { // small = left (buf+1), large = right (buf)
		b.buildHist(left, buf+1)
		b.subHist(buf, buf+1)
		node.left = b.buildNode(left, depth+1, buf+1)
		node.right = b.buildNode(right, depth+1, buf)
	} else { // small = right (buf+1), large = left (buf)
		b.buildHist(right, buf+1)
		b.subHist(buf, buf+1)
		node.right = b.buildNode(right, depth+1, buf+1)
		node.left = b.buildNode(left, depth+1, buf)
	}
	return node
}

// gbmGrower is the weak-learner grower shared by the exact (gbmBuilder) and
// histogram (histBuilder) boosting paths — grow sets the round target and
// returns one tree over the sample set idx.
type gbmGrower interface {
	grow(y []float64, idx []int) *gbmTree
}

// contribGrower is an optional gbmGrower extension: after a full-sample grow, it
// exposes the per-sample leaf value of the tree it just built, so the boosting
// loop can update its scores without re-walking the tree for every row. Only the
// histogram grower implements it (it routes every sample during its bin
// partition); the exact grower falls back to tree.root.predict.
type contribGrower interface {
	lastContrib() []float64
}

// grow adapts the exact builder to gbmGrower.
func (b *gbmBuilder) grow(y []float64, idx []int) *gbmTree { b.y = y; return b.fit(idx) }

// newGBMGrower builds the histogram grower when cfg enables it, else the exact one.
func newGBMGrower(c gbmConfig, x [][]float64, n, d int) gbmGrower {
	if c.histBins > 0 {
		return newHistBuilder(x, n, d, c.maxDepth, c.minSamplesLeaf, c.histBins)
	}
	return newGBMBuilder(x, n, d, c.maxDepth, c.minSamplesLeaf)
}
