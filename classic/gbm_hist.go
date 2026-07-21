package classic

import "sort"

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
type histBuilder struct {
	x                 [][]float64
	y                 []float64 // current round's target (residual); set per round via grow
	n, d              int
	maxDepth, minLeaf int
	nbins             int
	edges             [][]float64 // [d][≤nbins-1] ascending split thresholds
	binned            []uint16    // [n*d] row-major bin index (bin = #edges < x)
	hsum              []float64   // [d*nbins] scratch gradient histogram
	hcnt              []int32     // [d*nbins] scratch count histogram
	idxbuf            []int       // reusable full-sample index scratch
}

// newHistBuilder bins every feature once and allocates the reusable scratch.
func newHistBuilder(x [][]float64, n, d, maxDepth, minLeaf, nbins int) *histBuilder {
	if nbins < 2 {
		nbins = 2
	}
	b := &histBuilder{x: x, n: n, d: d, maxDepth: maxDepth, minLeaf: minLeaf, nbins: nbins}
	b.edges = make([][]float64, d)
	b.binned = make([]uint16, n*d)
	col := make([]float64, n)
	for f := 0; f < d; f++ {
		for i := 0; i < n; i++ {
			col[i] = x[i][f]
		}
		sort.Float64s(col)
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
	b.hsum = make([]float64, d*nbins)
	b.hcnt = make([]int32, d*nbins)
	b.idxbuf = make([]int, n)
	return b
}

// grow sets the round target and grows one weak-learner tree over sample set idx.
func (b *histBuilder) grow(y []float64, idx []int) *gbmTree {
	b.y = y
	work := b.idxbuf[:len(idx)]
	copy(work, idx) // partition works in place; never mutate the caller's idx
	return &gbmTree{root: b.buildNode(work, 0)}
}

func (b *histBuilder) buildNode(idx []int, depth int) *gbmNode {
	m := len(idx)
	var total float64
	for _, i := range idx {
		total += b.y[i]
	}
	value := total / float64(m)
	if depth >= b.maxDepth || m < 2*b.minLeaf {
		return &gbmNode{leaf: true, value: value}
	}
	nb := b.nbins
	for f := 0; f < b.d; f++ {
		base := f * nb
		for j := 0; j <= len(b.edges[f]); j++ { // bins 0..len(edges)
			b.hsum[base+j] = 0
			b.hcnt[base+j] = 0
		}
	}
	for _, i := range idx {
		yi := b.y[i]
		row := i * b.d
		for f := 0; f < b.d; f++ {
			c := f*nb + int(b.binned[row+f])
			b.hsum[c] += yi
			b.hcnt[c]++
		}
	}
	bestGain := 0.0
	bestFeat, bestBin := -1, -1
	for f := 0; f < b.d; f++ {
		base := f * nb
		var leftSum float64
		var nl int
		ne := len(b.edges[f])
		for sb := 0; sb < ne; sb++ { // split after bin sb → left = {x ≤ edges[f][sb]}
			leftSum += b.hsum[base+sb]
			nl += int(b.hcnt[base+sb])
			nr := m - nl
			if nl < b.minLeaf || nr < b.minLeaf {
				continue
			}
			meanL := leftSum / float64(nl)
			meanR := (total - leftSum) / float64(nr)
			diff := meanL - meanR
			gain := float64(nl) * float64(nr) / float64(m) * diff * diff
			if gain > bestGain {
				bestGain, bestFeat, bestBin = gain, f, sb
			}
		}
	}
	if bestFeat < 0 {
		return &gbmNode{leaf: true, value: value}
	}
	// Partition idx in place (unstable — node order is irrelevant to the mean /
	// histogram of the children): bin ≤ bestBin goes left.
	lo := 0
	for k := 0; k < m; k++ {
		s := idx[k]
		if int(b.binned[s*b.d+bestFeat]) <= bestBin {
			idx[lo], idx[k] = idx[k], idx[lo]
			lo++
		}
	}
	return &gbmNode{
		feature:   bestFeat,
		threshold: b.edges[bestFeat][bestBin],
		left:      b.buildNode(idx[:lo], depth+1),
		right:     b.buildNode(idx[lo:], depth+1),
	}
}

// gbmGrower is the weak-learner grower shared by the exact (gbmBuilder) and
// histogram (histBuilder) boosting paths — grow sets the round target and
// returns one tree over the sample set idx.
type gbmGrower interface {
	grow(y []float64, idx []int) *gbmTree
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
