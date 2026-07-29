package nn

import (
	"cmp"
	"fmt"
	"math"
	"slices"

	"github.com/jxsl13/goai/tensor"
)

// WandaScore returns the Wanda pruning-importance matrix (Sun, Liu, Bhojanapalli,
// Vishwanathan & Kolter 2023/2024, "A Simple and Effective Pruning Approach for Large
// Language Models", ICLR 2024, arXiv:2306.11695). For a linear layer Y = X·W the
// saliency of weight W[j,o] is
//
//	S[j,o] = |W[j,o]| · ‖X_j‖₂
//
// (Eq. 1/2), where ‖X_j‖₂ is the L2 norm of the j-th INPUT feature's activations over
// all calibration tokens — a single scalar per input channel, shared by every output
// that reads channel j. The insight over magnitude pruning is that a small weight
// multiplying a high-variance input feature can matter more than a large weight on a
// near-dead feature, so the two are combined. W is [C_in, C_out] and X is
// [tokens, C_in] calibration activations; the returned score is [C_in, C_out].
func WandaScore(w, x *tensor.Tensor) (*tensor.Tensor, error) {
	if w.Ndim() != 2 || x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: WandaScore wants rank-2 W[C_in,C_out], X[tokens,C_in], got %v, %v", w.Shape(), x.Shape())
	}
	cin, cout := w.Shape()[0], w.Shape()[1]
	if x.Shape()[1] != cin {
		return nil, fmt.Errorf("nn: WandaScore C_in mismatch: W has %d, X has %d", cin, x.Shape()[1])
	}
	norm := actL2Norm(x) // ‖X_j‖₂ per input channel
	s := tensor.New(w.Dtype(), w.Shape())
	// Typed fast path: score = |W[j,o]|·‖X_j‖ over the full C_in×C_out weight matrix.
	// Walk W and the score storage directly; the |·| and the per-row-invariant norm
	// multiply are unchanged, so bit-identical to the SetF64(|AtF64|·norm) loop.
	if ws := flatF64(w); ws != nil {
		ss := s.Storage().F64()
		for j := 0; j < cin; j++ {
			nj := norm[j]
			base := j * cout
			for o := 0; o < cout; o++ {
				ss[base+o] = math.Abs(ws[base+o]) * nj
			}
		}
		return s, nil
	}
	if ws := flatF32(w); ws != nil {
		ss := s.Storage().F32()
		for j := 0; j < cin; j++ {
			nj := norm[j]
			base := j * cout
			for o := 0; o < cout; o++ {
				ss[base+o] = float32(math.Abs(float64(ws[base+o])) * nj)
			}
		}
		return s, nil
	}
	for j := range cin {
		for o := range cout {
			s.SetF64(math.Abs(w.AtF64(j, o))*norm[j], j, o)
		}
	}
	return s, nil
}

// wandaLess is the TOTAL order the column selection and sort share: score ascending, ties
// broken by input index. Indices are unique, so no two elements compare equal — which is
// what lets an unstable partition reproduce the same selected SET, and what keeps Lomuto
// partitioning from degrading on a column of identical scores.
func wandaLess(a, b int, col []float64) bool {
	if ca, cb := col[a], col[b]; ca != cb {
		return ca < cb
	}
	return a < b
}

// wandaSelectK partitions idx in place so idx[:k] holds the k smallest entries under
// wandaLess, in unspecified order. The consumer only marks membership, so the order inside
// the prefix is never observed — which is why a selection may replace the full sort.
//
// O(n) expected against the sort's O(n log n): about 4k comparisons per column against
// 22.5k at cin=2048. Median-of-three pivoting is not optional here — score columns are
// |w|·‖x‖ products, frequently near-sorted, and a first-element pivot degrades to O(n²) on
// exactly that shape.
func wandaSelectK(idx []int, col []float64, k int) {
	if k <= 0 || k >= len(idx) {
		return
	}
	lo, hi := 0, len(idx)-1
	for lo < hi {
		p := wandaPartition(idx, col, lo, hi)
		switch {
		case p == k-1:
			return
		case p < k-1:
			lo = p + 1
		default:
			hi = p - 1
		}
	}
}

// wandaPartition is Lomuto partitioning around a median-of-three pivot, returning the
// pivot's final position.
func wandaPartition(idx []int, col []float64, lo, hi int) int {
	mid := lo + (hi-lo)/2
	if wandaLess(idx[mid], idx[lo], col) {
		idx[lo], idx[mid] = idx[mid], idx[lo]
	}
	if wandaLess(idx[hi], idx[lo], col) {
		idx[lo], idx[hi] = idx[hi], idx[lo]
	}
	if wandaLess(idx[hi], idx[mid], col) {
		idx[mid], idx[hi] = idx[hi], idx[mid]
	}
	idx[mid], idx[hi] = idx[hi], idx[mid] // pivot to the end
	pv := idx[hi]
	i := lo
	for j := lo; j < hi; j++ {
		if wandaLess(idx[j], pv, col) {
			idx[i], idx[j] = idx[j], idx[i]
			i++
		}
	}
	idx[i], idx[hi] = idx[hi], idx[i]
	return i
}

// wandaPanel is how many output columns are transposed at once. 128 keeps the score panel
// and its drop flags (about 2.3MB at cin=2048) inside L2 while amortizing each row-major
// sweep of the weight matrix over 128 outputs.
const wandaPanel = 128

// WandaPrune returns a pruned copy of W and its 0/1 keep-mask, zeroing the lowest-Wanda-
// importance weights to reach the target unstructured sparsity. Following the paper, the
// COMPARISON GROUP is per-OUTPUT: within each output neuron's incoming connections (each
// column of W) the ⌊sparsity·C_in⌋ smallest-importance weights are removed; the surviving
// weights are left UNCHANGED (no weight update — the key simplification over SparseGPT).
// W is [C_in, C_out], X is [tokens, C_in]; sparsity ∈ [0,1]. A pure-f64 post-training
// compression utility (like the quantizers), not differentiable.
func WandaPrune(w, x *tensor.Tensor, sparsity float64) (pruned, mask *tensor.Tensor, err error) {
	if sparsity < 0 || sparsity > 1 {
		return nil, nil, fmt.Errorf("nn: WandaPrune sparsity=%g out of [0,1]", sparsity)
	}
	s, err := WandaScore(w, x)
	if err != nil {
		return nil, nil, err
	}
	cin, cout := w.Shape()[0], w.Shape()[1]
	// ⌊sparsity·C_in⌋ weights to drop per output column, matching the doc and the paper's
	// int() truncation (Round over-pruned fractional cases, e.g. 0.5·3=1.5 dropped 2 not 1).
	// +1e-9 absorbs float error so an exact integer target (0.7·10=6.999… in f64) floors right.
	k := int(math.Floor(sparsity*float64(cin) + 1e-9))
	pruned = tensor.New(w.Dtype(), w.Shape())
	mask = tensor.New(w.Dtype(), w.Shape())
	idx := make([]int, cin)
	col := make([]float64, cin) // this output's per-input scores, hoisted out of the comparator
	drop := make([]bool, cin)   // reused across columns (cleared each output)
	sortColOf := func(col []float64) {
		// ascending by score, ties broken by input index for determinism. The total-order
		// comparator (score, then index) lets the faster unstable sort reproduce the stable
		// order exactly (indices are unique), and reads the hoisted col instead of dispatching.
		//
		// slices.SortFunc rather than sort.Slice: the latter reaches its swap through
		// reflectlite.Swapper, which ALLOCATES on every call (PS6009). This runs once per
		// output column — 2048 times for a 2048-wide layer. Same comparator, same total
		// order, so the resulting permutation is identical.
		slices.SortFunc(idx, func(x, y int) int {
			if cx, cy := col[x], col[y]; cx != cy {
				if cx < cy {
					return -1
				}
				return 1
			}
			return cmp.Compare(x, y)
		})
	}
	// Typed fast paths: read the score column and write the kept weights / mask through
	// the contiguous storage instead of AtF64/SetF64 per element. Identical values,
	// identical sort order, kept entries only (pruned/mask are zero-initialised).
	sf64 := s.Storage()
	if wsF := flatF64(w); wsF != nil {
		ss, ps, ms := sf64.F64(), pruned.Storage().F64(), mask.Storage().F64()
		// Processed in PANELS of output columns. Per output, the scores, the weights and
		// the mask are all indexed [j*cout+o] — three walks down a column of a row-major
		// [cin,cout] matrix, one cache line touched per input to use eight of its bytes.
		// At 2048x2048 that is 4.2M strided reads for the gather and as many again for the
		// write-back, and it is what the benchmark was spending its time on (PS6011).
		//
		// A panel transposes ob columns at a time, so both the gather and the write-back
		// sweep ss/wsF/ps/ms in ROW order. Bit-identical — same values, same per-column
		// sort, same entries written; only the visiting order changes.
		pn := min(wandaPanel, cout)
		colbuf := make([]float64, pn*cin)
		dropbuf := make([]bool, pn*cin)
		for o0 := 0; o0 < cout; o0 += pn {
			ob := min(pn, cout-o0)
			for j := 0; j < cin; j++ {
				row := ss[j*cout+o0 : j*cout+o0+ob]
				for t, v := range row {
					colbuf[t*cin+j] = v
				}
			}
			for t := 0; t < ob; t++ {
				col := colbuf[t*cin : t*cin+cin : t*cin+cin]
				for j := 0; j < cin; j++ {
					idx[j] = j
				}
				// Only the k SMALLEST are consumed — the full sorted order was never read
				// (PS6001). A selection produces the identical set because the comparator
				// is total and only membership is marked.
				wandaSelectK(idx, col, k)
				d := dropbuf[t*cin : t*cin+cin : t*cin+cin]
				clear(d)
				for r := 0; r < k; r++ {
					d[idx[r]] = true
				}
			}
			for j := 0; j < cin; j++ {
				base := j*cout + o0
				// dropbuf is indexed [t][j], so this reads at stride cin — PS6011 flags
				// it, deliberately. The transposed [j][t] layout was implemented and
				// measured SLOWER (0.982-0.993x): it makes this read contiguous but turns
				// the k selection writes strided and forces a full-buffer clear per panel.
				// The buffer is 256KB and L2-resident, so the stride costs almost nothing
				// — PERF-ACCUM-RESIDENCY-001 seen from the measurement side.
				for t := 0; t < ob; t++ {
					//perfscan:ignore PS6011
					if dropbuf[t*cin+j] {
						continue
					}
					ps[base+t] = wsF[base+t]
					ms[base+t] = 1
				}
			}
		}
		return pruned, mask, nil
	}
	if wsF := flatF32(w); wsF != nil {
		ss, ps, ms := sf64.F32(), pruned.Storage().F32(), mask.Storage().F32()
		// Same panel transpose and selection as the F64 branch above; see the commentary
		// there. The score panel is float64 because the comparator and the scores are,
		// exactly as the per-column buffer was.
		pn := min(wandaPanel, cout)
		colbuf := make([]float64, pn*cin)
		dropbuf := make([]bool, pn*cin)
		for o0 := 0; o0 < cout; o0 += pn {
			ob := min(pn, cout-o0)
			for j := 0; j < cin; j++ {
				row := ss[j*cout+o0 : j*cout+o0+ob]
				for t, v := range row {
					colbuf[t*cin+j] = float64(v)
				}
			}
			for t := 0; t < ob; t++ {
				col := colbuf[t*cin : t*cin+cin : t*cin+cin]
				for j := 0; j < cin; j++ {
					idx[j] = j
				}
				wandaSelectK(idx, col, k)
				d := dropbuf[t*cin : t*cin+cin : t*cin+cin]
				clear(d)
				for r := 0; r < k; r++ {
					d[idx[r]] = true
				}
			}
			for j := 0; j < cin; j++ {
				base := j*cout + o0
				// dropbuf is indexed [t][j], so this reads at stride cin — PS6011 flags
				// it, deliberately. The transposed [j][t] layout was implemented and
				// measured SLOWER (0.982-0.993x): it makes this read contiguous but turns
				// the k selection writes strided and forces a full-buffer clear per panel.
				// The buffer is 256KB and L2-resident, so the stride costs almost nothing
				// — PERF-ACCUM-RESIDENCY-001 seen from the measurement side.
				for t := 0; t < ob; t++ {
					//perfscan:ignore PS6011
					if dropbuf[t*cin+j] {
						continue
					}
					ps[base+t] = wsF[base+t]
					ms[base+t] = 1
				}
			}
		}
		return pruned, mask, nil
	}
	for o := range cout {
		for j := range cin {
			idx[j] = j
			col[j] = s.AtF64(j, o)
		}
		sortColOf(col)
		for j := range drop {
			drop[j] = false
		}
		for r := range k {
			drop[idx[r]] = true
		}
		for j := range cin {
			if drop[j] {
				continue // pruned → 0, mask 0
			}
			pruned.SetF64(w.AtF64(j, o), j, o)
			mask.SetF64(1, j, o)
		}
	}
	return pruned, mask, nil
}

// WandaPruneNM applies N:M structured Wanda pruning: within every group of m consecutive
// input weights feeding one output, the n lowest-importance weights are zeroed (e.g. 2:4
// keeps 2 of every 4). This is the sparsity pattern accelerated by NVIDIA sparse tensor
// cores. C_in must be divisible by m and 0 ≤ n ≤ m. Returns the pruned weights and 0/1
// keep-mask. W is [C_in, C_out], X is [tokens, C_in].
func WandaPruneNM(w, x *tensor.Tensor, n, m int) (pruned, mask *tensor.Tensor, err error) {
	if m <= 0 || n < 0 || n > m {
		return nil, nil, fmt.Errorf("nn: WandaPruneNM needs 0 ≤ n ≤ m with m>0, got n=%d m=%d", n, m)
	}
	s, err := WandaScore(w, x)
	if err != nil {
		return nil, nil, err
	}
	cin, cout := w.Shape()[0], w.Shape()[1]
	if cin%m != 0 {
		return nil, nil, fmt.Errorf("nn: WandaPruneNM C_in=%d not divisible by m=%d", cin, m)
	}
	pruned = tensor.New(w.Dtype(), w.Shape())
	mask = tensor.New(w.Dtype(), w.Shape())
	grp := make([]int, m)
	gsc := make([]float64, m) // block scores, hoisted out of the comparator (was s.AtF64 per compare)
	drop := make([]bool, m)   // reused per block
	// gsc[grp[·]-base] is the score of grp[·]; the SliceStable over identical values in
	// identical (input) order reproduces the original ordering exactly.
	// SortStableFunc, not SortFunc: this comparator is NOT total — it orders on the score
	// alone and relies on stability to keep equal scores in input order, which the comment
	// above depends on. sort.SliceStable would reach its swap through reflectlite.Swapper
	// and allocate on EVERY call (PS6009), and this runs once per block per output —
	// upwards of two million times for a 2048-wide layer at 2:4.
	sortGrp := func(base int) {
		slices.SortStableFunc(grp, func(x, y int) int {
			switch a, b := gsc[x-base], gsc[y-base]; {
			case a < b:
				return -1
			case a > b:
				return 1
			}
			return 0
		})
	}
	sf64 := s.Storage()
	if wsF := flatF64(w); wsF != nil {
		ss, ps, ms := sf64.F64(), pruned.Storage().F64(), mask.Storage().F64()
		for o := 0; o < cout; o++ {
			for base := 0; base < cin; base += m {
				for r := 0; r < m; r++ {
					grp[r] = base + r
					gsc[r] = ss[(base+r)*cout+o]
				}
				sortGrp(base)
				for r := range drop {
					drop[r] = false
				}
				for r := 0; r < n; r++ {
					drop[grp[r]-base] = true
				}
				for r := 0; r < m; r++ {
					if drop[r] {
						continue
					}
					off := (base+r)*cout + o
					ps[off] = wsF[off]
					ms[off] = 1
				}
			}
		}
		return pruned, mask, nil
	}
	if wsF := flatF32(w); wsF != nil {
		ss, ps, ms := sf64.F32(), pruned.Storage().F32(), mask.Storage().F32()
		for o := 0; o < cout; o++ {
			for base := 0; base < cin; base += m {
				for r := 0; r < m; r++ {
					grp[r] = base + r
					gsc[r] = float64(ss[(base+r)*cout+o])
				}
				sortGrp(base)
				for r := range drop {
					drop[r] = false
				}
				for r := 0; r < n; r++ {
					drop[grp[r]-base] = true
				}
				for r := 0; r < m; r++ {
					if drop[r] {
						continue
					}
					off := (base+r)*cout + o
					ps[off] = wsF[off]
					ms[off] = 1
				}
			}
		}
		return pruned, mask, nil
	}
	for o := range cout {
		for base := 0; base < cin; base += m {
			for r := range m {
				grp[r] = base + r
				gsc[r] = s.AtF64(base+r, o)
			}
			sortGrp(base)
			for r := range drop {
				drop[r] = false
			}
			for r := range n {
				drop[grp[r]-base] = true
			}
			for r := range m {
				j := base + r
				if drop[r] {
					continue
				}
				pruned.SetF64(w.AtF64(j, o), j, o)
				mask.SetF64(1, j, o)
			}
		}
	}
	return pruned, mask, nil
}

// actL2Norm returns the per-input-channel (per-column) L2 norm ‖X_j‖₂ =
// √(Σ_t X[t,j]²) over all tokens (rows) of X[tokens, C_in].
func actL2Norm(x *tensor.Tensor) []float64 {
	tokens, cin := x.Shape()[0], x.Shape()[1]
	ss := make([]float64, cin)
	// Typed fast path: walk X's contiguous storage instead of dispatching AtF64 per
	// element. Each ss[j] still accumulates over t ascending with the same v*v → the
	// sum (and its sqrt) is bit-identical; the AtF64 loop stays for exotic dtypes.
	if xs := flatF64(x); xs != nil {
		for t := 0; t < tokens; t++ {
			base := t * cin
			for j := 0; j < cin; j++ {
				v := xs[base+j]
				ss[j] += v * v
			}
		}
	} else if xs := flatF32(x); xs != nil {
		for t := 0; t < tokens; t++ {
			base := t * cin
			for j := 0; j < cin; j++ {
				v := float64(xs[base+j])
				ss[j] += v * v
			}
		}
	} else {
		for t := range tokens {
			for j := range cin {
				v := x.AtF64(t, j)
				ss[j] += v * v
			}
		}
	}
	for j := range cin {
		ss[j] = math.Sqrt(ss[j])
	}
	return ss
}
