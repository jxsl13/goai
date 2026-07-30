package nn

import (
	"fmt"
	"math"

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
	// selectDrop partitions idx so idx[:k] holds the k LOWEST-score inputs (the pruned set),
	// under a strict total order (score asc, then input index asc). The drop loop below marks
	// exactly that set order-independently, so a quickselect (avg O(cin)) is bit-identical to
	// the previous full O(cin log cin) sort — while running once per output column (PS3006).
	less := func(x, y int) bool {
		if col[x] != col[y] {
			return col[x] < col[y]
		}
		return x < y
	}
	selectDrop := func(k int) { selectPartition(idx, k, less) }
	// Typed fast paths: read the score column and write the kept weights / mask through
	// the contiguous storage instead of AtF64/SetF64 per element. Identical values,
	// identical sort order, kept entries only (pruned/mask are zero-initialised).
	sf64 := s.Storage()
	if wsF := flatF64(w); wsF != nil {
		ss, ps, ms := sf64.F64(), pruned.Storage().F64(), mask.Storage().F64()
		// Each output column is independent (its own scores, quickselect and disjoint strided
		// output j*cout+o), so fan the column loop out over GOMAXPROCS with PER-WORKER scratch
		// (idx/col/drop + a col-capturing comparator). Bit-identical to the serial loop.
		parallelRows(cout, cin, func(olo, ohi int) {
			idx := make([]int, cin)
			col := make([]float64, cin)
			drop := make([]bool, cin)
			less := func(x, y int) bool {
				if col[x] != col[y] {
					return col[x] < col[y]
				}
				return x < y
			}
			for o := olo; o < ohi; o++ {
				for j := 0; j < cin; j++ {
					idx[j] = j
					col[j] = ss[j*cout+o]
				}
				selectPartition(idx, k, less)
				for j := range drop {
					drop[j] = false
				}
				for r := 0; r < k; r++ {
					drop[idx[r]] = true
				}
				for j := 0; j < cin; j++ {
					if drop[j] {
						continue
					}
					off := j*cout + o
					ps[off] = wsF[off]
					ms[off] = 1
				}
			}
		})
		return pruned, mask, nil
	}
	if wsF := flatF32(w); wsF != nil {
		ss, ps, ms := sf64.F32(), pruned.Storage().F32(), mask.Storage().F32()
		parallelRows(cout, cin, func(olo, ohi int) {
			idx := make([]int, cin)
			col := make([]float64, cin)
			drop := make([]bool, cin)
			less := func(x, y int) bool {
				if col[x] != col[y] {
					return col[x] < col[y]
				}
				return x < y
			}
			for o := olo; o < ohi; o++ {
				for j := 0; j < cin; j++ {
					idx[j] = j
					col[j] = float64(ss[j*cout+o])
				}
				selectPartition(idx, k, less)
				for j := range drop {
					drop[j] = false
				}
				for r := 0; r < k; r++ {
					drop[idx[r]] = true
				}
				for j := 0; j < cin; j++ {
					if drop[j] {
						continue
					}
					off := j*cout + o
					ps[off] = wsF[off]
					ms[off] = 1
				}
			}
		})
		return pruned, mask, nil
	}
	for o := range cout {
		for j := range cin {
			idx[j] = j
			col[j] = s.AtF64(j, o)
		}
		selectDrop(k)
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

// selectPartition partitions a in place so that a[:k] holds the k elements that rank BEFORE
// the rest under less (a ranks before b ⟺ less(a,b)), in unspecified order — a Hoare
// quickselect, average O(len(a)). less MUST be a strict total order (no two elements compare
// equal), which makes the retained set unique and independent of pivot choice. Small ranges
// are finished by insertion sort. No-op when k ≤ 0 or k ≥ len(a).
func selectPartition(a []int, k int, less func(x, y int) bool) {
	if k <= 0 || k >= len(a) {
		return
	}
	lo, hi := 0, len(a)-1
	for lo < hi {
		if hi-lo < 12 { // insertion sort the small range; the split point is then settled
			for i := lo + 1; i <= hi; i++ {
				for j := i; j > lo && less(a[j], a[j-1]); j-- {
					a[j], a[j-1] = a[j-1], a[j]
				}
			}
			return
		}
		mid := lo + (hi-lo)/2
		// median-of-three (lo, mid, hi) so a[lo] ≤ a[mid] ≤ a[hi] under less; pivot = a[mid].
		// A median pivot bounds the inner scans and avoids O(n²) on sorted / organ-pipe input.
		if less(a[mid], a[lo]) {
			a[lo], a[mid] = a[mid], a[lo]
		}
		if less(a[hi], a[lo]) {
			a[lo], a[hi] = a[hi], a[lo]
		}
		if less(a[hi], a[mid]) {
			a[mid], a[hi] = a[hi], a[mid]
		}
		pivot := a[mid]
		i, j := lo, hi
		for {
			for less(a[i], pivot) {
				i++
			}
			for less(pivot, a[j]) {
				j--
			}
			if i >= j {
				break
			}
			a[i], a[j] = a[j], a[i]
			i++
			j--
		}
		if k <= j { // [lo,j] rank ≤ pivot, [j+1,hi] rank ≥ pivot
			hi = j
		} else {
			lo = j + 1
		}
	}
}

// stableSortGrpByScore sorts grp ascending by gsc[grp[r]-base], STABLE (equal scores keep their
// input order) — an alloc-free manual insertion sort replacing sort.SliceStable, which allocated
// an interface closure PER CALL (~2.1M allocs on a 2048² N:M prune, GC-capping the throughput).
// m = len(grp) is the N:M block size (tiny, typically 4), so insertion sort is both faster and
// zero-alloc. Bit-identical ordering to SliceStable with less = gsc[a]<gsc[b]: the strict `>`
// stop condition never reorders equal scores, so ties keep their ascending input order.
func stableSortGrpByScore(grp []int, gsc []float64, base int) {
	for i := 1; i < len(grp); i++ {
		gi := grp[i]
		ki := gsc[gi-base]
		j := i - 1
		for j >= 0 && gsc[grp[j]-base] > ki {
			grp[j+1] = grp[j]
			j--
		}
		grp[j+1] = gi
	}
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
	sortGrp := func(base int) {
		stableSortGrpByScore(grp, gsc, base)
	}
	sf64 := s.Storage()
	if wsF := flatF64(w); wsF != nil {
		ss, ps, ms := sf64.F64(), pruned.Storage().F64(), mask.Storage().F64()
		// Each output column is independent (its own N:M block selection, disjoint strided
		// output), so fan the column loop over GOMAXPROCS with PER-WORKER grp/gsc/drop scratch
		// and comparator. Bit-identical to the serial loop.
		parallelRows(cout, cin, func(olo, ohi int) {
			grp := make([]int, m)
			gsc := make([]float64, m)
			drop := make([]bool, m)
			sortGrp := func(base int) {
				stableSortGrpByScore(grp, gsc, base)
			}
			for o := olo; o < ohi; o++ {
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
		})
		return pruned, mask, nil
	}
	if wsF := flatF32(w); wsF != nil {
		ss, ps, ms := sf64.F32(), pruned.Storage().F32(), mask.Storage().F32()
		parallelRows(cout, cin, func(olo, ohi int) {
			grp := make([]int, m)
			gsc := make([]float64, m)
			drop := make([]bool, m)
			sortGrp := func(base int) {
				stableSortGrpByScore(grp, gsc, base)
			}
			for o := olo; o < ohi; o++ {
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
		})
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
