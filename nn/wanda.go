package nn

import (
	"fmt"
	"math"
	"sort"

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
	for o := range cout {
		for j := range cin {
			idx[j] = j
			col[j] = s.AtF64(j, o) // O(cin) dispatch once, vs O(cin·log·cin) inside the comparator
		}
		// ascending by score, ties broken by input index for determinism. The total-order
		// comparator (score, then index) lets the faster unstable sort reproduce the stable
		// order exactly (indices are unique), and reads the hoisted col instead of dispatching.
		sort.Slice(idx, func(a, b int) bool {
			if ca, cb := col[idx[a]], col[idx[b]]; ca != cb {
				return ca < cb
			}
			return idx[a] < idx[b]
		})
		drop := make([]bool, cin)
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
	for o := range cout {
		for base := 0; base < cin; base += m {
			for r := range m {
				grp[r] = base + r
			}
			// ascending by score within the block; drop the n smallest.
			sort.SliceStable(grp, func(a, b int) bool { return s.AtF64(grp[a], o) < s.AtF64(grp[b], o) })
			drop := make([]bool, m)
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
	for t := range tokens {
		for j := range cin {
			v := x.AtF64(t, j)
			ss[j] += v * v
		}
	}
	for j := range cin {
		ss[j] = math.Sqrt(ss[j])
	}
	return ss
}
