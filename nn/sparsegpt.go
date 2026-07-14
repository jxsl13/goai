package nn

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// SparseGPT one-shot layer-wise pruning (Frantar & Alistarh 2023, "SparseGPT: Massive
// Language Models Can Be Accurately Pruned in One-Shot", ICML 2023, arXiv:2301.00774).
// It is the pruning sibling of GPTQ (§GPTQuantize) and the accuracy-tier counterpart of
// Wanda (§WandaPrune): where Wanda simply zeros the lowest |W|·‖X‖ weights with NO weight
// update, SparseGPT solves the layer reconstruction ‖WX − ŴX‖² with the same Optimal
// Brain Surgeon (OBS) inverse-Hessian machinery as GPTQ. Each pruned weight injects an
// error that is COMPENSATED into the not-yet-processed weights, so the surviving weights
// absorb the removed ones — far lower reconstruction error than magnitude pruning at the
// same sparsity.
//
// The reconstruction Hessian is H = 2·X·Xᵀ (X the calibration activations [in,samples]),
// diagonally dampened, and Hinv is the upper Cholesky factor of H⁻¹ (exactly as GPTQ).
// The paper writes H = XXᵀ; the overall Hessian scale cancels in both the saliency ratio
// and the compensation, so the GPTQ-consistent 2·XXᵀ is equivalent. The pruning saliency
// of weight W[r,c] is ε = W[r,c]²/[Hinv]_cc² (Alg. 1 — the SQUARED inverse-Hessian
// Cholesky diagonal, the reconstruction error its removal would cause), and the
// ⌊sparsity⌋ smallest-ε weights are pruned PER ROW (per output neuron), selected
// adaptively in blocks of blockSize columns so the mask sees the running OBS
// compensation. (This follows the paper's Algorithm 1 per-row mask, which yields exact
// per-row sparsity and matches the released code's N:M path; that code's UNSTRUCTURED
// path instead thresholds each block globally.) On pruning, W[r,c]→0 and the error
// W[r,c]/[Hinv]_cc is propagated to columns j>c via W[:,j] -= e·Hinv[c,j]; kept weights
// are unchanged (error 0) beyond the compensation they receive.
//
// w is [out,in], x is [in,samples], sparsity ∈ [0,1]. Returns the pruned+compensated
// weights and the 0/1 keep-mask. A pure-f64 post-training utility (like GPTQ/Wanda), not
// differentiable.
func SparseGPTPrune(w, x *tensor.Tensor, sparsity float64, opts ...SparseGPTOption) (pruned, mask *tensor.Tensor, err error) {
	if sparsity < 0 || sparsity > 1 {
		return nil, nil, fmt.Errorf("nn: SparseGPTPrune sparsity=%g out of [0,1]", sparsity)
	}
	cfg := sparseGPTCfg{damp: 0.01, blockSize: 128}
	for _, o := range opts {
		o(&cfg)
	}
	// per block of blk columns, prune ⌊sparsity·blk⌋ smallest-ε weights per row.
	return sparseGPTCore(w, x, cfg.damp, cfg.blockSize, func(blk int) int {
		return int(math.Round(sparsity * float64(blk)))
	})
}

// SparseGPTPruneNM applies N:M structured SparseGPT pruning: within every group of m
// consecutive inputs feeding an output, the n lowest-saliency weights are zeroed (e.g.
// 2:4 keeps 2 of every 4) with the same OBS inverse-Hessian compensation as SparseGPTPrune.
// This is the semi-structured sparsity pattern accelerated by NVIDIA sparse tensor cores.
// The paper sets the adaptive-mask block size to m for N:M, so WithSparseGPTBlock is
// ignored here (WithSparseGPTDamp still applies). in must be divisible by m and 0 ≤ n ≤ m.
// w is [out,in], x is [in,samples]; returns the pruned+compensated weights and 0/1 mask.
func SparseGPTPruneNM(w, x *tensor.Tensor, n, m int, opts ...SparseGPTOption) (pruned, mask *tensor.Tensor, err error) {
	if m <= 0 || n < 0 || n > m {
		return nil, nil, fmt.Errorf("nn: SparseGPTPruneNM needs 0 ≤ n ≤ m with m>0, got n=%d m=%d", n, m)
	}
	if w.Ndim() == 2 && w.Shape()[1]%m != 0 {
		return nil, nil, fmt.Errorf("nn: SparseGPTPruneNM in=%d not divisible by m=%d", w.Shape()[1], m)
	}
	cfg := sparseGPTCfg{damp: 0.01, blockSize: 128}
	for _, o := range opts {
		o(&cfg)
	}
	// block size = m; prune exactly n per m-group per row.
	return sparseGPTCore(w, x, cfg.damp, m, func(int) int { return n })
}

// sparseGPTCore runs the shared SparseGPT OBS pruning: it builds the dampened inverse
// Hessian and greedily prunes each column block, calling kOf(blockLen) for how many
// weights to remove per row in that block. SparseGPTPrune (unstructured ratio) and
// SparseGPTPruneNM (N:M) differ only in kOf and the block size.
func sparseGPTCore(w, x *tensor.Tensor, damp float64, blockSize int, kOf func(blk int) int) (pruned, mask *tensor.Tensor, err error) {
	if w.Ndim() != 2 || x.Ndim() != 2 {
		return nil, nil, fmt.Errorf("nn: SparseGPT needs rank-2 w and x, got %v and %v", w.Shape(), x.Shape())
	}
	out, in := w.Shape()[0], w.Shape()[1]
	if x.Shape()[0] != in {
		return nil, nil, fmt.Errorf("nn: SparseGPT x must be [%d,samples], got %v", in, x.Shape())
	}
	samples := x.Shape()[1]
	bs := blockSize
	if bs <= 0 || bs > in {
		bs = in
	}

	// H = 2·X·Xᵀ (layer reconstruction Hessian), in×in — mirrors GPTQuantize.
	h := make([][]float64, in)
	for i := range in {
		h[i] = make([]float64, in)
		for j := range in {
			var s float64
			for k := range samples {
				s += x.AtF64(i, k) * x.AtF64(j, k)
			}
			h[i][j] = 2 * s
		}
	}
	wm := matAt(w) // working weight [out][in], mutated by OBS compensation

	// dead columns (no calibration signal) → prune to 0; then dampen the diagonal.
	var meanDiag float64
	for i := range in {
		meanDiag += h[i][i]
	}
	meanDiag /= float64(in)
	dead := make([]bool, in)
	for i := range in {
		if h[i][i] == 0 {
			h[i][i] = 1
			dead[i] = true
			for r := range out {
				wm[r][i] = 0
			}
		}
	}
	ridge := damp * meanDiag
	for i := range in {
		h[i][i] += ridge
	}

	// Hinv = upper Cholesky of H⁻¹ (hinv[i][j]=Lᵀ for j≥i), as in GPTQuantize.
	flat := make([]float64, in*in)
	for i := range in {
		copy(flat[i*in:(i+1)*in], h[i])
	}
	hinvT, err := linalg.Inverse(tensor.FromFloat64(tensor.Shape{in, in}, flat))
	if err != nil {
		return nil, nil, fmt.Errorf("nn: SparseGPTPrune inverse Hessian: %w", err)
	}
	loF, err := linalg.Cholesky(hinvT)
	if err != nil {
		return nil, nil, fmt.Errorf("nn: SparseGPTPrune Cholesky: %w", err)
	}
	hinv := make([][]float64, in)
	for i := range in {
		hinv[i] = make([]float64, in)
		for j := i; j < in; j++ {
			hinv[i][j] = loF.AtF64(j, i) // upper = Lᵀ
		}
	}

	prunedMask := make([][]bool, out)
	for r := range out {
		prunedMask[r] = make([]bool, in)
	}
	// block-adaptive pruning with OBS error propagation to all later columns (the
	// immediate full propagation is numerically identical to the paper's block+global
	// update; the block only bounds when each row's mask is chosen).
	for bStart := 0; bStart < in; bStart += bs {
		bEnd := min(bStart+bs, in)
		blk := bEnd - bStart
		k := kOf(blk) // weights to prune per row in this block (unstructured ⌊s·blk⌋ or N:M n)
		// per-row mask selection over the block using current (compensated) weights.
		if k > 0 {
			idx := make([]int, blk)
			for r := range out {
				for t := range blk {
					idx[t] = bStart + t
				}
				sort.SliceStable(idx, func(a, b int) bool {
					return sgSaliency(wm[r][idx[a]], hinv[idx[a]][idx[a]]) < sgSaliency(wm[r][idx[b]], hinv[idx[b]][idx[b]])
				})
				for t := range k {
					prunedMask[r][idx[t]] = true
				}
			}
		}
		// process each column in the block: prune→0 with OBS compensation, keep→unchanged.
		for c := bStart; c < bEnd; c++ {
			d := hinv[c][c]
			for r := range out {
				if dead[c] {
					prunedMask[r][c] = true
				}
				if !prunedMask[r][c] {
					continue // kept: q = wm[r][c], error 0, no propagation
				}
				e := wm[r][c] / d // pruned: q=0 ⇒ error = W[r,c]/[Hinv]_cc
				wm[r][c] = 0
				for j := c + 1; j < in; j++ {
					wm[r][j] -= e * hinv[c][j]
				}
			}
		}
	}

	pruned = tensor.New(w.Dtype(), tensor.Shape{out, in})
	mask = tensor.New(w.Dtype(), tensor.Shape{out, in})
	for r := range out {
		for c := range in {
			if prunedMask[r][c] {
				continue // pruned → 0, mask 0
			}
			pruned.SetF64(wm[r][c], r, c)
			mask.SetF64(1, r, c)
		}
	}
	return pruned, mask, nil
}

// sgSaliency is the SparseGPT pruning metric ε = w²/[Hinv]_cc² — the layer-reconstruction
// error that removing weight w (in a column with inverse-Hessian Cholesky diagonal d)
// would cause. Smaller ε ⇒ safer to prune.
func sgSaliency(w, d float64) float64 {
	wd := w / d
	return wd * wd
}

// SparseGPTOption configures SparseGPTPrune (functional-options idiom, §C12).
type SparseGPTOption func(*sparseGPTCfg)

type sparseGPTCfg struct {
	damp      float64
	blockSize int
}

// WithSparseGPTDamp sets the Hessian dampening fraction — a ridge of damp·mean(diag H) added
// to the Hessian diagonal before inversion (the same stabilizer GPTQ uses).
//
// In plain terms: SparseGPT decides which weights to prune and how to correct the survivors
// using an inverse curvature matrix; a small amount is added to that matrix's diagonal so the
// inverse stays stable. Boundary behavior — too small risks an ill-conditioned inverse (noisy
// pruning); too large over-regularizes and weakens the error compensation. Default 0.01
// (research-grounded: the SparseGPT reference dampening, §R193, mirroring GPTQ's 1%).
func WithSparseGPTDamp(d float64) SparseGPTOption {
	return func(c *sparseGPTCfg) { c.damp = d }
}

// WithSparseGPTBlock sets the adaptive-masking block size Bs — the number of columns over
// which each row's prune mask is chosen together using the running OBS error compensation.
//
// In plain terms: SparseGPT decides what to prune in chunks of Bs columns at a time, updating
// its error-compensation as it goes; the chunk size trades accuracy against speed. Boundary
// behavior — small Bs re-optimizes the mask more finely (better, slower); a block ≥ the number
// of input columns selects a single mask over the whole row (fastest, coarsest). Default 128
// (research-grounded: the SparseGPT reference block size, §R193).
func WithSparseGPTBlock(b int) SparseGPTOption {
	return func(c *sparseGPTCfg) { c.blockSize = b }
}
