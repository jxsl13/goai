package nn

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// NSABranches computes the three attention branches of Native Sparse Attention
// (Yuan et al. 2025, DeepSeek, arXiv:2502.11089, §T556): per head and query,
//
//	cmp — attention over COMPRESSED block tokens (here mean-pooled keys/values
//	      per block, the zero-parameter instantiation of the paper's learned φ;
//	      only COMPLETE blocks strictly before the query are visible),
//	slc — full attention over the raw tokens of the top-n blocks SELECTED by the
//	      cmp branch's attention weights (the query's own block is always
//	      selected — NSA's locality guarantee),
//	win — plain causal attention over the last `window` keys.
//
// NSA's output is g_cmp·cmp + g_slc·slc + g_win·win with learned sigmoid gates;
// the branches are returned separately so callers supply their own gates — the
// paper's novel mechanics (compression-attention-driven selection) live here.
// Host f64 analysis utility (SSD/MoBA mold). scale 0 → 1/√dk.
func NSABranches(q, k, v *tensor.Tensor, heads, blockSize, topN, window int, scale float64) (cmp, slc, win *tensor.Tensor, err error) {
	if q.Ndim() != 2 || !k.Shape().Equal(q.Shape()) || !v.Shape().Equal(q.Shape()) {
		return nil, nil, nil, fmt.Errorf("nn: NSABranches wants equal rank-2 q,k,v")
	}
	seq, dm := q.Shape()[0], q.Shape()[1]
	if heads <= 0 || dm%heads != 0 || blockSize <= 0 || topN <= 0 || window <= 0 {
		return nil, nil, nil, fmt.Errorf("nn: NSABranches bad geometry (heads %d, dm %d, block %d, topN %d, window %d)", heads, dm, blockSize, topN, window)
	}
	dk := dm / heads
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(dk))
	}
	cmp = tensor.New(q.Dtype(), q.Shape())
	slc = tensor.New(q.Dtype(), q.Shape())
	win = tensor.New(q.Dtype(), q.Shape())
	nBlocks := (seq + blockSize - 1) / blockSize

	// block mean-pools per head dim (compressed keys/values).
	poolK := make([]float64, nBlocks*dm)
	poolV := make([]float64, nBlocks*dm)
	for b := range nBlocks {
		lo, hi := b*blockSize, min((b+1)*blockSize, seq)
		for d := range dm {
			var sk, sv float64
			for j := lo; j < hi; j++ {
				sk += k.AtF64(j, d)
				sv += v.AtF64(j, d)
			}
			poolK[b*dm+d] = sk / float64(hi-lo)
			poolV[b*dm+d] = sv / float64(hi-lo)
		}
	}

	scores := make([]float64, max(seq, nBlocks))
	for h := range heads {
		off := h * dk
		for i := range seq {
			// ---- cmp: softmax over complete blocks strictly before the query.
			nPast := i / blockSize // complete blocks before i
			blockW := make([]float64, nPast)
			if nPast > 0 {
				m := math.Inf(-1)
				for b := range nPast {
					var s float64
					for d := range dk {
						s += q.AtF64(i, off+d) * poolK[b*dm+off+d]
					}
					s *= scale
					scores[b] = s
					if s > m {
						m = s
					}
				}
				var sum float64
				for b := range nPast {
					scores[b] = math.Exp(scores[b] - m)
					sum += scores[b]
				}
				for d := range dk {
					var o float64
					for b := range nPast {
						o += scores[b] / sum * poolV[b*dm+off+d]
					}
					cmp.SetF64(o, i, off+d)
				}
				for b := range nPast {
					blockW[b] = scores[b] / sum // importance for selection
				}
			}
			// ---- slc: top-n blocks by cmp importance, own block always in.
			type imp struct {
				block int
				w     float64
			}
			var imps []imp
			for b := range nPast {
				imps = append(imps, imp{b, blockW[b]})
			}
			sort.Slice(imps, func(a, b int) bool { return imps[a].w > imps[b].w })
			selected := map[int]bool{i / blockSize: true}
			for gi := 0; gi < len(imps) && len(selected) < topN; gi++ {
				selected[imps[gi].block] = true
			}
			attendMask(q, k, v, slc, i, off, dk, scale, scores, func(j int) bool { return selected[j/blockSize] })
			// ---- win: last `window` keys.
			attendMask(q, k, v, win, i, off, dk, scale, scores, func(j int) bool { return i-j < window })
		}
	}
	return cmp, slc, win, nil
}

// attendMask runs causal softmax attention for query row i over keys admitted by
// keep, writing the head slice [off,off+dk) of out.
func attendMask(q, k, v, out *tensor.Tensor, i, off, dk int, scale float64, scores []float64, keep func(j int) bool) {
	m := math.Inf(-1)
	for j := 0; j <= i; j++ {
		if !keep(j) {
			scores[j] = math.Inf(-1)
			continue
		}
		var s float64
		for d := range dk {
			s += q.AtF64(i, off+d) * k.AtF64(j, off+d)
		}
		s *= scale
		scores[j] = s
		if s > m {
			m = s
		}
	}
	var sum float64
	for j := 0; j <= i; j++ {
		if math.IsInf(scores[j], -1) {
			scores[j] = 0
			continue
		}
		scores[j] = math.Exp(scores[j] - m)
		sum += scores[j]
	}
	for d := range dk {
		var o float64
		if sum > 0 {
			for j := 0; j <= i; j++ {
				o += scores[j] / sum * v.AtF64(j, off+d)
			}
		}
		out.SetF64(o, i, off+d)
	}
}
