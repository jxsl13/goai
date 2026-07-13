package nn

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// DSAAttention computes DeepSeek Sparse Attention (DeepSeek-V3.2, §T558): a
// LIGHTNING INDEXER — a small, separate scoring head set — ranks every past
// token per query,
//
//	I(i,j) = Σ_h w_h · ReLU( ⟨qIdx_i,h , kIdx_j,h⟩ )
//
// the top-k tokens by I (the query itself always included) are selected, and
// ordinary causal softmax attention over q,k,v runs on the selected tokens
// only — O(n·k) attention after the O(n²) (but tiny-dim, FP8-cheap in the
// paper) indexer pass. qIdx/kIdx are [seq, idxHeads·idxDim], w is [idxHeads]
// (the paper trains the indexer to mimic dense attention; here it is an input
// — the MECHANICS are what this utility pins). topK ≥ seq degenerates to full
// causal attention. Host f64 analysis utility (MoBA/NSA mold). scale 0 → 1/√dk.
func DSAAttention(q, k, v, qIdx, kIdx *tensor.Tensor, w []float64, heads, topK int, scale float64) (*tensor.Tensor, error) {
	if q.Ndim() != 2 || !k.Shape().Equal(q.Shape()) || !v.Shape().Equal(q.Shape()) {
		return nil, fmt.Errorf("nn: DSAAttention wants equal rank-2 q,k,v")
	}
	seq, dm := q.Shape()[0], q.Shape()[1]
	if qIdx.Ndim() != 2 || !kIdx.Shape().Equal(qIdx.Shape()) || qIdx.Shape()[0] != seq {
		return nil, fmt.Errorf("nn: DSAAttention indexer shapes %v/%v want [seq=%d, idxHeads·idxDim]", qIdx.Shape(), kIdx.Shape(), seq)
	}
	idxHeads := len(w)
	if idxHeads == 0 || qIdx.Shape()[1]%idxHeads != 0 {
		return nil, fmt.Errorf("nn: DSAAttention indexer width %d not divisible by %d weights", qIdx.Shape()[1], idxHeads)
	}
	if heads <= 0 || dm%heads != 0 || topK <= 0 {
		return nil, fmt.Errorf("nn: DSAAttention bad geometry")
	}
	idxDim := qIdx.Shape()[1] / idxHeads
	dk := dm / heads
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(dk))
	}
	out := tensor.New(q.Dtype(), q.Shape())
	scores := make([]float64, seq)
	for i := range seq {
		// lightning indexer: rank all past tokens once per query (shared by heads).
		type ranked struct {
			j int
			s float64
		}
		var rank []ranked
		for j := 0; j <= i; j++ {
			var s float64
			for h := range idxHeads {
				var dot float64
				for d := range idxDim {
					dot += qIdx.AtF64(i, h*idxDim+d) * kIdx.AtF64(j, h*idxDim+d)
				}
				if dot > 0 { // ReLU
					s += w[h] * dot
				}
			}
			rank = append(rank, ranked{j, s})
		}
		sort.Slice(rank, func(a, b int) bool { return rank[a].s > rank[b].s })
		selected := map[int]bool{i: true} // the query itself is always attended
		for ri := 0; ri < len(rank) && len(selected) < topK; ri++ {
			selected[rank[ri].j] = true
		}
		for h := range heads {
			off := h * dk
			attendMask(q, k, v, out, i, off, dk, scale, scores, func(j int) bool { return selected[j] })
		}
	}
	return out, nil
}
