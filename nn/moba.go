package nn

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// MoBAAttention computes Mixture of Block Attention (Lu et al. 2025, Moonshot,
// arXiv:2502.13189, §T557): the MoE principle applied to attention. Keys are
// split into blocks of blockSize; per head and per query, each PAST block is
// gated by the affinity ⟨q, mean-pool(K_block)⟩, the top (topK−1) gated blocks
// plus ALWAYS the query's own current block are selected, and ordinary softmax
// attention runs over the selected keys only (causal within the current block).
// With topK ≥ the number of blocks this IS full causal attention (the collapse
// test); with topK = 1 it is block-local attention. q,k,v are [seq, heads·dk];
// host f64 analysis utility in the SSD/RetentionRecurrent mold (the trainable/
// fused form is a §T556-family follow-up). scale 0 → 1/√dk.
func MoBAAttention(q, k, v *tensor.Tensor, heads, blockSize, topK int, scale float64) (*tensor.Tensor, error) {
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("nn: MoBAAttention wants rank-2 q,k,v")
	}
	seq, dm := q.Shape()[0], q.Shape()[1]
	if !k.Shape().Equal(q.Shape()) || !v.Shape().Equal(q.Shape()) {
		return nil, fmt.Errorf("nn: MoBAAttention q/k/v shapes differ")
	}
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("nn: dm %d not divisible by heads %d", dm, heads)
	}
	if blockSize <= 0 || topK <= 0 {
		return nil, fmt.Errorf("nn: MoBAAttention needs blockSize, topK > 0")
	}
	dk := dm / heads
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(dk))
	}
	out := tensor.New(q.Dtype(), q.Shape())
	scores := make([]float64, seq)
	for h := range heads {
		off := h * dk
		for i := range seq {
			cur := i / blockSize
			// gate: affinity of q_i to each past block's mean-pooled key.
			type gated struct {
				block int
				score float64
			}
			var gates []gated
			for b := 0; b < cur; b++ {
				lo, hi := b*blockSize, min((b+1)*blockSize, seq)
				var s float64
				for d := range dk {
					var m float64
					for j := lo; j < hi; j++ {
						m += k.AtF64(j, off+d)
					}
					s += q.AtF64(i, off+d) * m / float64(hi-lo)
				}
				gates = append(gates, gated{b, s})
			}
			sort.Slice(gates, func(a, b int) bool { return gates[a].score > gates[b].score })
			selected := map[int]bool{cur: true} // the current block is always attended
			for gi := 0; gi < len(gates) && len(selected) < topK; gi++ {
				selected[gates[gi].block] = true
			}
			// softmax attention over the selected keys (causal overall).
			m := math.Inf(-1)
			for j := 0; j <= i; j++ {
				if !selected[j/blockSize] {
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
	}
	return out, nil
}
