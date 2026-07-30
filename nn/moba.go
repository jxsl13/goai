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
	// F64 fast path: the block-gate mean-pools each past block's keys PER QUERY though the
	// pool is query-independent — precompute the per-head block key-sums once (O(seq·dk) vs
	// the O(seq²·dk) re-pool) — and walk q/k/v/out contiguously (the query row is hoisted).
	// Bit-identical: same values, same ascending sums, same gate/score/softmax/P·V order.
	if qs, ks, vs, os := flatF64(q), flatF64(k), flatF64(v), flatF64(out); qs != nil && ks != nil && vs != nil && os != nil {
		nBlocks := (seq + blockSize - 1) / blockSize
		poolSum := make([]float64, nBlocks*dk) // per-block Σ_j k[j,off+d] (rebuilt per head)
		blockLen := make([]int, nBlocks)
		qrow := make([]float64, dk)
		for h := range heads {
			off := h * dk
			for b := range nBlocks {
				lo, hi := b*blockSize, min((b+1)*blockSize, seq)
				blockLen[b] = hi - lo
				pb := poolSum[b*dk : b*dk+dk : b*dk+dk]
				for d := range dk {
					pb[d] = 0
				}
				for j := lo; j < hi; j++ {
					krow := ks[j*dm+off : j*dm+off+dk]
					for d := range dk {
						pb[d] += krow[d]
					}
				}
			}
			for i := range seq {
				cur := i / blockSize
				for d := range dk {
					qrow[d] = qs[i*dm+off+d]
				}
				type gated struct {
					block int
					score float64
				}
				var gates []gated
				for b := 0; b < cur; b++ {
					pb := poolSum[b*dk : b*dk+dk : b*dk+dk]
					bl := float64(blockLen[b])
					var sgate float64
					for d := range dk {
						sgate += qrow[d] * pb[d] / bl
					}
					gates = append(gates, gated{b, sgate})
				}
				sort.Slice(gates, func(a, b int) bool { return gates[a].score > gates[b].score })
				selected := map[int]bool{cur: true}
				for gi := 0; gi < len(gates) && len(selected) < topK; gi++ {
					selected[gates[gi].block] = true
				}
				m := math.Inf(-1)
				// Score dot is a latency-bound serial reduction over d; jam two
				// selected output rows (prev,j) so their accumulators are
				// independent, hiding the FP-add latency. Each scores[x] still
				// sums d ascending → bit-identical; max is order-free.
				prev := -1
				for j := 0; j <= i; j++ {
					if !selected[j/blockSize] {
						scores[j] = math.Inf(-1)
						continue
					}
					if prev < 0 {
						prev = j
						continue
					}
					krow0 := ks[prev*dm+off : prev*dm+off+dk]
					krow1 := ks[j*dm+off : j*dm+off+dk]
					var sc0, sc1 float64
					for d := range dk {
						sc0 += qrow[d] * krow0[d]
						sc1 += qrow[d] * krow1[d]
					}
					sc0 *= scale
					sc1 *= scale
					scores[prev] = sc0
					scores[j] = sc1
					if sc0 > m {
						m = sc0
					}
					if sc1 > m {
						m = sc1
					}
					prev = -1
				}
				if prev >= 0 {
					krow := ks[prev*dm+off : prev*dm+off+dk]
					var sc float64
					for d := range dk {
						sc += qrow[d] * krow[d]
					}
					sc *= scale
					scores[prev] = sc
					if sc > m {
						m = sc
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
				// P·V j-OUTER / d-inner: read each v-row contiguously (vs[j*dm+off:+dk])
				// ONCE and axpy it into the dk-wide output row, instead of striding vs by
				// dm and re-streaming the column region per output channel d. Also hoists
				// scores[j]/sum out of the d loop (it was recomputed dk×). Bit-identical:
				// scores[j]/sum is deterministic and each fixed d sums j ascending (PS1006).
				orow := os[i*dm+off : i*dm+off+dk : i*dm+off+dk]
				for d := range dk {
					orow[d] = 0
				}
				if sum > 0 {
					for j := 0; j <= i; j++ {
						pj := scores[j] / sum
						vrow := vs[j*dm+off : j*dm+off+dk : j*dm+off+dk]
						for d := range dk {
							orow[d] += pj * vrow[d]
						}
					}
				}
			}
		}
		return out, nil
	}
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
