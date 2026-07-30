package nn

import (
	"fmt"
	"math"
	"slices"

	"github.com/jxsl13/goai/tensor"
)

// mobaGate ranks one block by its gate score.
type mobaGate struct {
	block int
	score float64
}

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
	// gates, the selected-block set and the sort were all allocated per (head, query):
	// a growing slice, sort.Slice's reflectlite.Swapper, and a map. At 8 heads x 512
	// queries that was ~20k allocations for one call. All three are hoisted and reused.
	// F64 fast path: the block-gate mean-pools each past block's keys PER QUERY though the
	// pool is query-independent — precompute the per-head block key-sums once (O(seq·dk) vs
	// the O(seq²·dk) re-pool) — and walk q/k/v/out contiguously (the query row is hoisted).
	// Bit-identical: same values, same ascending sums, same gate/score/softmax/P·V order.
	if qs, ks, vs, os := flatF64(q), flatF64(k), flatF64(v), flatF64(out); qs != nil && ks != nil && vs != nil && os != nil {
		nBlocks := (seq + blockSize - 1) / blockSize
		// PARALLEL over heads, with every per-query buffer owned by the WORKER. Head h writes
		// only the disjoint output columns [off:off+dk] and reads its own slice of q/k/v, so the
		// partition runs each head's exact serial computation — bit-identical. The buffers below
		// were hoisted out of the query loop to remove ~20k allocations per call at 8 heads x 512
		// queries; at function scope they would be shared across workers and race, so they move
		// in here and are paid once per worker instead of once per query.
		parallelRows(heads, seq*seq*dk, func(hlo, hhi int) {
			scores := make([]float64, seq)
			act := make([]int, 0, seq)
			gates := make([]mobaGate, 0, seq)
			selBlk := make([]bool, seq)
			poolSum := make([]float64, nBlocks*dk)
			blockLen := make([]int, nBlocks)
			qrow := make([]float64, dk)
			for h := hlo; h < hhi; h++ {
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
					gates = gates[:0]
					for b := 0; b < cur; b++ {
						pb := poolSum[b*dk : b*dk+dk : b*dk+dk]
						bl := float64(blockLen[b])
						var sgate float64
						for d := range dk {
							sgate += qrow[d] * pb[d] / bl
						}
						gates = append(gates, mobaGate{b, sgate})
					}
					// TOTAL order: gate score descending, then block index ascending. Ranking
					// on score alone leaves tied blocks in whatever order the sort produces,
					// and that order decides WHICH blocks are attended. slices.SortFunc also
					// avoids sort.Slice's per-call Swapper allocation (PS6009).
					slices.SortFunc(gates, func(x, y mobaGate) int {
						switch {
						case x.score > y.score:
							return -1
						case x.score < y.score:
							return 1
						}
						return x.block - y.block
					})
					clear(selBlk)
					selBlk[cur] = true
					nSel := 1
					for gi := 0; gi < len(gates) && nSel < topK; gi++ {
						if b := gates[gi].block; !selBlk[b] {
							selBlk[b] = true
							nSel++
						}
					}
					m := math.Inf(-1)
					// Score dot is a latency-bound serial reduction over d; jam two
					// selected output rows (prev,j) so their accumulators are
					// independent, hiding the FP-add latency. Each scores[x] still
					// sums d ascending → bit-identical; max is order-free.
					prev := -1
					for j := 0; j <= i; j++ {
						if !selBlk[j/blockSize] {
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
					// Collect the ACTIVE keys in the pass that already walks every j; the
					// normalize and the P·V below then touch only those. MoBA attends topK
					// BLOCKS of up to i keys, so most of this range is masked and was being
					// multiplied by an exact zero once per output channel.
					//
					// Bit-identical for finite v: a masked key has scores[j] == 0 exactly and
					// o + 0*v is o. (It differs only if a masked key's v were ±Inf or NaN,
					// where the old code propagated NaN through a position the mask excludes.)
					act = act[:0]
					var sum float64
					for j := 0; j <= i; j++ {
						if math.IsInf(scores[j], -1) {
							scores[j] = 0
							continue
						}
						scores[j] = math.Exp(scores[j] - m)
						sum += scores[j]
						act = append(act, j)
					}
					orow := os[i*dm+off : i*dm+off+dk]
					if sum <= 0 {
						clear(orow)
						continue
					}
					// Normalize ONCE. Dividing inside the P·V loop cost d_k divides per key
					// instead of one; `scores[j] / sum * v` associates left, so folding the
					// divide out is the same arithmetic, not a reassociation.
					for _, j := range act {
						scores[j] /= sum
					}
					// Four output channels per pass. Walking j for a fixed d strides v by dm,
					// touching one cache line per key to consume eight of its bytes and
					// repeating that d_k times; four adjacent channels read v[j, off+d..d+3]
					// from the SAME line (PS6011). Each accumulator still sums over ascending j.
					//
					// Composing this with the contiguous-v-row axpy from the other merge side was
					// MEASURED and LOST, so that form was not merged in. All three candidates are
					// bit-identical, so the choice is purely about traffic (nn, -count=4, 3
					// interleaved rounds, M2 Pro):
					//
					//                         act+regblock   act+axpy        all-j+axpy
					//   DSAAttention_seq1024      48.89ms   53.14ms +8.7%   79.01ms +61.6%
					//   MoBAAttention             46.07ms   57.20ms +24.2%  68.40ms +48.5%
					//
					// Iterating only the selected keys is the dominant win; the axpy is a
					// REGRESSION on top of it. The register accumulators are what keep the OUTPUT
					// row out of memory — an axpy into orow reloads and restores all d_k elements
					// once per active key, and under a sparse mask that costs more than the v-row
					// locality it buys.
					d := 0
					for ; d+4 <= dk; d += 4 {
						var o0, o1, o2, o3 float64
						for _, j := range act {
							sj := scores[j]
							vq := vs[j*dm+off+d : j*dm+off+d+4 : j*dm+off+d+4]
							o0 += sj * vq[0]
							o1 += sj * vq[1]
							o2 += sj * vq[2]
							o3 += sj * vq[3]
						}
						orow[d], orow[d+1], orow[d+2], orow[d+3] = o0, o1, o2, o3
					}
					for ; d < dk; d++ {
						var o float64
						for _, j := range act {
							o += scores[j] * vs[j*dm+off+d]
						}
						orow[d] = o
					}
				}
			}
		})
		return out, nil
	}
	// PARALLEL over heads, with every per-query buffer owned by the WORKER. Head h writes
	// only the disjoint output columns [off:off+dk] and reads its own slice of q/k/v, so the
	// partition runs each head's exact serial computation — bit-identical. The buffers below
	// were hoisted out of the query loop to remove ~20k allocations per call at 8 heads x 512
	// queries; at function scope they would be shared across workers and race, so they move
	// in here and are paid once per worker instead of once per query.
	parallelRows(heads, seq*seq*dk, func(hlo, hhi int) {
		scores := make([]float64, seq)
		for h := hlo; h < hhi; h++ {
			off := h * dk
			for i := range seq {
				cur := i / blockSize
				// gate: affinity of q_i to each past block's mean-pooled key.
				var gates []mobaGate
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
					gates = append(gates, mobaGate{b, s})
				}
				// Same total order as the typed path above.
				slices.SortFunc(gates, func(x, y mobaGate) int {
					switch {
					case x.score > y.score:
						return -1
					case x.score < y.score:
						return 1
					}
					return x.block - y.block
				})
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
	})
	return out, nil
}
