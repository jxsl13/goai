package nn

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// mobaGate pairs a past block with its gate score. Named rather than declared inside the
// query loop so one buffer can be reused across queries.
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
	// F64 fast path: the block-gate mean-pools each past block's keys PER QUERY though the
	// pool is query-independent — precompute the per-head block key-sums once (O(seq·dk) vs
	// the O(seq²·dk) re-pool) — and walk q/k/v/out contiguously (the query row is hoisted).
	// Bit-identical: same values, same ascending sums, same gate/score/softmax/P·V order.
	if qs, ks, vs, os := flatF64(q), flatF64(k), flatF64(v), flatF64(out); qs != nil && ks != nil && vs != nil && os != nil {
		nBlocks := (seq + blockSize - 1) / blockSize
		// Heads are independent: head h writes only the disjoint out columns [off:off+dk] and reads
		// the shared read-only q/k/v, so the head loop fans out over GOMAXPROCS bit-identically to the
		// serial loop (each worker runs a head's exact serial code). Per-worker scratch: poolSum/blockLen
		// (per-head block key-sums), qrow, scores. Gated on heads·seq²·dk.
		parallelRows(heads, seq*seq*dk, func(hlo, hhi int) {
			poolSum := make([]float64, nBlocks*dk)
			blockLen := make([]int, nBlocks)
			qrow := make([]float64, dk)
			scores := make([]float64, seq)
			// THE SELECTED-BLOCK SET WAS A map[int]bool BUILT PER QUERY and probed once per key in
			// the innermost loop — 5.4% of this benchmark sat in runtime.mapaccess1_fast64 alone,
			// on top of a map allocation per query. The keys are block indices, dense in
			// [0,nBlocks), so a slice is the natural container; stamping it with a generation
			// counter makes the per-query reset free instead of an nBlocks clear.
			selStamp := make([]int32, nBlocks)
			var gen int32
			// The gate slice was rebuilt per query too. One reused buffer holds the same values in
			// the same order — it is refilled from scratch on every query.
			gates := make([]mobaGate, 0, nBlocks)
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
					sort.Slice(gates, func(a, b int) bool { return gates[a].score > gates[b].score })
					gen++
					selStamp[cur] = gen
					nSel := 1
					// The block indices in gates are distinct — b runs once over [0,cur) — so counting
					// adds is exactly the map's len, which is what bounded this loop.
					for gi := 0; gi < len(gates) && nSel < topK; gi++ {
						selStamp[gates[gi].block] = gen
						nSel++
					}
					m := math.Inf(-1)
					// Score dot is a latency-bound serial reduction over d; jam two
					// selected output rows (prev,j) so their accumulators are
					// independent, hiding the FP-add latency. Each scores[x] still
					// sums d ascending → bit-identical; max is order-free.
					prev := -1
					for j := 0; j <= i; j++ {
						if selStamp[j/blockSize] != gen {
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
						// EIGHT KEYS PER PASS. orow does not vary with j, so the loop below loaded and
						// stored the whole output row once per attended key for a single multiply-add each;
						// eight keys at a time hold orow[d] in a register across eight of them. Widths 6 and
						// 8 measure the same (7.45 vs 7.44 ms) and 2 leaves half the win on the table (8.35).
						//
						// BIT-IDENTICAL: each orow[d] still sums j ascending, and the accumulator is an
						// EXPLICIT LOCAL — a compound assignment would add the SUM of the eight products,
						// which associates differently (T1183). Every width from 2 to 8 leaves the digests
						// unchanged, which is what says the association is right.
						//
						// SKIPPING THE UNSELECTED KEYS WAS TRIED AND IS WORSE. Most j are outside the top-K
						// blocks, so their score is -Inf, their softmax weight is exactly +0, and the d loop
						// adds nothing — but a per-key branch instead of this jam measured 8.46 ms against
						// 7.44. It would not be equivalent either: 0 times an infinite or NaN value is NaN,
						// not zero, so a v with a non-finite entry would get a different answer, and the
						// digests cannot see that because ordinary data has no infinities.
						jj := 0
						for ; jj+7 <= i; jj += 8 {
							p0 := scores[jj+0] / sum
							p1 := scores[jj+1] / sum
							p2 := scores[jj+2] / sum
							p3 := scores[jj+3] / sum
							p4 := scores[jj+4] / sum
							p5 := scores[jj+5] / sum
							p6 := scores[jj+6] / sum
							p7 := scores[jj+7] / sum
							v0 := vs[(jj+0)*dm+off : (jj+0)*dm+off+dk]
							v1 := vs[(jj+1)*dm+off : (jj+1)*dm+off+dk]
							v2 := vs[(jj+2)*dm+off : (jj+2)*dm+off+dk]
							v3 := vs[(jj+3)*dm+off : (jj+3)*dm+off+dk]
							v4 := vs[(jj+4)*dm+off : (jj+4)*dm+off+dk]
							v5 := vs[(jj+5)*dm+off : (jj+5)*dm+off+dk]
							v6 := vs[(jj+6)*dm+off : (jj+6)*dm+off+dk]
							v7 := vs[(jj+7)*dm+off : (jj+7)*dm+off+dk]
							for d := range dk {
								a := orow[d]
								a += p0 * v0[d]
								a += p1 * v1[d]
								a += p2 * v2[d]
								a += p3 * v3[d]
								a += p4 * v4[d]
								a += p5 * v5[d]
								a += p6 * v6[d]
								a += p7 * v7[d]
								orow[d] = a
							}
						}
						for ; jj <= i; jj++ {
							pj := scores[jj] / sum
							vrow := vs[jj*dm+off : jj*dm+off+dk : jj*dm+off+dk]
							for d := range dk {
								orow[d] += pj * vrow[d]
							}
						}
					}
				}
			}
		})
		return out, nil
	}
	// Same per-head independence as the fast path above; each worker owns its own scores.
	parallelRows(heads, seq*seq*dk, func(hlo, hhi int) {
		scores := make([]float64, seq)
		for h := hlo; h < hhi; h++ {
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
	})
	return out, nil
}
