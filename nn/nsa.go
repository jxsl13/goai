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
	//perfscan:ignore PS3034,PS3063 block mean-pool setup, one-time/call, tiny vs O(heads·seq²) attn | host f64 analysis utility, not wired to any
	for b := range nBlocks {
		lo, hi := b*blockSize, min((b+1)*blockSize, seq)
		for d := range dm {
			var sk, sv float64
			//perfscan:ignore PS1001 pool-setup AtF64 O(seq·dm), negligible vs attention; utility
			for j := lo; j < hi; j++ {
				sk += k.AtF64(j, d)
				sv += v.AtF64(j, d)
			}
			poolK[b*dm+d] = sk / float64(hi-lo)
			poolV[b*dm+d] = sv / float64(hi-lo)
		}
	}

	// Heads are independent: head h writes only the disjoint output columns [off:off+dk] of
	// cmp/slc/win and reads its own head slice of q/k/v plus the shared read-only poolK/poolV
	// (pooled once above), with no cross-head reduction. Parallelizing over heads runs each
	// head's exact serial computation on a worker, so the result is bit-identical regardless of
	// the internal softmax/sort. Each worker owns its own scores/qrow scratch (reused across the
	// head's queries). Gated on heads·seq²·dk so a tiny problem stays serial.
	parallelRows(heads, seq*seq*dk, func(hlo, hhi int) {
		//perfscan:ignore PS6008 resource-class on hoisted qrow gather, no wallclock
		scores := make([]float64, max(seq, nBlocks))
		//perfscan:ignore PS6008 resource-only alloc class, no wallclock; utility
		qrow := make([]float64, dk) // q_i[off:off+dk] hoisted per (head,query); all 3 branches re-read it
		for h := hlo; h < hhi; h++ {
			off := h * dk
			for i := range seq {
				//perfscan:ignore PS1001 cmp-branch dot over flat pool slice, small nPast; utility
				for d := range dk {
					qrow[d] = q.AtF64(i, off+d)
				}
				// ---- cmp: softmax over complete blocks strictly before the query.
				nPast := i / blockSize // complete blocks before i
				//perfscan:ignore PS3035 cmp score dot, flat pool slice already; utility
				blockW := make([]float64, nPast)
				if nPast > 0 {
					m := math.Inf(-1)
					//perfscan:ignore PS6010 resource-class, no wallclock
					for b := range nPast {
						var s float64
						//perfscan:ignore PS3010 cmp softmax scalar region; utility, tiny nPast
						for d := range dk {
							s += qrow[d] * poolK[b*dm+off+d]
						}
						s *= scale
						scores[b] = s
						if s > m {
							m = s
						}
					}
					var sum float64
					//perfscan:ignore PS3010,PS3066 cmp output write O(seq·dk), small; utility | same cmp write region; utility
					for b := range nPast {
						scores[b] = math.Exp(scores[b] - m)
						sum += scores[b]
					}
					for b := range nPast {
						scores[b] /= sum // normalize once, not per channel d
					}
					//perfscan:ignore PS1001,PS1006,PS3053 cmp output over flat slice; utility, negligible | cmp reduce over small nBlocks; utility, marginal | same cmp
					for d := range dk {
						var o float64
						//perfscan:ignore PS3010 blockW copy, tiny; utility
						for b := range nPast {
							//perfscan:ignore PS6011 slice-of-slices false positive; utility
							o += scores[b] * poolV[b*dm+off+d]
						}
						cmp.SetF64(o, i, off+d)
					}
					for b := range nPast {
						blockW[b] = scores[b] // already normalized
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
				//perfscan:ignore PS3002,PS6009 imps sort is block-count (short) not vocab; utility path | resource-class, no wallclock
				sort.Slice(imps, func(a, b int) bool { return imps[a].w > imps[b].w })
				//perfscan:ignore PS3083 attendMask call site; utility
				selected := map[int]bool{i / blockSize: true}
				for gi := 0; gi < len(imps) && len(selected) < topN; gi++ {
					selected[imps[gi].block] = true
				}
				attendMask(qrow, k, v, slc, i, off, dk, scale, scores, func(j int) bool { return selected[j/blockSize] })
				// ---- win: last `window` keys.
				attendMask(qrow, k, v, win, i, off, dk, scale, scores, func(j int) bool { return i-j < window })
			}
		}
	})
	return cmp, slc, win, nil
}

// attendMask runs causal softmax attention for query row i over keys admitted by
// keep, writing the head slice [off,off+dk) of out.
// attendMask takes the query row pre-hoisted as qrow = q_i[off:off+dk] (it is re-read for
// every key j, so hoisting it out of the caller kills a dk×(i+1) redundant gather).
func attendMask(qrow []float64, k, v, out *tensor.Tensor, i, off, dk int, scale float64, scores []float64, keep func(j int) bool) {
	// F64 fast path: the score dot reads k[j,off+d] and the P·V reads v[j,off+d] on every
	// (j,d) — an AtF64 dispatch per element over the O(seq·dk) attention compute. Walk the
	// contiguous k/v/out storage directly; qrow was already hoisted. Bit-identical (same
	// values, same ascending-d score dot and ascending-j P·V). AtF64 fallback for other dtypes.
	if ks, vs, os := flatF64(k), flatF64(v), flatF64(out); ks != nil && vs != nil && os != nil {
		dm := k.Shape()[1]
		m := math.Inf(-1)
		// FOUR KEYS PER PASS OVER THE QUERY ROW. Each score is a dk-term dot into ONE
		// accumulator, so the loop runs at add latency rather than throughput, and qrow is
		// re-streamed once per key. Four keys read it once and interleave four chains.
		// Bit-identical: every score sums over the same ascending d into its own accumulator,
		// and the scale, the store and the running maximum stay in ascending j.
		//
		// A group straddling the mask takes the original step one key at a time, exactly as the
		// masked attention kernel does — abandoning the fast path for the rest of the row costs
		// more than the branch on any mask whose live region is not a prefix.
		jj := 0
		//perfscan:ignore PS3076 inside attendMask flatF64 fast path, already typed walk
		for ; jj+3 <= i; jj += 4 {
			if !keep(jj) || !keep(jj+1) || !keep(jj+2) || !keep(jj+3) {
				for o := range 4 {
					j := jj + o
					if !keep(j) {
						scores[j] = math.Inf(-1)
						continue
					}
					krow := ks[j*dm+off : j*dm+off+dk : j*dm+off+dk]
					var sc float64
					//perfscan:ignore PS3010 attendMask flatF64 fast path already optimal
					for d := range dk {
						sc += qrow[d] * krow[d]
					}
					sc *= scale
					scores[j] = sc
					if sc > m {
						m = sc
					}
				}
				continue
			}
			b0 := jj*dm + off
			k0 := ks[b0 : b0+dk : b0+dk]
			k1 := ks[b0+dm : b0+dm+dk : b0+dm+dk]
			k2 := ks[b0+2*dm : b0+2*dm+dk : b0+2*dm+dk]
			k3 := ks[b0+3*dm : b0+3*dm+dk : b0+3*dm+dk]
			var s0, s1, s2, s3 float64
			//perfscan:ignore PS3010 attendMask AtF64 declined-dtype fallback, correct to keep
			for d := range dk {
				q := qrow[d]
				s0 += q * k0[d]
				s1 += q * k1[d]
				s2 += q * k2[d]
				s3 += q * k3[d]
			}
			for o, sv := range [4]float64{s0, s1, s2, s3} {
				sc := sv * scale
				scores[jj+o] = sc
				if sc > m {
					m = sc
				}
			}
		}
		for j := jj; j <= i; j++ {
			if !keep(j) {
				scores[j] = math.Inf(-1)
				continue
			}
			krow := ks[j*dm+off : j*dm+off+dk : j*dm+off+dk]
			var sc float64
			//perfscan:ignore PS3010 attendMask AtF64 fallback branch, correct to keep
			for d := range dk {
				sc += qrow[d] * krow[d]
			}
			sc *= scale
			scores[j] = sc
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
		if sum > 0 {
			for j := 0; j <= i; j++ {
				scores[j] /= sum
			}
		}
		// P·V j-OUTER / d-inner: read each selected v-row contiguously (vs[j*dm+off:+dk])
		// ONCE and axpy it into a dk-wide output-row accumulator, instead of striding vs by
		// dm and re-streaming the whole column region once per output channel d. For each
		// fixed d the sum still runs j ascending, so it is bit-identical (PS1006).
		orow := os[i*dm+off : i*dm+off+dk : i*dm+off+dk]
		for d := range dk {
			orow[d] = 0
		}
		if sum > 0 {
			// FOUR KEYS PER PASS AGAIN: orow does not vary with j, so the key-at-a-time form
			// makes a full load-store round trip through it for one addition each. Holding
			// orow[d] in a local across four additions stores once. Bit-identical — the same
			// four additions in the same ascending j, in a register instead of memory.
			jv := 0
			for ; jv+3 <= i; jv += 4 {
				p0, p1 := scores[jv], scores[jv+1]
				p2, p3 := scores[jv+2], scores[jv+3]
				b0 := jv*dm + off
				v0 := vs[b0 : b0+dk : b0+dk]
				v1 := vs[b0+dm : b0+dm+dk : b0+dm+dk]
				v2 := vs[b0+2*dm : b0+2*dm+dk : b0+2*dm+dk]
				v3 := vs[b0+3*dm : b0+3*dm+dk : b0+3*dm+dk]
				for d := range dk {
					t := orow[d]
					t += p0 * v0[d]
					t += p1 * v1[d]
					t += p2 * v2[d]
					t += p3 * v3[d]
					orow[d] = t
				}
			}
			//perfscan:ignore PS1007 stale line beyond current file; already flatF64-optimized
			for j := jv; j <= i; j++ {
				pj := scores[j]
				vrow := vs[j*dm+off : j*dm+off+dk : j*dm+off+dk]
				for d := range dk {
					orow[d] += pj * vrow[d]
				}
			}
		}
		return
	}
	m := math.Inf(-1)
	for j := 0; j <= i; j++ {
		if !keep(j) {
			scores[j] = math.Inf(-1)
			continue
		}
		var s float64
		//perfscan:ignore PS3010 stale line beyond current 218-line file; already optimized
		for d := range dk {
			s += qrow[d] * k.AtF64(j, off+d)
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
	// Normalize by sum ONCE, not once per value channel d.
	if sum > 0 {
		for j := 0; j <= i; j++ {
			scores[j] /= sum
		}
	}
	for d := range dk {
		var o float64
		if sum > 0 {
			//perfscan:ignore PS3010 stale line beyond current file; utility, already fast-pathed
			for j := 0; j <= i; j++ {
				o += scores[j] * v.AtF64(j, off+d)
			}
		}
		out.SetF64(o, i, off+d)
	}
}
