package nn

import (
	"fmt"
	"math"
	"slices"

	"github.com/jxsl13/goai/tensor"
)

// blockImp ranks one compressed block by its cmp-branch importance.
type blockImp struct {
	block int
	w     float64
}

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
	act := make([]int, 0, seq)  // active (unmasked) key indices, reused by attendMask
	qrow := make([]float64, dk) // q_i[off:off+dk] hoisted per (head,query); all 3 branches re-read it
	// Every buffer below used to be allocated once per (head, query): blockW, the
	// importance slice, the selected-set map, and a capturing closure for each of the two
	// attendMask calls. At seq 512 with 4 heads that was ~14.6k allocations for a single
	// call. They are loop-invariant in SIZE, so they are allocated once and reset per query.
	blockW := make([]float64, nBlocks)
	imps := make([]blockImp, nBlocks)
	selBlk := make([]bool, nBlocks) // set membership as a bitmap, not a map[int]bool
	keepSlc := make([]bool, seq)    // per-key admission, precomputed instead of a callback
	keepWin := make([]bool, seq)
	for h := range heads {
		off := h * dk
		for i := range seq {
			for d := range dk {
				qrow[d] = q.AtF64(i, off+d)
			}
			// ---- cmp: softmax over complete blocks strictly before the query.
			nPast := i / blockSize // complete blocks before i
			if nPast > 0 {
				m := math.Inf(-1)
				for b := range nPast {
					var s float64
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
				for b := range nPast {
					scores[b] = math.Exp(scores[b] - m)
					sum += scores[b]
				}
				for b := range nPast {
					scores[b] /= sum // normalize once, not per channel d
				}
				for d := range dk {
					var o float64
					for b := range nPast {
						o += scores[b] * poolV[b*dm+off+d]
					}
					cmp.SetF64(o, i, off+d)
				}
				for b := range nPast {
					blockW[b] = scores[b] // already normalized
				}
			}
			// ---- slc: top-n blocks by cmp importance, own block always in.
			//
			// The comparator is TOTAL — weight descending, then block index ascending.
			// Ranking on weight alone leaves tied blocks in whatever order the sort
			// happens to produce, which then decides which blocks slc attends; identical
			// pooled rows tie exactly, so this was reachable, not theoretical.
			cand := imps[:nPast]
			for b := range nPast {
				cand[b] = blockImp{b, blockW[b]}
			}
			slices.SortFunc(cand, func(x, y blockImp) int {
				switch {
				case x.w > y.w:
					return -1
				case x.w < y.w:
					return 1
				}
				return x.block - y.block
			})
			clear(selBlk)
			own := i / blockSize
			selBlk[own] = true
			nSel := 1
			for gi := 0; gi < len(cand) && nSel < topN; gi++ {
				if b := cand[gi].block; !selBlk[b] {
					selBlk[b] = true
					nSel++
				}
			}
			// Materialize both admission masks instead of passing closures. This drops two
			// closure allocations per query AND an indirect call per key inside the score
			// loop, which is what actually dominated there — the shared-operand reload the
			// scan flagged is second order behind a call that cannot inline.
			for j := 0; j <= i; j++ {
				keepSlc[j] = selBlk[j/blockSize]
				keepWin[j] = i-j < window
			}
			attendMask(qrow, k, v, slc, i, off, dk, scale, scores, act, keepSlc)
			// ---- win: last `window` keys.
			attendMask(qrow, k, v, win, i, off, dk, scale, scores, act, keepWin)
		}
	}
	return cmp, slc, win, nil
}

// attendMask runs causal softmax attention for query row i over keys admitted by
// keep, writing the head slice [off,off+dk) of out.
// attendMask takes the query row pre-hoisted as qrow = q_i[off:off+dk] (it is re-read for
// every key j, so hoisting it out of the caller kills a dk×(i+1) redundant gather).
func attendMask(qrow []float64, k, v, out *tensor.Tensor, i, off, dk int, scale float64, scores []float64, act []int, keep []bool) {
	// F64 fast path: the score dot reads k[j,off+d] and the P·V reads v[j,off+d] on every
	// (j,d) — an AtF64 dispatch per element over the O(seq·dk) attention compute. Walk the
	// contiguous k/v/out storage directly; qrow was already hoisted. Bit-identical (same
	// values, same ascending-d score dot and ascending-j P·V). AtF64 fallback for other dtypes.
	if ks, vs, os := flatF64(k), flatF64(v), flatF64(out); ks != nil && vs != nil && os != nil {
		dm := k.Shape()[1]
		m := math.Inf(-1)
		// Four keys per pass so one qrow[d] load feeds four dots. Blocked only across runs
		// where all four keys are admitted; a partial run falls back to the scalar form
		// rather than computing scores for masked keys, which must stay -Inf. Each dot
		// still sums over ascending d and m is still updated in ascending j, so the bits
		// and the softmax maximum are unchanged.
		j := 0
		for ; j+4 <= i+1; j += 4 {
			if !(keep[j] && keep[j+1] && keep[j+2] && keep[j+3]) {
				for e := j; e < j+4; e++ {
					if !keep[e] {
						scores[e] = math.Inf(-1)
						continue
					}
					krow := ks[e*dm+off : e*dm+off+dk : e*dm+off+dk]
					var sc float64
					for d := range dk {
						sc += qrow[d] * krow[d]
					}
					sc *= scale
					scores[e] = sc
					if sc > m {
						m = sc
					}
				}
				continue
			}
			k0 := ks[j*dm+off : j*dm+off+dk : j*dm+off+dk]
			k1 := ks[(j+1)*dm+off : (j+1)*dm+off+dk : (j+1)*dm+off+dk]
			k2 := ks[(j+2)*dm+off : (j+2)*dm+off+dk : (j+2)*dm+off+dk]
			k3 := ks[(j+3)*dm+off : (j+3)*dm+off+dk : (j+3)*dm+off+dk]
			var s0, s1, s2, s3 float64
			for d, qd := range qrow {
				s0 += qd * k0[d]
				s1 += qd * k1[d]
				s2 += qd * k2[d]
				s3 += qd * k3[d]
			}
			for e, sc := range [4]float64{s0, s1, s2, s3} {
				sc *= scale
				scores[j+e] = sc
				if sc > m {
					m = sc
				}
			}
		}
		for ; j <= i; j++ {
			if !keep[j] {
				scores[j] = math.Inf(-1)
				continue
			}
			krow := ks[j*dm+off : j*dm+off+dk : j*dm+off+dk]
			var sc float64
			for d := range dk {
				sc += qrow[d] * krow[d]
			}
			sc *= scale
			scores[j] = sc
			if sc > m {
				m = sc
			}
		}
		// The ACTIVE keys are collected here, in the pass that already walks every j.
		// Everything downstream — the normalize and the P·V — then touches only those.
		// Under a sparse mask that is the difference between O(i·d_k) and O(topK·d_k):
		// DSA attends 64 of up to 1024 keys, so the P·V was multiplying by an exact zero
		// fifteen times out of sixteen.
		//
		// Bit-identical for finite v: a masked key has scores[j] == 0 exactly, and
		// o + 0*v is o. (It differs only if a masked key's v were ±Inf or NaN, where the
		// old code would propagate NaN and this does not — an improvement, but stated
		// rather than hidden.)
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
		if sum > 0 {
			for _, j := range act {
				scores[j] /= sum
			}
		}
		// P·V. Walking j for a fixed d strides v by dm — one cache line per key to consume
		// eight of its bytes, repeated d_k times. Four adjacent output channels per pass
		// read v[j, off+d .. off+d+3], four doubles from the SAME line, so the line that
		// was fetched anyway serves four accumulators. Each o still sums over ascending j.
		orow := os[i*dm+off : i*dm+off+dk : i*dm+off+dk]
		if sum <= 0 {
			clear(orow)
			return
		}
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
		return
	}
	m := math.Inf(-1)
	for j := 0; j <= i; j++ {
		if !keep[j] {
			scores[j] = math.Inf(-1)
			continue
		}
		var s float64
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
			for j := 0; j <= i; j++ {
				o += scores[j] * v.AtF64(j, off+d)
			}
		}
		out.SetF64(o, i, off+d)
	}
}
