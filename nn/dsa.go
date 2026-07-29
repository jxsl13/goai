package nn

import (
	"fmt"
	"math"
	"slices"

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
	qrow := make([]float64, dk) // q_i[off:off+dk] hoisted per (query,head) for attendMask
	// The rank slice, the selected-set map and one closure PER HEAD were all allocated
	// inside the query loop. Sizes are loop-invariant, so allocate once and reset.
	keepBuf := make([]bool, seq)
	type ranked struct {
		j int
		s float64
	}
	rankBuf := make([]ranked, seq)
	// F64 fast path: the lightning-indexer ranking scores qIdx·kIdx per (i,j) with a double
	// AtF64 dispatch over O(seq²·idxHeads·idxDim) — the main attention (attendMask) is already
	// typed, so this ranking dispatch is now the dominant cost. Walk the index rows contiguously
	// and pass the query row directly. Bit-identical: same ascending-d dot, ReLU, weighting,
	// ranking and attention.
	if qis, kis, qs := flatF64(qIdx), flatF64(kIdx), flatF64(q); qis != nil && kis != nil && qs != nil {
		idxWidth := qIdx.Shape()[1]
		// Hoisted scratch: a reused rank slice (no per-query realloc growth) and a []bool
		// membership set (no per-query map alloc + hashing). Both are cleared between
		// queries, so the selected set, sort order and attention math are IDENTICAL.
		rank := make([]ranked, 0, seq)
		sel := make([]bool, seq)
		var setIdx []int
		for i := range seq {
			qir := qis[i*idxWidth : i*idxWidth+idxWidth : i*idxWidth+idxWidth]
			rank = rank[:0]
			for j := 0; j <= i; j++ {
				kjr := kis[j*idxWidth : j*idxWidth+idxWidth : j*idxWidth+idxWidth]
				var s float64
				for h := range idxHeads {
					base := h * idxDim
					var dot float64
					for d := range idxDim {
						dot += qir[base+d] * kjr[base+d]
					}
					if dot > 0 {
						s += w[h] * dot
					}
				}
				rank = append(rank, ranked{j, s})
			}
			// Total comparator: score descending, then index ascending. Ranking on score
			// alone leaves tied candidates in an unspecified order that then decides which
			// keys are attended. ReLU'd dots tie at exactly zero routinely, so this was
			// reachable in ordinary use, not a corner case.
			slices.SortFunc(rank, func(x, y ranked) int {
				switch {
				case x.s > y.s:
					return -1
				case x.s < y.s:
					return 1
				}
				return x.j - y.j
			})
			setIdx = setIdx[:0]
			sel[i] = true
			setIdx = append(setIdx, i)
			cnt := 1
			for ri := 0; ri < len(rank) && cnt < topK; ri++ {
				if j := rank[ri].j; !sel[j] {
					sel[j] = true
					setIdx = append(setIdx, j)
					cnt++
				}
			}
			for h := range heads {
				off := h * dk
				qr := qs[i*dm+off : i*dm+off+dk]
				attendMask(qr, k, v, out, i, off, dk, scale, scores, sel)
			}
			for _, j := range setIdx {
				sel[j] = false
			}
		}
		return out, nil
	}
	for i := range seq {
		rank := rankBuf[:0]
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
		// Total comparator: score descending, then index ascending. Ranking on score
		// alone leaves tied candidates in an unspecified order that then decides which
		// keys are attended. ReLU'd dots tie at exactly zero routinely, so this was
		// reachable in ordinary use, not a corner case.
		slices.SortFunc(rank, func(x, y ranked) int {
			switch {
			case x.s > y.s:
				return -1
			case x.s < y.s:
				return 1
			}
			return x.j - y.j
		})
		clear(keepBuf)
		keepBuf[i] = true // the query itself is always attended
		nSel := 1
		for ri := 0; ri < len(rank) && nSel < topK; ri++ {
			if j := rank[ri].j; !keepBuf[j] {
				keepBuf[j] = true
				nSel++
			}
		}
		for h := range heads {
			off := h * dk
			for d := range dk {
				qrow[d] = q.AtF64(i, off+d)
			}
			attendMask(qrow, k, v, out, i, off, dk, scale, scores, keepBuf)
		}
	}
	return out, nil
}
