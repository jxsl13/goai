package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/fmath"
	"github.com/jxsl13/goai/tensor"
)

// gspoKernel is the fused GSPO policy loss (Zheng et al. 2025, "Group Sequence
// Policy Optimization", arXiv:2507.18071 — the RL objective used for Qwen3, §T549).
// GRPO's token-level importance ratios are replaced by ONE length-normalized
// SEQUENCE ratio per response, applied uniformly to its tokens:
//
//	s_i   = exp( (1/|y_i|) · Σ_t (logπθ_t − logπ_old_t) )     // per sequence i
//	surr_i = min(s_i·Â_i, clip(s_i, 1−ε, 1+ε)·Â_i)
//	loss   = −(1/G) · Σ_i surr_i                               // minimize −J^GSPO
//
// Inputs: logπθ[batch], logπ_old[batch] flat concatenations of the G sequences'
// per-token log-probs (segmentation via GSPOAttrs.Lengths), advantage[G] per
// sequence (GroupAdvantage). The paper drops GRPO's KL-to-reference term and clips
// FAR tighter (ε default 3e-4): the length-normalized ratios concentrate near 1,
// which is exactly what stabilizes long-sequence RL. With every length == 1 the
// loss equals GRPO with β=0 EXACTLY (the collapse test). Accumulation in f64.
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func gspoKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: gspo wants (logpNew, logpOld, advantage), got %d inputs", len(in))
	}
	lpNew, lpOld, adv := in[0], in[1], in[2]
	pa, _ := attrs.(backend.GSPOAttrs)
	pa = pa.WithDefaults()
	if len(pa.Lengths) == 0 {
		return nil, fmt.Errorf("ref: gspo needs GSPOAttrs.Lengths")
	}
	g := len(pa.Lengths)
	total := 0
	for i, l := range pa.Lengths {
		if l <= 0 {
			return nil, fmt.Errorf("ref: gspo length[%d] = %d, want > 0", i, l)
		}
		total += l
	}
	if lpNew.Ndim() != 1 || lpNew.Shape()[0] != total || !lpOld.Shape().Equal(lpNew.Shape()) {
		return nil, fmt.Errorf("ref: gspo logp shapes %v/%v, want [%d] (Σ lengths)", lpNew.Shape(), lpOld.Shape(), total)
	}
	if adv.Ndim() != 1 || adv.Shape()[0] != g {
		return nil, fmt.Errorf("ref: gspo advantage %v, want [%d] (one per sequence)", adv.Shape(), g)
	}
	eps := pa.Epsilon

	// The per-token logp difference is read via AtF64 on every token — two interface
	// dispatches per token over Σlengths. Walk the contiguous logp slices directly on the
	// typed fast path (lpNew/lpOld are rank-1). Bit-identical: same ascending-t sum, same
	// exp/clip. AtF64 fallback for exotic dtypes.
	var sum float64
	off := 0
	// MIN AND MAX THROUGH internal/fmath, NOT math. On arm64 math.Min compiles to a CALL into
	// math.archMin inside a 48-byte frame; the min builtin compiles to a single FMIND. They are
	// not interchangeable — math.Min documents -Inf as beating NaN and the builtin propagates
	// NaN — and the difference is REACHABLE here, not theoretical: a log-probability of -Inf
	// makes the ratio exactly +0, and +0 times an infinite advantage is the NaN that pairs with
	// the -Inf the other surrogate branch produces. fmath takes the instruction and consults
	// math only when the instruction returns NaN, which is the only result the two can disagree
	// on, so the loss is bit-identical.
	nf, nok := f64Data(lpNew)
	of, ook := f64Data(lpOld)
	if nok && ook {
		for i, l := range pa.Lengths {
			var d float64
			//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
			for t := 0; t < l; t++ {
				d += nf[off+t] - of[off+t]
			}
			s := math.Exp(d / float64(l))
			a := adv.AtF64(i)
			surr := fmath.Min(s*a, fmath.Max(1-eps, fmath.Min(1+eps, s))*a)
			sum += surr
			off += l
		}
	} else {
		for i, l := range pa.Lengths {
			var d float64
			for t := range l {
				d += lpNew.AtF64(off+t) - lpOld.AtF64(off+t)
			}
			s := math.Exp(d / float64(l))
			a := adv.AtF64(i)
			surr := fmath.Min(s*a, fmath.Max(1-eps, fmath.Min(1+eps, s))*a)
			sum += surr
			off += l
		}
	}
	out := tensor.NewOn(ctx.Device(), lpNew.Dtype(), tensor.Shape{})
	out.SetF64(-sum / float64(g))
	return []*tensor.Tensor{out}, nil
}

func init() {
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpGSPO, tensor.F32, gspoKernel)
	std.add(backend.OpGSPO, tensor.F64, gspoKernel)
}
