package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
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

	var sum float64
	off := 0
	for i, l := range pa.Lengths {
		var d float64
		for t := range l {
			d += lpNew.AtF64(off+t) - lpOld.AtF64(off+t)
		}
		s := math.Exp(d / float64(l))
		a := adv.AtF64(i)
		surr := math.Min(s*a, math.Max(1-eps, math.Min(1+eps, s))*a)
		sum += surr
		off += l
	}
	out := tensor.NewOn(ctx.Device(), lpNew.Dtype(), tensor.Shape{})
	out.SetF64(-sum / float64(g))
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpGSPO, tensor.F32, gspoKernel)
	std.add(backend.OpGSPO, tensor.F64, gspoKernel)
}
