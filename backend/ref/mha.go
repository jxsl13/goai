package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaKernel is fused multi-head scaled dot-product attention (§T32, Vaswani
// 2017): inputs Q,K,V each [seq, dmodel], attrs "heads" (int) and "causal"
// (bool). Head split/concat and the causal mask live INSIDE the kernel, so the
// whole op is one differentiable unit (its VJP is the fused SDPA backward) — no
// view ops on the tape (avoids §B15). Per-head:
//
//	S = (Qₕ·Kₕᵀ)/√dk  (+ causal −∞ for j>i);  A = softmax_j(S);  Oₕ = A·Vₕ
//
// f64 accumulation (§V10). Output columns [h·dk:(h+1)dk] hold head h.
func mhaKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: mha wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("ref: mha needs rank-2 [seq,dmodel]")
	}
	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("ref: mha dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	// GQA (§T38b): kv_heads ≤ heads key/value heads; query head h shares KV head
	// h/(heads/kv_heads). Default kv_heads==heads = standard MHA; kv_heads==1 = MQA.
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("ref: mha heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	// Q may be shorter than K/V (KV-cache incremental decode, §T35): the sq
	// queries align to the LAST sq key positions (abs pos = off+i, off = sk−sq).
	if k.Shape()[1] != kvHeads*dk || !v.Shape().Equal(k.Shape()) {
		return nil, fmt.Errorf("ref: mha K/V must be [sk,%d] (kv_heads·dk), got %v/%v", kvHeads*dk, k.Shape(), v.Shape())
	}
	if sq > sk {
		return nil, fmt.Errorf("ref: mha query len %d exceeds key len %d", sq, sk)
	}
	causal := pa.Causal
	// "attn_scale" multiplies the pre-softmax scores (default 1) — the YaRN
	// attention temperature (backend.YaRNAttnScale, §R66) that keeps the softmax
	// sharpness right when the context is extended. Folded into the 1/√dk scale.
	scale := pa.Scale / math.Sqrt(float64(dk))
	off := sk - sq

	// ALiBi (§T60, §R60): add a static per-head linear bias slopeₕ·(j−iₐᵦₛ) to the
	// pre-softmax scores, penalizing distant keys. No positional embeddings, no
	// learnable params. Applied per query head h with its own slope.
	var slopes []float64
	if pa.ALiBi {
		slopes = backend.ALiBiSlopes(heads)
	}

	// Sliding-window attention (§T62, §R62): a positive "window" restricts each
	// query at abs pos iₐᵦₛ to the W most recent keys, j ∈ [iₐᵦₛ−W+1, jmax); with
	// W ≥ sk it is a no-op (full attention). Mistral-style local attention (used
	// with causal); the effective receptive field grows with depth.
	window := pa.Window

	out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{sq, dm})
	row := make([]float64, sk)
	for h := range heads {
		qOff := h * dk
		kvOff := (h / rep) * dk
		for i := range sq {
			jmax := sk
			if causal {
				jmax = off + i + 1 // query abs pos + 1
			}
			jmin := 0
			if window > 0 {
				if lo := off + i - window + 1; lo > 0 {
					jmin = lo
				}
			}
			m := math.Inf(-1)
			for j := jmin; j < jmax; j++ {
				var s float64
				for d := range dk {
					s += q.AtF64(i, qOff+d) * k.AtF64(j, kvOff+d)
				}
				s *= scale
				if slopes != nil {
					s += slopes[h] * float64(j-(off+i)) // ALiBi bias
				}
				row[j] = s
				if s > m {
					m = s
				}
			}
			var sum float64
			for j := jmin; j < jmax; j++ {
				row[j] = math.Exp(row[j] - m)
				sum += row[j]
			}
			for d := range dk {
				var o float64
				for j := jmin; j < jmax; j++ {
					o += (row[j] / sum) * v.AtF64(j, kvOff+d)
				}
				out.SetF64(o, i, qOff+d)
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpMHA, tensor.F32, mhaKernel)
	std.add(backend.OpMHA, tensor.F64, mhaKernel)
}
