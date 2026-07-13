package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// RWKV-4 WKV VJP (Peng et al. 2023, §T516). Per channel c the forward is a
// softmax-weighted average over the causal window:
//
//	wkv_t = Σ_{i≤t} p_{t,i}·v_i,   p_{t,i} = e^{a_{t,i}} / Σ_{j≤t} e^{a_{t,j}}
//	a_{t,i} = k_i − (t−1−i)·w   (i<t),   a_{t,t} = u + k_t
//
// so the gradients are the standard softmax-average forms: with g_t upstream,
//
//	dv_i = Σ_{t≥i} g_t·p_{t,i}
//	dk_i = Σ_{t≥i} g_t·p_{t,i}·(v_i − wkv_t)
//	du   = Σ_t     g_t·p_{t,t}·(v_t − wkv_t)
//	dw   = Σ_t Σ_{i<t} −(t−1−i)·g_t·p_{t,i}·(v_i − wkv_t)
//
// Implemented as an O(T²)-per-channel reverse pass with per-row log-sum-exp
// stabilization — exact but quadratic; a linear-time backward (the official
// CUDA kernel's reverse recurrence) is a separate optimization task. Passes
// the §V2 finite-difference check.
func init() {
	RegisterVJP(backend.OpWKV, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		k, v, w, u := in[0], in[1], in[2], in[3]
		seq, d := k.Shape()[0], k.Shape()[1]

		dk := tensor.New(k.Dtype(), k.Shape())
		dv := tensor.New(v.Dtype(), v.Shape())
		dw := tensor.New(w.Dtype(), w.Shape())
		du := tensor.New(u.Dtype(), u.Shape())

		loga := make([]float64, seq)
		p := make([]float64, seq)
		for c := range d {
			wc, uc := w.AtF64(c), u.AtF64(c)
			var dwc, duc float64
			for t := range seq {
				gt := g.AtF64(t, c)
				// row t: softmax weights over i ≤ t, stabilized by the row max.
				m := math.Inf(-1)
				for i := 0; i <= t; i++ {
					a := k.AtF64(i, c) - float64(t-1-i)*wc
					if i == t {
						a = uc + k.AtF64(t, c)
					}
					loga[i] = a
					if a > m {
						m = a
					}
				}
				var den, wkv float64
				for i := 0; i <= t; i++ {
					p[i] = math.Exp(loga[i] - m)
					den += p[i]
					wkv += p[i] * v.AtF64(i, c)
				}
				wkv /= den
				if gt == 0 {
					continue
				}
				for i := 0; i <= t; i++ {
					pi := p[i] / den
					vi := v.AtF64(i, c)
					dv.SetF64(dv.AtF64(i, c)+gt*pi, i, c)
					dk.SetF64(dk.AtF64(i, c)+gt*pi*(vi-wkv), i, c)
					if i == t {
						duc += gt * pi * (vi - wkv)
					} else {
						dwc -= float64(t-1-i) * gt * pi * (vi - wkv)
					}
				}
			}
			dw.SetF64(dwc, c)
			du.SetF64(duc, c)
		}
		return []*tensor.Tensor{dk, dv, dw, du}, nil
	})
}
