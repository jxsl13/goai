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

		// Devirtualized fast paths: switch on dtype once, take raw row-major
		// slices, and index by hand — same math, same forward-t / inner-i
		// iteration order, same accumulation order. All read/written tensors
		// share the drive dtype (dk/dv/dw/du are allocated from k/v/w/u), so
		// the guard only has to confirm k,v,w,u,g agree before committing.
		// (k,v,g are [seq,d] row-major → element (r,c) at r*d+c; w,u are [d].)
		switch k.Dtype() {
		case tensor.F64:
			if v.Dtype() == tensor.F64 && w.Dtype() == tensor.F64 && u.Dtype() == tensor.F64 && g.Dtype() == tensor.F64 {
				ks := k.Contiguous().Storage().F64()
				vs := v.Contiguous().Storage().F64()
				ws := w.Contiguous().Storage().F64()
				us := u.Contiguous().Storage().F64()
				gs := g.Contiguous().Storage().F64()
				dks := dk.Storage().F64()
				dvs := dv.Storage().F64()
				dws := dw.Storage().F64()
				dus := du.Storage().F64()
				for c := 0; c < d; c++ {
					wc, uc := ws[c], us[c]
					var dwc, duc float64
					for t := 0; t < seq; t++ {
						gt := gs[t*d+c]
						m := math.Inf(-1)
						for i := 0; i <= t; i++ {
							a := ks[i*d+c] - float64(t-1-i)*wc
							if i == t {
								a = uc + ks[t*d+c]
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
							wkv += p[i] * vs[i*d+c]
						}
						wkv /= den
						if gt == 0 {
							continue
						}
						for i := 0; i <= t; i++ {
							pi := p[i] / den
							vi := vs[i*d+c]
							dvs[i*d+c] += gt * pi
							dks[i*d+c] += gt * pi * (vi - wkv)
							if i == t {
								duc += gt * pi * (vi - wkv)
							} else {
								dwc -= float64(t-1-i) * gt * pi * (vi - wkv)
							}
						}
					}
					dws[c] = dwc
					dus[c] = duc
				}
				return []*tensor.Tensor{dk, dv, dw, du}, nil
			}
		case tensor.F32:
			if v.Dtype() == tensor.F32 && w.Dtype() == tensor.F32 && u.Dtype() == tensor.F32 && g.Dtype() == tensor.F32 {
				ks := k.Contiguous().Storage().F32()
				vs := v.Contiguous().Storage().F32()
				ws := w.Contiguous().Storage().F32()
				us := u.Contiguous().Storage().F32()
				gs := g.Contiguous().Storage().F32()
				dks := dk.Storage().F32()
				dvs := dv.Storage().F32()
				dws := dw.Storage().F32()
				dus := du.Storage().F32()
				// Read inputs as float64, keep all scan state in float64, and
				// round only on store — matching the original AtF64/SetF64
				// rounding (each accumulating store to an F32 tensor rounds).
				for c := 0; c < d; c++ {
					wc, uc := float64(ws[c]), float64(us[c])
					var dwc, duc float64
					for t := 0; t < seq; t++ {
						gt := float64(gs[t*d+c])
						m := math.Inf(-1)
						for i := 0; i <= t; i++ {
							a := float64(ks[i*d+c]) - float64(t-1-i)*wc
							if i == t {
								a = uc + float64(ks[t*d+c])
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
							wkv += p[i] * float64(vs[i*d+c])
						}
						wkv /= den
						if gt == 0 {
							continue
						}
						for i := 0; i <= t; i++ {
							pi := p[i] / den
							vi := float64(vs[i*d+c])
							dvs[i*d+c] = float32(float64(dvs[i*d+c]) + gt*pi)
							dks[i*d+c] = float32(float64(dks[i*d+c]) + gt*pi*(vi-wkv))
							if i == t {
								duc += gt * pi * (vi - wkv)
							} else {
								dwc -= float64(t-1-i) * gt * pi * (vi - wkv)
							}
						}
					}
					dws[c] = float32(dwc)
					dus[c] = float32(duc)
				}
				return []*tensor.Tensor{dk, dv, dw, du}, nil
			}
		}

		// Generic fallback (exotic/mixed dtypes): original AtF64/SetF64 loop.
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
