package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// WKV computes the RWKV-4 WKV operator (Peng et al. 2023, "RWKV: Reinventing RNNs
// for the Transformer Era", EMNLP Findings, arXiv:2305.13048) — the linear-attention
// "time-mixing" that gives RWKV Transformer-level quality with RNN-style O(1)-per-step
// inference. Unlike GLA/DeltaNet's matrix-valued state, WKV is D INDEPENDENT scalar
// weighted-averages (one per channel): each output is a recency-decayed average of the
// past values plus the current token with a special bonus weight (Eq. for wkv_t):
//
//	wkv_t = ( Σ_{i<t} e^{−(t−1−i)·w + k_i}·v_i + e^{u + k_t}·v_t )
//	        / ( Σ_{i<t} e^{−(t−1−i)·w + k_i} + e^{u + k_t} )
//
// w > 0 is the per-channel time-decay (older tokens fade as e^{−(t−1−i)w}) and u is
// the per-channel bonus for the current token (it gets weight e^{u+k_t} instead of
// decaying). At t=1 the output is simply v_1. This computes only the WKV core; the
// full RWKV time-mixing then applies the receptance gate σ(r)⊙(W_o·wkv).
//
// k, v are [seq, D]; w (decay) and u (bonus) are [D]. Returns [seq, D]. Implemented
// with the numerically-stable running-max recurrence (a pure-f64 forward utility, like
// RetentionRecurrent). For TRAINING, dispatch backend.OpWKV instead — the same
// recurrence as an op with a registered VJP (§T516).
func WKV(k, v, w, u *tensor.Tensor) (*tensor.Tensor, error) {
	if k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("nn: WKV wants rank-2 [seq,D] k,v, got %v, %v", k.Shape(), v.Shape())
	}
	if !k.Shape().Equal(v.Shape()) {
		return nil, fmt.Errorf("nn: WKV k/v shape mismatch: %v vs %v", k.Shape(), v.Shape())
	}
	seq, d := k.Shape()[0], k.Shape()[1]
	if w.Ndim() != 1 || w.Shape()[0] != d || u.Ndim() != 1 || u.Shape()[0] != d {
		return nil, fmt.Errorf("nn: WKV w,u must be [D=%d], got %v, %v", d, w.Shape(), u.Shape())
	}
	out := tensor.New(k.Dtype(), k.Shape())
	// Typed fast paths: the recurrence reads k[t,c]/v[t,c] and writes out[t,c] on every
	// (c,t) — three interface dispatches per element, comparable to the four math.Exp. Grab
	// the contiguous typed slices once (w,u are [D]); the AtF64 loop stays as the exotic-dtype
	// fallback. Bit-identical: the same reads/writes and the same per-channel recurrence order.
	kc, vc := k.Contiguous(), v.Contiguous()
	wc0, uc0 := w.Contiguous(), u.Contiguous()
	switch k.Dtype() {
	case tensor.F64:
		ks, vs, os := kc.Storage().F64(), vc.Storage().F64(), out.Storage().F64()
		ws, us := wc0.Storage().F64(), uc0.Storage().F64()
		for c := range d {
			wc, uc := ws[c], us[c]
			aa, bb, pp := 0.0, 0.0, -1e38
			for t := range seq {
				o := t*d + c
				kk, vv := ks[o], vs[o]
				ww := uc + kk
				q := math.Max(pp, ww)
				e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
				os[o] = (e1*aa + e2*vv) / (e1*bb + e2)
				q = math.Max(pp-wc, kk)
				e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
				aa = e1*aa + e2*vv
				bb = e1*bb + e2
				pp = q
			}
		}
		return out, nil
	case tensor.F32:
		ks, vs, os := kc.Storage().F32(), vc.Storage().F32(), out.Storage().F32()
		ws, us := wc0.Storage().F32(), uc0.Storage().F32()
		for c := range d {
			wc, uc := float64(ws[c]), float64(us[c])
			aa, bb, pp := 0.0, 0.0, -1e38
			for t := range seq {
				o := t*d + c
				kk, vv := float64(ks[o]), float64(vs[o])
				ww := uc + kk
				q := math.Max(pp, ww)
				e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
				os[o] = float32((e1*aa + e2*vv) / (e1*bb + e2))
				q = math.Max(pp-wc, kk)
				e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
				aa = e1*aa + e2*vv
				bb = e1*bb + e2
				pp = q
			}
		}
		return out, nil
	}
	for c := range d { // exotic-dtype fallback: per-element dispatch
		wc, uc := w.AtF64(c), u.AtF64(c)
		aa, bb, pp := 0.0, 0.0, -1e38 // running numerator, denominator, and max exponent
		for t := range seq {
			kk, vv := k.AtF64(t, c), v.AtF64(t, c)
			ww := uc + kk
			q := math.Max(pp, ww)
			e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
			out.SetF64((e1*aa+e2*vv)/(e1*bb+e2), t, c)
			q = math.Max(pp-wc, kk)
			e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
			aa = e1*aa + e2*vv
			bb = e1*bb + e2
			pp = q
		}
	}
	return out, nil
}
