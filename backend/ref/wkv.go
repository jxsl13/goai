package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/fmath"
	"github.com/jxsl13/goai/internal/parallel"
	"github.com/jxsl13/goai/tensor"
)

// wkvKernel is the RWKV-4 WKV time-mixing recurrence (Peng et al. 2023,
// arXiv:2305.13048) as a dispatched op (§T516 — nn.WKV is the same algorithm as
// a host utility; the op form puts it on the tape for training):
//
//	wkv_t = ( Σ_{i<t} e^{−(t−1−i)·w + k_i}·v_i + e^{u + k_t}·v_t )
//	        / ( Σ_{i<t} e^{−(t−1−i)·w + k_i} + e^{u + k_t} )
//
// Inputs (k[seq,D], v[seq,D], w[D] decay, u[D] bonus); numerically stable
// running-max recurrence, linear in seq. The VJP lives in autograd (O(T²)
// reverse pass with per-row log-sum-exp).
func wkvKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("ref: wkv wants (k,v,w,u), got %d inputs", len(in))
	}
	k, v, w, u := in[0], in[1], in[2], in[3]
	if k.Ndim() != 2 || !k.Shape().Equal(v.Shape()) {
		return nil, fmt.Errorf("ref: wkv k/v must be equal rank-2, got %v/%v", k.Shape(), v.Shape())
	}
	seq, d := k.Shape()[0], k.Shape()[1]
	if w.Ndim() != 1 || w.Shape()[0] != d || u.Ndim() != 1 || u.Shape()[0] != d {
		return nil, fmt.Errorf("ref: wkv w,u must be [D=%d], got %v/%v", d, w.Shape(), u.Shape())
	}
	out := tensor.NewOn(ctx.Device(), k.Dtype(), k.Shape())

	// Devirtualised fast paths (§T645): the generic AtF64/SetF64 loop below pays a
	// dtype dispatch + flat-offset computation per element. When every read tensor
	// shares the output dtype we grab the raw typed slices once (row-major: (t,c) of
	// [seq,d] at t*d+c, channel c of [d] at c) and index directly. The channel/time
	// iteration order, the per-channel running numerator/denominator/max state, the
	// max-tracking exp rescaling and every accumulation are byte-for-byte identical
	// to the generic path; the running state stays float64 on ALL paths and the F32
	// path only rounds the STORED output. Contiguous() is called once per tensor
	// (returns self when already contiguous).
	switch out.Dtype() {
	case tensor.F64:
		kc, vc, wcv, ucv := k.Contiguous(), v.Contiguous(), w.Contiguous(), u.Contiguous()
		if kc.Dtype() == tensor.F64 && vc.Dtype() == tensor.F64 &&
			wcv.Dtype() == tensor.F64 && ucv.Dtype() == tensor.F64 {
			ks, vs := kc.Storage().F64(), vc.Storage().F64()
			ws, us := wcv.Storage().F64(), ucv.Storage().F64()
			os := out.Storage().F64()
			parallel.Rows(d, func(clo, chi int) {
				for c := clo; c < chi; c++ {
					wc, uc := ws[c], us[c]
					aa, bb, pp := 0.0, 0.0, -1e38 // running numerator, denominator, max exponent
					for t := range seq {
						//perfscan:ignore PS6011 reference oracle: intentionally simple, correctness baseline not an optimization target
						kk, vv := ks[t*d+c], vs[t*d+c]
						ww := uc + kk
						q := fmath.Max(pp, ww)
						e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
						os[t*d+c] = (e1*aa + e2*vv) / (e1*bb + e2)
						q = fmath.Max(pp-wc, kk)
						e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
						aa = e1*aa + e2*vv
						bb = e1*bb + e2
						pp = q
					}
				}
			})
			return []*tensor.Tensor{out}, nil
		}
	case tensor.F32:
		kc, vc, wcv, ucv := k.Contiguous(), v.Contiguous(), w.Contiguous(), u.Contiguous()
		if kc.Dtype() == tensor.F32 && vc.Dtype() == tensor.F32 &&
			wcv.Dtype() == tensor.F32 && ucv.Dtype() == tensor.F32 {
			ks, vs := kc.Storage().F32(), vc.Storage().F32()
			ws, us := wcv.Storage().F32(), ucv.Storage().F32()
			os := out.Storage().F32()
			parallel.Rows(d, func(clo, chi int) {
				for c := clo; c < chi; c++ {
					wc, uc := float64(ws[c]), float64(us[c])
					aa, bb, pp := 0.0, 0.0, -1e38 // running state stays float64; only the store rounds
					for t := range seq {
						//perfscan:ignore PS6011 reference oracle: intentionally simple, correctness baseline not an optimization target
						kk, vv := float64(ks[t*d+c]), float64(vs[t*d+c])
						ww := uc + kk
						q := fmath.Max(pp, ww)
						e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
						os[t*d+c] = float32((e1*aa + e2*vv) / (e1*bb + e2))
						q = fmath.Max(pp-wc, kk)
						e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
						aa = e1*aa + e2*vv
						bb = e1*bb + e2
						pp = q
					}
				}
			})
			return []*tensor.Tensor{out}, nil
		}
	}

	// Generic fallback for exotic dtypes / mixed inputs (verbatim original loop).
	for c := range d {
		wc, uc := w.AtF64(c), u.AtF64(c)
		aa, bb, pp := 0.0, 0.0, -1e38 // running numerator, denominator, max exponent
		for t := range seq {
			kk, vv := k.AtF64(t, c), v.AtF64(t, c)
			ww := uc + kk
			q := fmath.Max(pp, ww)
			e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
			out.SetF64((e1*aa+e2*vv)/(e1*bb+e2), t, c)
			q = fmath.Max(pp-wc, kk)
			e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
			aa = e1*aa + e2*vv
			bb = e1*bb + e2
			pp = q
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpWKV, tensor.F32, wkvKernel)
	std.add(backend.OpWKV, tensor.F64, wkvKernel)
}
