package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
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
	for c := range d {
		wc, uc := w.AtF64(c), u.AtF64(c)
		aa, bb, pp := 0.0, 0.0, -1e38 // running numerator, denominator, max exponent
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
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpWKV, tensor.F32, wkvKernel)
	std.add(backend.OpWKV, tensor.F64, wkvKernel)
}
