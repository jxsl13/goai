package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// wkvKernelCPU is the channel-vectorized RWKV-4 WKV time-mixing recurrence (Peng et
// al. 2023, arXiv:2305.13048). The d channels each carry an independent running
// numerator/denominator/max-exponent, so simd.WKVScanF64 runs the numerically-stable
// log-space scan 4 channels at a time (expF64x4v; ~1 ulp vs libm, riding the model
// f64 tolerance — RWKV goldens gate at 1e-9). Same recurrence and iteration order as
// ref.wkvKernel's F64 fast path; the non-AVX build's simd fallback is the byte-for-
// byte scalar scan, so the two backends stay consistent per build. F64-only — F32 and
// exotic/mixed-dtype inputs fall through to the ref scalar kernel; the mixed-dtype
// path below (verbatim ref) guards the rare case where OpWKV dispatches here on F64
// output with a non-F64 input.
func wkvKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("cpu: wkv wants (k,v,w,u), got %d inputs", len(in))
	}
	k, v, w, u := in[0], in[1], in[2], in[3]
	if k.Ndim() != 2 || !k.Shape().Equal(v.Shape()) {
		return nil, fmt.Errorf("cpu: wkv k/v must be equal rank-2, got %v/%v", k.Shape(), v.Shape())
	}
	seq, d := k.Shape()[0], k.Shape()[1]
	if w.Ndim() != 1 || w.Shape()[0] != d || u.Ndim() != 1 || u.Shape()[0] != d {
		return nil, fmt.Errorf("cpu: wkv w,u must be [D=%d], got %v/%v", d, w.Shape(), u.Shape())
	}
	out := tensor.NewOn(ctx.Device(), k.Dtype(), k.Shape())

	// Fast path: all inputs F64-contiguous → the channel-vectorized scan (grabs the
	// raw typed slices once; (t,c) of [seq,d] at t*d+c, channel c of [d] at c).
	kc, vc, wcv, ucv := k.Contiguous(), v.Contiguous(), w.Contiguous(), u.Contiguous()
	if kc.Dtype() == tensor.F64 && vc.Dtype() == tensor.F64 &&
		wcv.Dtype() == tensor.F64 && ucv.Dtype() == tensor.F64 {
		simd.WKVScanF64(kc.Storage().F64(), vc.Storage().F64(),
			wcv.Storage().F64(), ucv.Storage().F64(), out.Storage().F64(), seq, d)
		return []*tensor.Tensor{out}, nil
	}

	// Mixed/exotic dtype fallback: the generic scalar scan (verbatim ref path).
	for c := range d {
		wc, uc := w.AtF64(c), u.AtF64(c)
		aa, bb, pp := 0.0, 0.0, -1e38
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
	std.add(backend.OpWKV, tensor.F64, wkvKernelCPU)
}
