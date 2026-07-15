//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"math"
	"unsafe"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// General-backend op kernels (§I4): make backend.Get(CUDA) a fuller general backend, not
// just OpMatMul. Each kernel is the naive synchronous shape — upload the host input(s),
// run the existing device kernel, download the result — so a caller using the generic
// backend.Execute path runs these on the GPU instead of falling back to the CPU reference.
// This is the composability/coverage path (per-op H2D/D2H, transfer-bound; the FAST
// inference path stays the device-resident engine, ADR-0021). F32 only — F64 and any
// unsupported shape return (nil,false)/fall through so the executor uses Pure-Go.

// refFallback re-runs op on the reference backend (for shapes/dtypes the device path
// doesn't cover), so registering a CUDA kernel never changes a result vs Pure-Go.
func refFallback(ctx *backend.Context, op backend.Op, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), op, in, attrs)
}

// deviceUnary uploads x, applies an in-place device kernel over all its floats, downloads.
func deviceUnary(x *tensor.Tensor, kern func(d unsafe.Pointer, n C.int) C.int) (*tensor.Tensor, bool) {
	xc := x.Contiguous()
	s := xc.Storage().F32()
	if len(s) == 0 {
		return nil, false
	}
	d := C.cu_upload_f32((*C.float)(&s[0]), C.int(len(s)))
	if d == nil {
		return nil, false
	}
	defer C.cu_free_f32(d)
	if kern(d, C.int(len(s))) != 0 {
		return nil, false
	}
	out := tensor.New(tensor.F32, xc.Shape())
	if C.cu_download_f32(d, (*C.float)(&out.Storage().F32()[0]), C.int(len(s))) != 0 {
		return nil, false
	}
	return out, true
}

func siluF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 1 && in[0].Dtype() == tensor.F32 {
		if out, ok := deviceUnary(in[0], func(d unsafe.Pointer, n C.int) C.int { return C.cu_silu_f32(d, n) }); ok {
			return []*tensor.Tensor{out}, nil
		}
	}
	return refFallback(ctx, backend.OpSiLU, in, attrs)
}

func geluF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 1 && in[0].Dtype() == tensor.F32 {
		if out, ok := deviceUnary(in[0], func(d unsafe.Pointer, n C.int) C.int { return C.cu_gelu_f32(d, n) }); ok {
			return []*tensor.Tensor{out}, nil
		}
	}
	return refFallback(ctx, backend.OpGELU, in, attrs)
}

// deviceBinary uploads a and b (same numel), applies dst OP= src in place on a, downloads.
func deviceBinary(a, b *tensor.Tensor, kern func(dst, src unsafe.Pointer, n C.int) C.int) (*tensor.Tensor, bool) {
	ac, bc := a.Contiguous(), b.Contiguous()
	as, bs := ac.Storage().F32(), bc.Storage().F32()
	if len(as) == 0 || len(as) != len(bs) { // broadcast/empty → caller falls back
		return nil, false
	}
	da := C.cu_upload_f32((*C.float)(&as[0]), C.int(len(as)))
	if da == nil {
		return nil, false
	}
	defer C.cu_free_f32(da)
	db := C.cu_upload_f32((*C.float)(&bs[0]), C.int(len(bs)))
	if db == nil {
		return nil, false
	}
	defer C.cu_free_f32(db)
	if kern(da, db, C.int(len(as))) != 0 {
		return nil, false
	}
	out := tensor.New(tensor.F32, ac.Shape())
	if C.cu_download_f32(da, (*C.float)(&out.Storage().F32()[0]), C.int(len(as))) != 0 {
		return nil, false
	}
	return out, true
}

func addF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 2 && in[0].Dtype() == tensor.F32 && in[1].Dtype() == tensor.F32 {
		if out, ok := deviceBinary(in[0], in[1], func(dst, src unsafe.Pointer, n C.int) C.int { return C.cu_add_f32(dst, src, n) }); ok {
			return []*tensor.Tensor{out}, nil
		}
	}
	return refFallback(ctx, backend.OpAdd, in, attrs)
}

func mulF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 2 && in[0].Dtype() == tensor.F32 && in[1].Dtype() == tensor.F32 {
		if out, ok := deviceBinary(in[0], in[1], func(dst, src unsafe.Pointer, n C.int) C.int { return C.cu_mul_f32(dst, src, n) }); ok {
			return []*tensor.Tensor{out}, nil
		}
	}
	return refFallback(ctx, backend.OpMul, in, attrs)
}

func normEps(attrs backend.Attrs) float32 {
	if na, ok := attrs.(backend.NormAttrs); ok {
		return float32(na.WithDefaults().Eps)
	}
	return float32(backend.NormAttrs{}.WithDefaults().Eps)
}

// deviceNorm uploads x (+ gamma, and beta for LayerNorm), runs the device norm over the
// last axis, and downloads. gamma/beta must be [d] where d is x's last dim.
func deviceNorm(x, gamma, beta *tensor.Tensor, eps float32) (*tensor.Tensor, bool) {
	xc := x.Contiguous()
	nd := xc.Ndim()
	if nd < 1 {
		return nil, false
	}
	d := xc.Shape()[nd-1]
	if d == 0 || gamma.Ndim() != 1 || gamma.Shape()[0] != d {
		return nil, false
	}
	if beta != nil && (beta.Ndim() != 1 || beta.Shape()[0] != d) {
		return nil, false
	}
	rows := xc.Numel() / d
	xs := xc.Storage().F32()
	gs := gamma.Cast(tensor.F32).Contiguous().Storage().F32()
	dx := C.cu_upload_f32((*C.float)(&xs[0]), C.int(len(xs)))
	if dx == nil {
		return nil, false
	}
	defer C.cu_free_f32(dx)
	dout := C.cu_alloc_f32(C.int(len(xs)))
	if dout == nil {
		return nil, false
	}
	defer C.cu_free_f32(dout)
	dg := C.cu_upload_f32((*C.float)(&gs[0]), C.int(len(gs)))
	if dg == nil {
		return nil, false
	}
	defer C.cu_free_f32(dg)
	if beta != nil {
		bs := beta.Cast(tensor.F32).Contiguous().Storage().F32()
		db := C.cu_upload_f32((*C.float)(&bs[0]), C.int(len(bs)))
		if db == nil {
			return nil, false
		}
		defer C.cu_free_f32(db)
		if C.cu_layernorm_f32(dx, dout, dg, db, C.int(rows), C.int(d), C.float(eps)) != 0 {
			return nil, false
		}
	} else if C.cu_rmsnorm_f32(dx, dout, dg, C.int(rows), C.int(d), C.float(eps)) != 0 {
		return nil, false
	}
	out := tensor.New(tensor.F32, xc.Shape())
	if C.cu_download_f32(dout, (*C.float)(&out.Storage().F32()[0]), C.int(len(xs))) != 0 {
		return nil, false
	}
	return out, true
}

func rmsnormF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 2 && in[0].Dtype() == tensor.F32 && in[1].Dtype() == tensor.F32 {
		if out, ok := deviceNorm(in[0], in[1], nil, normEps(attrs)); ok {
			return []*tensor.Tensor{out}, nil
		}
	}
	return refFallback(ctx, backend.OpRMSNorm, in, attrs)
}

func layernormF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 3 && in[0].Dtype() == tensor.F32 && in[1].Dtype() == tensor.F32 && in[2].Dtype() == tensor.F32 {
		if out, ok := deviceNorm(in[0], in[1], in[2], normEps(attrs)); ok {
			return []*tensor.Tensor{out}, nil
		}
	}
	return refFallback(ctx, backend.OpLayerNorm, in, attrs)
}

// ropeF32 runs rotary embeddings on the GPU for the 2D non-XPos case (PI/YaRN scaling is
// carried by backend.RoPEFreqs); XPos, non-2D and non-F32 fall back to the reference.
func ropeF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	ra, ok := attrs.(backend.RoPEAttrs)
	if len(in) == 1 && in[0].Dtype() == tensor.F32 && in[0].Ndim() == 2 && ok && !ra.XPos {
		if d, err := UploadF32(in[0]); err == nil {
			defer d.Free()
			if d.RoPE(ra) == nil {
				if out, e := d.ToHost(); e == nil {
					return []*tensor.Tensor{out}, nil
				}
			}
		}
	}
	return refFallback(ctx, backend.OpRoPE, in, attrs)
}

// mhaF32 runs (grouped-query) multi-head attention on the GPU: Q,K,V are 2D [seq,dmodel]
// (K/V may be longer than Q for KV-cache decode), causal + score-scale + GQA supported via
// Recorder.MHA. ALiBi, sliding-window (window < sk) and score shapes the device path can't
// serve fall back to the reference, so the result never changes.
func mhaF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	pa, ok := attrs.(backend.AttnAttrs)
	if ok && len(in) == 3 && in[0].Dtype() == tensor.F32 && in[1].Dtype() == tensor.F32 && in[2].Dtype() == tensor.F32 &&
		in[0].Ndim() == 2 && in[1].Ndim() == 2 && in[2].Ndim() == 2 {
		pa = pa.WithDefaults()
		q, k, v := in[0], in[1], in[2]
		sq, dm := q.Shape()[0], q.Shape()[1]
		sk := k.Shape()[0]
		heads, kvHeads := pa.Heads, pa.KVHeads
		// Only the plain (no-ALiBi) GQA-causal path is on device; everything else → ref.
		if !pa.ALiBi && heads > 0 && dm%heads == 0 && kvHeads > 0 && heads%kvHeads == 0 {
			dk := dm / heads
			if k.Shape()[1] == kvHeads*dk && v.Shape().Equal(k.Shape()) && sq <= sk &&
				!(pa.Window > 0 && pa.Window < sk) {
				if out, done := mhaDevice(q, k, v, sq, sk, dm, heads, kvHeads, dk, pa); done {
					return []*tensor.Tensor{out}, nil
				}
			}
		}
	}
	return refFallback(ctx, backend.OpMHA, in, attrs)
}

func mhaDevice(q, k, v *tensor.Tensor, sq, sk, dm, heads, kvHeads, dk int, pa backend.AttnAttrs) (*tensor.Tensor, bool) {
	dq, err := UploadF32(q)
	if err != nil {
		return nil, false
	}
	defer dq.Free()
	dkb, err := UploadF32(k)
	if err != nil {
		return nil, false
	}
	defer dkb.Free()
	dv, err := UploadF32(v)
	if err != nil {
		return nil, false
	}
	defer dv.Free()
	out, err := NewDeviceF32(sq, dm)
	if err != nil {
		return nil, false
	}
	defer out.Free()
	rec, err := NewRecorder()
	if err != nil {
		return nil, false
	}
	defer rec.Free()
	causal := 0
	if pa.Causal {
		causal = 1
	}
	scale := float32(pa.Scale / math.Sqrt(float64(dk)))
	if rec.MHA(dq, dkb, dv, out, sq, sk, dm, heads, kvHeads, dk, causal, pa.Window, scale) != nil {
		return nil, false
	}
	h, err := out.ToHost()
	if err != nil {
		return nil, false
	}
	return h, true
}

func softmaxF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 1 && in[0].Dtype() == tensor.F32 && in[0].Ndim() >= 1 {
		x := in[0].Contiguous()
		sh := x.Shape()
		cols := sh[len(sh)-1]
		rows := 1
		for i := 0; i < len(sh)-1; i++ {
			rows *= sh[i]
		}
		s := x.Storage().F32()
		if cols > 0 && len(s) == rows*cols {
			d := C.cu_upload_f32((*C.float)(&s[0]), C.int(len(s)))
			if d != nil {
				defer C.cu_free_f32(d)
				if C.cu_softmax_f32(d, C.int(rows), C.int(cols)) == 0 {
					out := tensor.New(tensor.F32, sh)
					if C.cu_download_f32(d, (*C.float)(&out.Storage().F32()[0]), C.int(len(s))) == 0 {
						return []*tensor.Tensor{out}, nil
					}
				}
			}
		}
	}
	return refFallback(ctx, backend.OpSoftmax, in, attrs)
}
