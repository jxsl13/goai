//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
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
