//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/jxsl13/goai/tensor"
)

// ResidentBF16QKV is the fused Q/K/V projection weight: wq|wk|wv concatenated along N into
// ONE resident f16 [K, nq+nk+nv]. Prefill then runs a single wide GEMM instead of three
// separate ones. Measured (BenchmarkF16Proj_*, RTX 3060, M=128): the fused GEMM is 1.39× the
// three separate calls — the tiny GQA k/v projections (N=head_dim·kv_heads, e.g. 256) waste
// tensor cores as their own GEMMs, but ride along at full utilization inside the N=2560 fused
// call. gate/up is deliberately NOT fused (measured a 1.07× LOSS — both already large and
// well-utilized, so the wider N only picks a worse cuBLAS tile).
type ResidentBF16QKV struct {
	ptr           unsafe.Pointer // device f16 weight [K, nq+nk+nv]
	k, nq, nk, nv int
}

// NewResidentBF16QKV concatenates wq|wk|wv (each [K, Nx], any float dtype) along N and uploads
// the result as one resident f16 weight. The three must share the inner dim K.
func NewResidentBF16QKV(wq, wk, wv *tensor.Tensor) (*ResidentBF16QKV, error) {
	for _, w := range []*tensor.Tensor{wq, wk, wv} {
		if w.Ndim() != 2 {
			return nil, fmt.Errorf("cuda: NewResidentBF16QKV needs rank-2, got %dD", w.Ndim())
		}
	}
	qc := wq.Cast(tensor.F32).Contiguous()
	kc := wk.Cast(tensor.F32).Contiguous()
	vc := wv.Cast(tensor.F32).Contiguous()
	k := qc.Shape()[0]
	if kc.Shape()[0] != k || vc.Shape()[0] != k {
		return nil, fmt.Errorf("cuda: NewResidentBF16QKV inner-dim mismatch q[%d,*] k[%d,*] v[%d,*]",
			qc.Shape()[0], kc.Shape()[0], vc.Shape()[0])
	}
	nq, nk, nv := qc.Shape()[1], kc.Shape()[1], vc.Shape()[1]
	ntot := nq + nk + nv
	qf, kf, vf := qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32()
	// Interleave rows: [K, ntot] row-major = per row k the concat [wq row | wk row | wv row].
	cat := make([]float32, k*ntot)
	for r := 0; r < k; r++ {
		off := r * ntot
		copy(cat[off:off+nq], qf[r*nq:(r+1)*nq])
		copy(cat[off+nq:off+nq+nk], kf[r*nk:(r+1)*nk])
		copy(cat[off+nq+nk:off+ntot], vf[r*nv:(r+1)*nv])
	}
	p := C.cu_upload_f16((*C.float)(&cat[0]), C.long(len(cat)))
	if p == nil {
		return nil, fmt.Errorf("cuda: fused-QKV f16 weight upload failed [%d,%d]", k, ntot)
	}
	return &ResidentBF16QKV{ptr: p, k: k, nq: nq, nk: nk, nv: nv}, nil
}

// MatMulQKV runs the single fused GEMM a·[wq|wk|wv] and splits the [M, nq+nk+nv] result into
// three contiguous outputs q[M,nq], k[M,nk], v[M,nv] via strided column-block copies (cheap
// vs the GEMM: ~M·ntot floats moved once). Bit-identical per element to three separate
// MatMulDevice calls — the split only reorders storage, the arithmetic is unchanged.
func (r *ResidentBF16QKV) MatMulQKV(a *DeviceF32) (q, k, v *DeviceF32, err error) {
	if r.ptr == nil || a.ptr == nil {
		return nil, nil, nil, fmt.Errorf("cuda: fused-QKV MatMulQKV on a freed handle")
	}
	if a.cols != r.k {
		return nil, nil, nil, fmt.Errorf("cuda: fused-QKV inner dim mismatch a[%d,%d]·W[%d,*]", a.rows, a.cols, r.k)
	}
	m := a.rows
	ntot := r.nq + r.nk + r.nv
	out := C.cu_alloc_f32(C.int(m * ntot))
	if out == nil {
		return nil, nil, nil, fmt.Errorf("cuda: fused-QKV output alloc failed")
	}
	defer C.cu_free_f32(out)
	var rc C.int
	if f16AccEnabled() {
		rc = C.cu_matmul_f16w_acc16(a.ptr, r.ptr, out, C.int(m), C.int(r.k), C.int(ntot), C.float(0))
	} else {
		rc = C.cu_matmul_f16w(a.ptr, r.ptr, out, C.int(m), C.int(r.k), C.int(ntot), C.float(0))
	}
	if rc != 0 {
		return nil, nil, nil, fmt.Errorf("cuda: fused-QKV matmul failed (code %d)", int(rc))
	}
	// Column-block split: q=[0,nq), k=[nq,nq+nk), v=[nq+nk,ntot) of each row-major row.
	alloc := func(cols int) (*DeviceF32, unsafe.Pointer) {
		p := C.cu_alloc_f32(C.int(m * cols))
		if p == nil {
			return nil, nil
		}
		return &DeviceF32{ptr: p, rows: m, cols: cols}, p
	}
	dq, pq := alloc(r.nq)
	dk, pk := alloc(r.nk)
	dv, pv := alloc(r.nv)
	if pq == nil || pk == nil || pv == nil {
		if pq != nil {
			C.cu_free_f32(pq)
		}
		if pk != nil {
			C.cu_free_f32(pk)
		}
		if pv != nil {
			C.cu_free_f32(pv)
		}
		return nil, nil, nil, fmt.Errorf("cuda: fused-QKV split alloc failed")
	}
	split := func(dst unsafe.Pointer, srcOff, cols int) C.int {
		return C.cu_copy2d(dst, 0, C.int(cols), out, C.int(srcOff), C.int(ntot), C.int(m), C.int(cols))
	}
	if split(pq, 0, r.nq) != 0 || split(pk, r.nq, r.nk) != 0 || split(pv, r.nq+r.nk, r.nv) != 0 {
		C.cu_free_f32(pq)
		C.cu_free_f32(pk)
		C.cu_free_f32(pv)
		return nil, nil, nil, fmt.Errorf("cuda: fused-QKV column split failed")
	}
	return dq, dk, dv, nil
}

// Free releases the resident fused-QKV weight.
func (r *ResidentBF16QKV) Free() {
	if r.ptr != nil {
		C.cu_free_f32(r.ptr)
		r.ptr = nil
	}
}
