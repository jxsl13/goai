//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	_ "embed"
	"fmt"
	"math"
	"unsafe"
)

// Tw-FLASHATTN slice 1: the first nvcc-compiled tensor-core kernel in the CUDA backend. The
// mma.h/WMMA GEMM below is built by nvcc (scripts/cuda-nvcc-env.sh, `make cuda-cubin`) into a
// committed fatbin and loaded at runtime via cuModuleLoadDataEx — the path nvrtc could not
// take. This proves the nvcc->cubin->driver pipeline before the fused FlashAttention prefill
// kernel (the real lever on vLLM's prefill lead) is built on it. Plain GEMM is correctness-
// only here (cuBLAS is faster); the win is the fused kernel this unblocks.

//go:embed kernels/wmma_gemm.fatbin
var wmmaGemmFatbin []byte

// WMMAGemm computes C[M,N] f32 = A[M,K] · B[K,N] on tensor cores (inputs rounded to f16 on
// upload, f32 accumulate). A, B, C are f32 host slices; M, N, K must be multiples of 16.
func WMMAGemm(a, b, c []float32, m, k, n int) error {
	if m%16 != 0 || k%16 != 0 || n%16 != 0 {
		return fmt.Errorf("cuda: WMMA GEMM needs M,K,N %% 16 == 0 (got %d,%d,%d)", m, k, n)
	}
	if len(a) != m*k || len(b) != k*n || len(c) != m*n {
		return fmt.Errorf("cuda: WMMA GEMM buffer sizes A=%d B=%d C=%d for [%d,%d,%d]", len(a), len(b), len(c), m, k, n)
	}
	da := C.cu_upload_f16((*C.float)(&a[0]), C.long(len(a)))
	db := C.cu_upload_f16((*C.float)(&b[0]), C.long(len(b)))
	dc := C.cu_alloc_f32(C.int(m * n))
	if da == nil || db == nil || dc == nil {
		freeIf(da)
		freeIf(db)
		freeIf(dc)
		return fmt.Errorf("cuda: WMMA GEMM device alloc failed")
	}
	defer C.cu_free_f32(da)
	defer C.cu_free_f32(db)
	defer C.cu_free_f32(dc)
	rc := C.cu_wmma_gemm(unsafe.Pointer(&wmmaGemmFatbin[0]), C.int(len(wmmaGemmFatbin)),
		da, db, dc, C.int(m), C.int(k), C.int(n))
	if rc != 0 {
		return fmt.Errorf("cuda: WMMA GEMM failed (code %d)", int(rc))
	}
	if C.cu_download_f32(dc, (*C.float)(&c[0]), C.int(m*n)) != 0 {
		return fmt.Errorf("cuda: WMMA GEMM result download failed")
	}
	return nil
}

func freeIf(p unsafe.Pointer) {
	if p != nil {
		C.cu_free_f32(p)
	}
}

//go:embed kernels/wmma_attn.fatbin
var wmmaAttnFatbin []byte

// WMMAAttn computes fused causal prefill attention O[heads,seq,hd] = softmax(scale·Q·Kᵀ +
// causal)·V on tensor cores in one nvcc-compiled kernel. Q,K,V,O are f32 host slices laid out
// [heads,seq,hd] row-major; hd must be 64 and seq a multiple of 16. Inputs round to f16.
func WMMAAttn(q, k, v, o []float32, heads, seq, hd int, scale float32) error {
	if hd != 64 {
		return fmt.Errorf("cuda: WMMAAttn currently requires hd==64 (got %d)", hd)
	}
	if seq%16 != 0 {
		return fmt.Errorf("cuda: WMMAAttn needs seq %% 16 == 0 (got %d)", seq)
	}
	n := heads * seq * hd
	if len(q) != n || len(k) != n || len(v) != n || len(o) != n {
		return fmt.Errorf("cuda: WMMAAttn buffer sizes must all be heads*seq*hd=%d", n)
	}
	dq := C.cu_upload_f16((*C.float)(&q[0]), C.long(n))
	dk := C.cu_upload_f16((*C.float)(&k[0]), C.long(n))
	dv := C.cu_upload_f16((*C.float)(&v[0]), C.long(n))
	do := C.cu_alloc_f32(C.int(n))
	if dq == nil || dk == nil || dv == nil || do == nil {
		freeIf(dq)
		freeIf(dk)
		freeIf(dv)
		freeIf(do)
		return fmt.Errorf("cuda: WMMAAttn device alloc failed")
	}
	defer C.cu_free_f32(dq)
	defer C.cu_free_f32(dk)
	defer C.cu_free_f32(dv)
	defer C.cu_free_f32(do)
	rc := C.cu_wmma_attn(unsafe.Pointer(&wmmaAttnFatbin[0]), C.int(len(wmmaAttnFatbin)),
		dq, dk, dv, do, C.int(heads), C.int(seq), C.int(hd), C.float(scale))
	if rc != 0 {
		return fmt.Errorf("cuda: WMMAAttn failed (code %d)", int(rc))
	}
	if C.cu_download_f32(do, (*C.float)(&o[0]), C.int(n)) != 0 {
		return fmt.Errorf("cuda: WMMAAttn download failed")
	}
	return nil
}

// wmmaAttnPtr is the raw-pointer form of WMMAAttn for internal resident-buffer benchmarks
// (cgo cannot be used directly in _test.go). dq/dk/dv are f16 device buffers, do is f32.
func wmmaAttnPtr(dq, dk, dv, do unsafe.Pointer, heads, seq, hd int, scale float32) int {
	return int(C.cu_wmma_attn(unsafe.Pointer(&wmmaAttnFatbin[0]), C.int(len(wmmaAttnFatbin)),
		dq, dk, dv, do, C.int(heads), C.int(seq), C.int(hd), C.float(scale)))
}

//go:embed kernels/wmma_attn_gqa.fatbin
var wmmaAttnGqaFatbin []byte

// GroupedQueryAttentionWMMA is the fused-tensor-core drop-in for GroupedQueryAttention on the
// prefill forward: causal attention over q[seq,qHeads·hd], k/v[seq,kvHeads·hd] f32 device
// buffers, returning o[seq,qHeads·hd] f32 — computed in one nvcc mma.h kernel (no materialized
// global scores). hd must be 64 and seq a multiple of 16. Measured 3.0x the cuBLAS-batched path.
func GroupedQueryAttentionWMMA(q, k, v *DeviceF32, qHeads, kvHeads int) (*DeviceF32, error) {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil {
		return nil, fmt.Errorf("cuda: GQA-WMMA on a freed handle")
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || q.cols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: GQA-WMMA bad head config q=%d kv=%d width=%d", qHeads, kvHeads, q.cols)
	}
	seq := q.rows
	hd := q.cols / qHeads
	if hd != 64 {
		return nil, fmt.Errorf("cuda: GQA-WMMA requires hd==64 (got %d)", hd)
	}
	if seq%16 != 0 {
		return nil, fmt.Errorf("cuda: GQA-WMMA needs seq %% 16 == 0 (got %d)", seq)
	}
	if k.cols != kvHeads*hd || v.cols != kvHeads*hd || k.rows != seq || v.rows != seq {
		return nil, fmt.Errorf("cuda: GQA-WMMA k/v shape mismatch")
	}
	out := C.cu_alloc_f32(C.int(seq * q.cols))
	if out == nil {
		return nil, fmt.Errorf("cuda: GQA-WMMA output alloc failed")
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_wmma_attn_gqa(unsafe.Pointer(&wmmaAttnGqaFatbin[0]), C.int(len(wmmaAttnGqaFatbin)),
		q.ptr, k.ptr, v.ptr, out, C.int(seq), C.int(qHeads), C.int(kvHeads), C.int(hd), C.float(scale))
	if rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: GQA-WMMA failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: seq, cols: q.cols}, nil
}
