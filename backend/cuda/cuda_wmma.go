//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	_ "embed"
	"fmt"
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
