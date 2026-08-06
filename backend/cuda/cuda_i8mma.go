//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "unsafe"

// i8MMA computes C[M,N] (int32) = A[M,K]·W[K,N] (both int8 row-major) via tiled mma.sync int8
// tensor cores. Internal — the primitive the int8-MMQ prefill GEMM will build on.
func i8MMA(a, w []int8, m, k, n int) []int32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a[0])), C.int(len(a)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&w[0])), C.int(len(w)))
	dC := C.cu_alloc_i8(C.int(m * n * 4))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mma(dA, dW, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]int32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}

func i8mmaRaw(dA, dW, dC unsafe.Pointer, m, k, n int) {
	C.cu_matmul_i8_mma(dA, dW, dC, C.int(m), C.int(k), C.int(n))
}
func i8uploadRaw(b []int8) unsafe.Pointer {
	return unsafe.Pointer(C.cu_upload_i8((*C.schar)(unsafe.Pointer(&b[0])), C.int(len(b))))
}
func i8allocRaw(n int) unsafe.Pointer { return unsafe.Pointer(C.cu_alloc_i8(C.int(n))) }
func i8syncRaw()                      { C.cu_graph_sync() }

func i8mmatRaw(dA, dW, dC unsafe.Pointer, m, k, n int) {
	C.cu_matmul_i8_mma_t(dA, dW, dC, C.int(m), C.int(k), C.int(n))
}

// i8MMAT (tiled) — same contract as i8MMA, N%64==0.
func i8MMAT(a, w []int8, m, k, n int) []int32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a[0])), C.int(len(a)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&w[0])), C.int(len(w)))
	dC := C.cu_alloc_i8(C.int(m * n * 4))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mma_t(dA, dW, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]int32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}

func i8mmarbRaw(dA, dW, dC unsafe.Pointer, m, k, n int) {
	C.cu_matmul_i8_mma_rb(dA, dW, dC, C.int(m), C.int(k), C.int(n))
}

// i8MMARB (register-blocked) — same contract as i8MMA. M%64==0, N%64==0, K%32==0.
func i8MMARB(a, w []int8, m, k, n int) []int32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a[0])), C.int(len(a)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&w[0])), C.int(len(w)))
	dC := C.cu_alloc_i8(C.int(m * n * 4))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mma_rb(dA, dW, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]int32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}

func i8mmadbRaw(dA, dW, dC unsafe.Pointer, m, k, n int) {
	C.cu_matmul_i8_mma_db(dA, dW, dC, C.int(m), C.int(k), C.int(n))
}

// i8MMADB (double-buffered cp.async) — same contract as i8MMARB. M%64==0, N%64==0, K%32==0.
func i8MMADB(a, w []int8, m, k, n int) []int32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a[0])), C.int(len(a)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&w[0])), C.int(len(w)))
	dC := C.cu_alloc_i8(C.int(m * n * 4))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mma_db(dA, dW, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]int32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}

func i8mmawtRaw(dA, dWt, dC unsafe.Pointer, m, k, n int) {
	C.cu_matmul_i8_mma_wt(dA, dWt, dC, C.int(m), C.int(k), C.int(n))
}

// i8MMAWT — W is [N][K] row-major (native GGUF weight layout). C[m][n] = sum_k a[m][k]*wt[n][k].
func i8MMAWT(a, wt []int8, m, k, n int) []int32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a[0])), C.int(len(a)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&wt[0])), C.int(len(wt)))
	dC := C.cu_alloc_i8(C.int(m * n * 4))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mma_wt(dA, dW, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]int32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}

func i8mmawpRaw(dA, dWt, dC unsafe.Pointer, m, k, n int) {
	C.cu_matmul_i8_mma_wp(dA, dWt, dC, C.int(m), C.int(k), C.int(n))
}

// i8MMAWP — padded-stride variant of i8MMAWT. W is [N][K] row-major.
func i8MMAWP(a, wt []int8, m, k, n int) []int32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a[0])), C.int(len(a)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&wt[0])), C.int(len(wt)))
	dC := C.cu_alloc_i8(C.int(m * n * 4))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mma_wp(dA, dW, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]int32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}

// i8MMQ — true per-32-K-block MMQ. a8 [M][K], wt8 [N][K] int8; aSc [M][K/32], wSc [N][K/32] f32.
// Returns f32 C[M][N] = sum_block (sum_k a8*w8) * aSc[block] * wSc[block].
func i8MMQ(a8, wt8 []int8, aSc, wSc []float32, m, k, n int) []float32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a8[0])), C.int(len(a8)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&wt8[0])), C.int(len(wt8)))
	dAs := C.cu_upload_f32((*C.float)(unsafe.Pointer(&aSc[0])), C.int(len(aSc)))
	dWs := C.cu_upload_f32((*C.float)(unsafe.Pointer(&wSc[0])), C.int(len(wSc)))
	dC := C.cu_alloc_f32(C.int(m * n))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dAs)
	defer C.cu_free_f32(dWs)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mmq(dA, dW, dAs, dWs, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]float32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}

// i8MMQLM is i8MMQ using the ldmatrix-core kernel (cu_matmul_i8_mmq_lm) — same math, faster loads.
func i8MMQLM(a8, wt8 []int8, aSc, wSc []float32, m, k, n int) []float32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a8[0])), C.int(len(a8)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&wt8[0])), C.int(len(wt8)))
	dAs := C.cu_upload_f32((*C.float)(unsafe.Pointer(&aSc[0])), C.int(len(aSc)))
	dWs := C.cu_upload_f32((*C.float)(unsafe.Pointer(&wSc[0])), C.int(len(wSc)))
	dC := C.cu_alloc_f32(C.int(m * n))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dAs)
	defer C.cu_free_f32(dWs)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mmq_lm(dA, dW, dAs, dWs, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]float32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}

// i8mmqLmForBench / i8mmqForBench launch the raw kernel on pre-uploaded device pointers (no per-call
// upload) — for the ldmatrix-vs-manual-load A/B benchmark.
func i8mmqLmForBench(dA, dW, dAs, dWs, dC unsafe.Pointer, m, k, n int) int {
	return int(C.cu_matmul_i8_mmq_lm(dA, dW, dAs, dWs, dC, C.int(m), C.int(k), C.int(n)))
}
func i8mmqLm2ForBench(dA, dW, dAs, dWs, dC unsafe.Pointer, m, k, n int) int {
	return int(C.cu_matmul_i8_mmq_lm2(dA, dW, dAs, dWs, dC, C.int(m), C.int(k), C.int(n)))
}

// i8MMQLM2 is i8MMQ using the 128x64-tile ldmatrix kernel.
func i8MMQLM2(a8, wt8 []int8, aSc, wSc []float32, m, k, n int) []float32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a8[0])), C.int(len(a8)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&wt8[0])), C.int(len(wt8)))
	dAs := C.cu_upload_f32((*C.float)(unsafe.Pointer(&aSc[0])), C.int(len(aSc)))
	dWs := C.cu_upload_f32((*C.float)(unsafe.Pointer(&wSc[0])), C.int(len(wSc)))
	dC := C.cu_alloc_f32(C.int(m * n))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dAs)
	defer C.cu_free_f32(dWs)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mmq_lm2(dA, dW, dAs, dWs, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]float32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}
func i8mmqForBench(dA, dW, dAs, dWs, dC unsafe.Pointer, m, k, n int) int {
	return int(C.cu_matmul_i8_mmq(dA, dW, dAs, dWs, dC, C.int(m), C.int(k), C.int(n)))
}

func f32uploadRaw(f []float32) unsafe.Pointer {
	return unsafe.Pointer(C.cu_upload_f32((*C.float)(unsafe.Pointer(&f[0])), C.int(len(f))))
}
func f32allocRaw(n int) unsafe.Pointer { return unsafe.Pointer(C.cu_alloc_f32(C.int(n))) }
func i8mmqRaw(dA, dW, dAs, dWs, dC unsafe.Pointer, m, k, n int) {
	C.cu_matmul_i8_mmq(dA, dW, dAs, dWs, dC, C.int(m), C.int(k), C.int(n))
}

func i8mmqrRaw(dA, dW, dAs, dWs, dC unsafe.Pointer, m, k, n int) {
	C.cu_matmul_i8_mmq_r(dA, dW, dAs, dWs, dC, C.int(m), C.int(k), C.int(n))
}

// i8MMQR — MMQ with per-row activation scale aSc[M] + per-block weight scale wSc[N][K/32].
func i8MMQR(a8, wt8 []int8, aSc, wSc []float32, m, k, n int) []float32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a8[0])), C.int(len(a8)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&wt8[0])), C.int(len(wt8)))
	dAs := C.cu_upload_f32((*C.float)(unsafe.Pointer(&aSc[0])), C.int(len(aSc)))
	dWs := C.cu_upload_f32((*C.float)(unsafe.Pointer(&wSc[0])), C.int(len(wSc)))
	dC := C.cu_alloc_f32(C.int(m * n))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dAs)
	defer C.cu_free_f32(dWs)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mmq_r(dA, dW, dAs, dWs, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]float32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}

func i8quantRowsRaw(dAf, dA8, daSc unsafe.Pointer, m, k int) {
	C.cu_quant_rows_i8(dAf, dA8, daSc, C.int(m), C.int(k))
}

// mmqPrefill: the full quantized-prefill GEMM. f32 activations af[M][K] (host) are uploaded and
// dynamically per-row int8-quantized ON DEVICE, then MMQ'd against resident per-block int8 weights
// wt8[N][K] + wSc[N][K/32]. Returns f32 C[M][N]. Mirrors how prefill would call it with on-GPU acts.
func mmqPrefill(af []float32, wt8 []int8, wSc []float32, m, k, n int) []float32 {
	dAf := f32uploadRaw(af)
	dA8 := i8allocRaw(m * k)
	dAs := f32allocRaw(m)
	dW := unsafe.Pointer(C.cu_upload_i8((*C.schar)(unsafe.Pointer(&wt8[0])), C.int(len(wt8))))
	dWs := f32uploadRaw(wSc)
	dC := f32allocRaw(m * n)
	defer C.cu_free_f32(dAf)
	defer C.cu_free_f32(dA8)
	defer C.cu_free_f32(dAs)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dWs)
	defer C.cu_free_f32(dC)
	i8quantRowsRaw(dAf, dA8, dAs, m, k)
	if int(C.cu_matmul_i8_mmq_r(dA8, dW, dAs, dWs, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]float32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}

func i8mmalmRaw(dA, dWt, dC unsafe.Pointer, m, k, n int) {
	C.cu_matmul_i8_mma_lm(dA, dWt, dC, C.int(m), C.int(k), C.int(n))
}

// i8MMALM — ldmatrix-load variant of i8MMAWP. W is [N][K] row-major. M%64,N%64,K%32==0.
func i8MMALM(a, wt []int8, m, k, n int) []int32 {
	dA := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&a[0])), C.int(len(a)))
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&wt[0])), C.int(len(wt)))
	dC := C.cu_alloc_i8(C.int(m * n * 4))
	defer C.cu_free_f32(dA)
	defer C.cu_free_f32(dW)
	defer C.cu_free_f32(dC)
	if int(C.cu_matmul_i8_mma_lm(dA, dW, dC, C.int(m), C.int(k), C.int(n))) != 0 {
		return nil
	}
	out := make([]int32, m*n)
	C.cu_download_f32(dC, (*C.float)(unsafe.Pointer(&out[0])), C.int(m*n))
	return out
}
