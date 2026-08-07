//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "fmt"

// This file adds bf16×bf16 GEMM wrappers (both operands already resident as DeviceBf16, no per-call
// conversion). They exist to (a) support concat-fused projection GEMMs — where the shared-LHS activation
// and the concatenated weight are both cached bf16 — and (b) bf16 activation residency. The forward,
// weight-grad, and input-grad variants mirror cu_matmul_bf16_ddd / _ddd_at / _ddd_bt.

// MatMulBB computes out[M,N]f32 = a[M,K]bf16 · b[K,N]bf16 (both operands pre-cached bf16).
func MatMulBB(a, b *DeviceBf16, out *DeviceF32) error {
	m, k := a.rows, a.cols
	if b.rows != k {
		return fmt.Errorf("cuda: MatMulBB a[%d,%d]·b[%d,%d] inner mismatch", a.rows, a.cols, b.rows, b.cols)
	}
	n := b.cols
	if out.rows != m || out.cols != n {
		return fmt.Errorf("cuda: MatMulBB out[%d,%d], want [%d,%d]", out.rows, out.cols, m, n)
	}
	if rc := C.cu_matmul_bf16_ddd(a.ptr, b.ptr, out.ptr, C.int(m), C.int(k), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: MatMulBB rc=%d", int(rc))
	}
	return nil
}

// MatMulBBGradW computes dW[K,N]f32 = x[M,K]bf16ᵀ · dY[M,N]bf16 — weight gradient, both activations bf16.
func MatMulBBGradW(x, dY *DeviceBf16, dW *DeviceF32) error {
	m, k := x.rows, x.cols
	if dY.rows != m {
		return fmt.Errorf("cuda: MatMulBBGradW x[%d,%d], dY[%d,%d] row mismatch", x.rows, x.cols, dY.rows, dY.cols)
	}
	n := dY.cols
	if dW.rows != k || dW.cols != n {
		return fmt.Errorf("cuda: MatMulBBGradW dW[%d,%d], want [%d,%d]", dW.rows, dW.cols, k, n)
	}
	if rc := C.cu_matmul_bf16_ddd_at(x.ptr, dY.ptr, dW.ptr, C.int(m), C.int(k), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: MatMulBBGradW rc=%d", int(rc))
	}
	return nil
}

// MatMulBBGradX computes dX[M,K]f32 = dY[M,N]bf16 · w[K,N]bf16ᵀ — input gradient, dY bf16 + w cached bf16.
func MatMulBBGradX(dY, w *DeviceBf16, dX *DeviceF32) error {
	m, n := dY.rows, dY.cols
	if w.cols != n {
		return fmt.Errorf("cuda: MatMulBBGradX dY[%d,%d], w[%d,%d] inner mismatch", dY.rows, dY.cols, w.rows, w.cols)
	}
	k := w.rows
	if dX.rows != m || dX.cols != k {
		return fmt.Errorf("cuda: MatMulBBGradX dX[%d,%d], want [%d,%d]", dX.rows, dX.cols, m, k)
	}
	if rc := C.cu_matmul_bf16_ddd_bt(dY.ptr, w.ptr, dX.ptr, C.int(m), C.int(n), C.int(k)); rc != 0 {
		return fmt.Errorf("cuda: MatMulBBGradX rc=%d", int(rc))
	}
	return nil
}
