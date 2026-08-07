//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "fmt"

// MatMulGradW computes the linear-layer weight gradient dW[K,N] = xᵀ·dY, where x is the [M,K] layer
// input and dY the [M,N] output gradient — all device-resident. This is the transpose-A GEMM the
// backward of Y = x·W needs; combined with DeviceAdam it lets a linear layer train entirely on the GPU.
func MatMulGradW(x, dY, dW *DeviceF32) error {
	m, k := x.rows, x.cols
	if dY.rows != m {
		return fmt.Errorf("cuda: MatMulGradW x[%d,%d]·dY[%d,%d] row mismatch", x.rows, x.cols, dY.rows, dY.cols)
	}
	n := dY.cols
	if dW.rows != k || dW.cols != n {
		return fmt.Errorf("cuda: MatMulGradW dW[%d,%d], want [%d,%d]", dW.rows, dW.cols, k, n)
	}
	if rc := C.cu_matmul_f32_ddd_at(x.ptr, dY.ptr, dW.ptr, C.int(m), C.int(k), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: MatMulGradW rc=%d", int(rc))
	}
	return nil
}

// MatMulGradX computes the linear-layer input gradient dX[M,K] = dY·Wᵀ, where dY is [M,N] and W is
// [K,N] — all device-resident. dX[m,k] = Σ_n dY[m,n]·W[k,n]; the A·Bᵀ GEMM with contraction over N.
func MatMulGradX(dY, w, dX *DeviceF32) error {
	m, n := dY.rows, dY.cols
	if w.cols != n {
		return fmt.Errorf("cuda: MatMulGradX dY[%d,%d]·W[%d,%d] inner mismatch", dY.rows, dY.cols, w.rows, w.cols)
	}
	k := w.rows
	if dX.rows != m || dX.cols != k {
		return fmt.Errorf("cuda: MatMulGradX dX[%d,%d], want [%d,%d]", dX.rows, dX.cols, m, k)
	}
	if rc := C.cu_matmul_f32_ddd_bt(dY.ptr, w.ptr, dX.ptr, C.int(m), C.int(n), C.int(k)); rc != 0 {
		return fmt.Errorf("cuda: MatMulGradX rc=%d", int(rc))
	}
	return nil
}
