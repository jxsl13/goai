//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "fmt"

// MatMul computes out[M,N] = a[M,K]·b[K,N], all device-resident (the forward of a linear layer). It is
// the exported f32 device GEMM the GPU training loop uses for the forward pass.
func MatMul(a, b, out *DeviceF32) error {
	m, k := a.rows, a.cols
	if b.rows != k {
		return fmt.Errorf("cuda: MatMul a[%d,%d]·b[%d,%d] inner mismatch", a.rows, a.cols, b.rows, b.cols)
	}
	n := b.cols
	if out.rows != m || out.cols != n {
		return fmt.Errorf("cuda: MatMul out[%d,%d], want [%d,%d]", out.rows, out.cols, m, n)
	}
	if rc := C.cu_matmul_f32_ddd(a.ptr, b.ptr, out.ptr, C.int(m), C.int(k), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: MatMul rc=%d", int(rc))
	}
	return nil
}

// RMSNormBackward computes the VJP of RMSNorm y = x·(1/√(mean(x²)+eps))·gamma over the last axis
// (cols = norm dim, rows = batch). Given the output gradient dy it writes dx and dgamma, both
// device-resident. dgamma is the sum over rows, so it is zeroed then atomically accumulated. x, dy, dx
// are [rows,cols]; gamma and dgamma are length cols.
func RMSNormBackward(dx, dgamma, x, dy, gamma *DeviceF32, eps float32) error {
	rows, cols := x.rows, x.cols
	if dy.rows != rows || dy.cols != cols || dx.rows != rows || dx.cols != cols {
		return fmt.Errorf("cuda: RMSNormBackward x/dy/dx shape mismatch")
	}
	if gamma.rows*gamma.cols != cols || dgamma.rows*dgamma.cols != cols {
		return fmt.Errorf("cuda: RMSNormBackward gamma/dgamma len != cols %d", cols)
	}
	if rc := C.cu_zero_f32(dgamma.ptr, C.int(cols)); rc != 0 {
		return fmt.Errorf("cuda: RMSNormBackward zero dgamma rc=%d", int(rc))
	}
	if rc := C.cu_rmsnorm_backward_f32(x.ptr, dy.ptr, gamma.ptr, dx.ptr, dgamma.ptr, C.int(rows), C.int(cols), C.float(eps)); rc != 0 {
		return fmt.Errorf("cuda: RMSNormBackward rc=%d", int(rc))
	}
	return nil
}

// SwiGLUBackward computes the VJP of o = SiLU(g)⊙u: given the output gradient dO, it writes
// dg = dO⊙u⊙SiLU'(g) and du = dO⊙SiLU(g), all device-resident — the GPU SwiGLU backward for training a
// transformer FFN. g and u are the forward inputs (the gate and up projections).
func SwiGLUBackward(dg, du, g, u, dO *DeviceF32) error {
	n := g.rows * g.cols
	for _, x := range []*DeviceF32{dg, du, u, dO} {
		if x.rows*x.cols != n {
			return fmt.Errorf("cuda: SwiGLUBackward shape mismatch")
		}
	}
	if rc := C.cu_swiglu_backward_f32(dg.ptr, du.ptr, g.ptr, u.ptr, dO.ptr, C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: SwiGLUBackward rc=%d", int(rc))
	}
	return nil
}

// SubScaled computes out = scale*(a-b) element-wise (out may alias a or b). Used for loss gradients,
// e.g. the MSE gradient dL/dY = (2/M)*(Y-T).
func SubScaled(out, a, b *DeviceF32, scale float32) error {
	n := a.rows * a.cols
	if b.rows*b.cols != n || out.rows*out.cols != n {
		return fmt.Errorf("cuda: SubScaled shape mismatch")
	}
	if rc := C.cu_sub_scaled_f32(out.ptr, a.ptr, b.ptr, C.int(n), C.float(scale)); rc != 0 {
		return fmt.Errorf("cuda: SubScaled rc=%d", int(rc))
	}
	return nil
}

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
