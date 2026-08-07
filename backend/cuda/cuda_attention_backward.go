//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "fmt"

// Scale multiplies a device tensor in place by a scalar.
func Scale(x *DeviceF32, scale float32) error {
	if rc := C.cu_scale_f32(x.ptr, C.int(x.rows*x.cols), C.float(scale)); rc != 0 {
		return fmt.Errorf("cuda: Scale rc=%d", int(rc))
	}
	return nil
}

// AttentionBackward computes the VJP of single-head scaled-dot-product attention (no mask):
//
//	S = (Q·Kᵀ)·scale ;  P = softmax(S) row-wise ;  O = P·V
//
// Given the output gradient dO it writes dQ, dK, dV, all device-resident [L, d] (L = sequence length,
// d = head dim). scale is typically 1/√d. It recomputes S and P internally (flash-attn style), using
// [L,L] scratch, so no forward activations need to be stored. This composes the matmul backward GEMMs
// with SoftmaxBackward — the attention piece of transformer GPU training.
func AttentionBackward(dQ, dK, dV, Q, K, V, dO *DeviceF32, scale float32) error {
	L, d := Q.rows, Q.cols
	if K.rows != L || K.cols != d || V.rows != L || V.cols != d || dO.rows != L || dO.cols != d {
		return fmt.Errorf("cuda: AttentionBackward Q/K/V/dO must all be [%d,%d]", L, d)
	}
	if dQ.rows != L || dQ.cols != d || dK.rows != L || dK.cols != d || dV.rows != L || dV.cols != d {
		return fmt.Errorf("cuda: AttentionBackward dQ/dK/dV must all be [%d,%d]", L, d)
	}
	p, err := NewDeviceF32(L, L) // holds S then P
	if err != nil {
		return err
	}
	defer p.Free()
	dP, err := NewDeviceF32(L, L)
	if err != nil {
		return err
	}
	defer dP.Free()
	dS, err := NewDeviceF32(L, L)
	if err != nil {
		return err
	}
	defer dS.Free()

	// Forward recompute: S = Q·Kᵀ·scale, P = softmax(S).
	if err := MatMulGradX(Q, K, p); err != nil { // p = Q·Kᵀ  [L,L]
		return err
	}
	if err := Scale(p, scale); err != nil {
		return err
	}
	if rc := C.cu_softmax_f32(p.ptr, C.int(L), C.int(L)); rc != 0 { // p = softmax(S)
		return fmt.Errorf("cuda: AttentionBackward softmax rc=%d", int(rc))
	}
	// dV = Pᵀ·dO
	if err := MatMulGradW(p, dO, dV); err != nil {
		return err
	}
	// dP = dO·Vᵀ
	if err := MatMulGradX(dO, V, dP); err != nil {
		return err
	}
	// dS = softmax_backward(P, dP), then grad w.r.t. Q·Kᵀ = dS·scale.
	if err := SoftmaxBackward(dS, p, dP); err != nil {
		return err
	}
	if err := Scale(dS, scale); err != nil {
		return err
	}
	// dQ = dS·K ; dK = dSᵀ·Q
	if err := MatMul(dS, K, dQ); err != nil {
		return err
	}
	if err := MatMulGradW(dS, Q, dK); err != nil {
		return err
	}
	return nil
}
