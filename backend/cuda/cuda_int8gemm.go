//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Int8GemmProbe times the cublasGemmEx IMMA int8 GEMM (Ampere int8 tensor cores) at fixed dims.
// Buffers are left uninitialized — int8 GEMM throughput is data-independent, and this is a raw
// speed probe (the W8A8 decode-GEMM lever, gated on whether cublas IMMA beats its own f16 GEMM).
type Int8GemmProbe struct {
	a8, w8, c32 unsafe.Pointer
	m, k, n     int
}

func NewInt8GemmProbe(m, k, n int) (*Int8GemmProbe, error) {
	a8 := C.cu_alloc_i8(C.int(m * k))
	w8 := C.cu_alloc_i8(C.int(k * n))
	c32 := C.cu_alloc_i8(C.int(m * n * 4)) // int32 output
	if a8 == nil || w8 == nil || c32 == nil {
		freeIf(a8)
		freeIf(w8)
		freeIf(c32)
		return nil, fmt.Errorf("cuda: Int8GemmProbe alloc failed")
	}
	return &Int8GemmProbe{a8: a8, w8: w8, c32: c32, m: m, k: k, n: n}, nil
}

func (p *Int8GemmProbe) Run() error {
	if rc := C.cu_gemm_int8(p.a8, p.w8, p.c32, C.int(p.m), C.int(p.k), C.int(p.n)); rc != 0 {
		return fmt.Errorf("cuda: cu_gemm_int8 rc=%d", int(rc))
	}
	return nil
}

func (p *Int8GemmProbe) Free() {
	C.cu_free_f32(p.a8)
	C.cu_free_f32(p.w8)
	C.cu_free_f32(p.c32)
	p.a8, p.w8, p.c32 = nil, nil, nil
}

// F16PureProbe times cu_gemm_f16_pure (f16 in, f16 out, NO per-call f32<->f16 convert) vs the
// converting cu_matmul_f16w path — the delta is A1's per-GEMM conversion cost (fp16-activations lever).
type F16PureProbe struct {
	a16, w16, c16 unsafe.Pointer
	m, k, n       int
}

func NewF16PureProbe(m, k, n int) (*F16PureProbe, error) {
	a16 := C.cu_alloc_u16(C.int(m * k))
	w16 := C.cu_alloc_u16(C.int(k * n))
	c16 := C.cu_alloc_u16(C.int(m * n))
	if a16 == nil || w16 == nil || c16 == nil {
		freeIf(a16)
		freeIf(w16)
		freeIf(c16)
		return nil, fmt.Errorf("cuda: F16PureProbe alloc failed")
	}
	return &F16PureProbe{a16: a16, w16: w16, c16: c16, m: m, k: k, n: n}, nil
}

func (p *F16PureProbe) Run() error {
	if rc := matmulF16pure(p.a16, p.w16, p.c16, p.m, p.k, p.n); rc != 0 {
		return fmt.Errorf("cuda: matmulF16pure rc=%d", rc)
	}
	return nil
}

func (p *F16PureProbe) Free() {
	C.cu_free_f32(p.a16)
	C.cu_free_f32(p.w16)
	C.cu_free_f32(p.c16)
	p.a16, p.w16, p.c16 = nil, nil, nil
}
