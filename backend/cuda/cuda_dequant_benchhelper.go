//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "unsafe"

// bench-only C wrappers (cgo can't live in _test.go): let the internal dequant bench allocate an f16
// scratch and invoke a format's dequant kernel in isolation, to profile it against the roofline.
func allocU16ForBench(n int) unsafe.Pointer { return C.cu_alloc_u16(C.int(n)) }
func freeF32ForBench(p unsafe.Pointer)      { C.cu_free_f32(p) }
func dequantQ4KForBench(r *ResidentBQ4K, bf16 unsafe.Pointer, k, n int) int {
	return int(C.cu_dequant_q4k_to_f16(r.q, bf16, C.int(k), C.int(n)))
}

func dequantQ6KForBench(r *ResidentBQ6K, bf16 unsafe.Pointer, k, n int) int {
	return int(C.cu_dequant_q6k_to_f16(r.q, bf16, C.int(k), C.int(n)))
}

func dequantQ5KForBench(r *ResidentBQ5K, bf16 unsafe.Pointer, k, n int) int {
	return int(C.cu_dequant_q5k_to_f16(r.q, bf16, C.int(k), C.int(n)))
}

func dequantQ2KForBench(r *ResidentBQ2K, bf16 unsafe.Pointer, k, n int) int {
	return int(C.cu_dequant_q2k_to_f16(r.q, bf16, C.int(k), C.int(n)))
}
