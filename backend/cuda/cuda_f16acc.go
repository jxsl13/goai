//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"os"
	"sync"
	"unsafe"
)

// f16AccEnabled reports whether the prefill f16 GEMMs should use f16 ACCUMULATE (Tw61) —
// ≈1.5-2× (+21% e2e) on GeForce, validated accurate on TinyLlama + Qwen 0.5/1.5/3B. It
// DEFAULTS ON for GeForce/consumer cards (which run f32-accumulate at half rate) and OFF for
// datacenter cards (full-rate f32-accumulate, where f16-accum would only cost precision).
// GOAI_CUDA_F16ACC=1/0 overrides the auto-detection either way (exactness on demand).
func f16AccEnabled() bool {
	if v, ok := os.LookupEnv("GOAI_CUDA_F16ACC"); ok {
		return v == "1"
	}
	f16AccOnce.Do(func() { f16AccAuto = C.cu_gpu_is_geforce() == 1 })
	return f16AccAuto
}

var (
	f16AccOnce sync.Once
	f16AccAuto bool
)

// Internal wrappers for the Tw61 f16-accumulate prefill-GEMM probe (used by the internal benchmark).

func i8Upload(b []int8) unsafe.Pointer {
	if len(b) == 0 {
		return nil
	}
	return C.cu_upload_i8((*C.schar)(unsafe.Pointer(&b[0])), C.int(len(b)))
}

func f32Upload(f []float32) unsafe.Pointer {
	if len(f) == 0 {
		return nil
	}
	return C.cu_upload_f32((*C.float)(&f[0]), C.int(len(f)))
}

func f16Upload(f []float32) unsafe.Pointer {
	if len(f) == 0 {
		return nil
	}
	return C.cu_upload_f16((*C.float)(&f[0]), C.long(len(f)))
}

func devAllocBytes(n int) unsafe.Pointer { return C.cu_alloc_i8(C.int(n)) }
func devFree(p unsafe.Pointer)           { C.cu_free_f32(p) }
func devSync()                           { C.cu_graph_sync() }

func matmulF16w(a, w, c unsafe.Pointer, m, k, n int) int {
	return int(C.cu_matmul_f16w(a, w, c, C.int(m), C.int(k), C.int(n), C.float(0)))
}

func matmulF16acc16(a, w, c unsafe.Pointer, m, k, n int) int {
	return int(C.cu_matmul_f16acc16(a, w, c, C.int(m), C.int(k), C.int(n)))
}

func matmulF16pure(a, w, c unsafe.Pointer, m, k, n int) int {
	return int(C.cu_gemm_f16_pure(a, w, c, C.int(m), C.int(k), C.int(n)))
}

// matmulF32ddd is the fully-resident f32 Sgemm baseline (no tensor cores) — the
// current cu_gqa_scores/cu_gqa_out attention GEMM class, for the Tw65 f16-vs-f32 probe.
func matmulF32ddd(a, b, c unsafe.Pointer, m, k, n int) int {
	return int(C.cu_matmul_f32_ddd(a, b, c, C.int(m), C.int(k), C.int(n)))
}

func downloadF32(src unsafe.Pointer, dst []float32) {
	if len(dst) > 0 {
		C.cu_download_f32(src, (*C.float)(&dst[0]), C.int(len(dst)))
	}
}

func downloadU16(src unsafe.Pointer, dst []uint16) {
	if len(dst) > 0 {
		C.cu_download_u16(src, (*C.ushort)(&dst[0]), C.int(len(dst)))
	}
}
