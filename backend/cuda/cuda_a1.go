//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"unsafe"

	"github.com/jxsl13/goai/backend"
)

// A1 fp16-activation kernel wrappers (raw device pointers). These thread f16 (u16) activations
// through the decode forward so the per-GEMM f32<->f16 conversions vanish (measured ~5-7% e2e),
// and make the vLLM benchmark fully f16-matched (Iw8). Exported for the parity test + f16 forward.

func AllocU16(n int) unsafe.Pointer { return unsafe.Pointer(C.cu_alloc_u16(C.int(n))) }
func FreeDev(p unsafe.Pointer)      { C.cu_free_f32(p) }

func CvtF32ToF16(dst16, src32 unsafe.Pointer, n int) int {
	return int(C.cu_cvt_f32_to_f16(dst16, src32, C.long(n)))
}
func DownloadU16(src unsafe.Pointer, dst []uint16) {
	if len(dst) == 0 {
		return
	}
	C.cu_download_u16(src, (*C.ushort)(&dst[0]), C.int(len(dst)))
}

func RMSNormF16(in, out, gamma unsafe.Pointer, rows, cols int, eps float32) int {
	return int(C.cu_rmsnorm_f16(in, out, gamma, C.int(rows), C.int(cols), C.float(eps)))
}
func SwiGLUF16(gate, up unsafe.Pointer, n int) int {
	return int(C.cu_swiglu_f16(gate, up, C.int(n)))
}
func RoPEF16(x, inv unsafe.Pointer, seq, heads, hd, posOffset int, posDiv float64) int {
	return int(C.cu_rope_f16(x, inv, C.int(seq), C.int(heads), C.int(hd), C.int(posOffset), C.double(posDiv)))
}
func AddF16(dst, src unsafe.Pointer, n int) int { return int(C.cu_add_f16(dst, src, C.int(n))) }

// GemmF16Pure: C16[M,N] = A16[M,K]·W16[K,N], all f16, no per-call conversion (the A1 GEMM).
func GemmF16Pure(a16, w16, c16 unsafe.Pointer, m, k, n int) int {
	return int(C.cu_gemm_f16_pure(a16, w16, c16, C.int(m), C.int(k), C.int(n)))
}

// WPtr exposes a ResidentBF16's raw device f16 weight pointer ([K,N]) for the A1 f16 GEMM path.
func (r *ResidentBF16) WPtr() unsafe.Pointer { return r.ptr }

// DevPtr exposes a DeviceF32's raw device pointer (for building f16 shadows / A1 wiring).
func (d *DeviceF32) DevPtr() unsafe.Pointer { return d.ptr }

// VecPtr exposes a ResidentVec's raw device pointer (gamma for f16 rmsnorm).
func (r *ResidentVec) VecPtr() unsafe.Pointer { return r.ptr }

func CvtF16ToF32(dst32, src16 unsafe.Pointer, n int) int {
	return int(C.cu_cvt_f16_to_f32(dst32, src16, C.long(n)))
}

func AddF16ToF32(dst32, src16 unsafe.Pointer, n int) int {
	return int(C.cu_addf16_to_f32(dst32, src16, C.long(n)))
}

// RoPEF16DposRaw: device-position f16 RoPE on a raw u16 pointer (graph-capturable decode).
func RoPEF16DposRaw(x, inv unsafe.Pointer, seq, heads, hd int, pos *DevicePos, attrs backend.RoPEAttrs) int {
	_, posDiv := backend.RoPEFreqs(hd, attrs)
	return int(C.cu_rope_f16_dpos(x, inv, C.int(seq), C.int(heads), C.int(hd), pos.Ptr(), C.double(posDiv)))
}
