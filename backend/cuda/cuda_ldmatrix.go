//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "unsafe"

// ldmatrixProbe returns the 128 u32 (32 lanes × 4 regs) ldmatrix.x4.b16 output for the probe tile.
func ldmatrixProbe() []uint32 {
	dOut := C.cu_alloc_i8(C.int(128 * 4))
	defer C.cu_free_f32(dOut)
	if int(C.cu_ldmatrix_probe(dOut)) != 0 {
		return nil
	}
	out := make([]uint32, 128)
	C.cu_download_f32(dOut, (*C.float)(unsafe.Pointer(&out[0])), C.int(128))
	return out
}
