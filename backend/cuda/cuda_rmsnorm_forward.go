//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "fmt"

// RMSNormForward computes out = x·(1/√(mean(x²)+eps))·gamma over the last axis, with gamma a device
// tensor (a trainable parameter) rather than a resident weight — the forward companion to
// RMSNormBackward for GPU training. x and out are [rows, cols]; gamma is length cols.
func RMSNormForward(out, x, gamma *DeviceF32, eps float32) error {
	rows, cols := x.rows, x.cols
	if out.rows != rows || out.cols != cols {
		return fmt.Errorf("cuda: RMSNormForward out shape mismatch")
	}
	if gamma.rows*gamma.cols != cols {
		return fmt.Errorf("cuda: RMSNormForward gamma len != cols %d", cols)
	}
	if rc := C.cu_rmsnorm_f32(x.ptr, out.ptr, gamma.ptr, C.int(rows), C.int(cols), C.float(eps)); rc != 0 {
		return fmt.Errorf("cuda: RMSNormForward rc=%d", int(rc))
	}
	return nil
}
