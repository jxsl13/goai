//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "fmt"

// SwiGLUForward computes out = SiLU(gate)⊙up into a separate buffer, leaving gate and up intact so the
// backward pass (SwiGLUBackward) can use them. This is the training forward of the SwiGLU FFN
// nonlinearity (the in-place DeviceF16.SwiGLU / cu_swiglu_f32 would clobber the gate).
func SwiGLUForward(out, gate, up *DeviceF32) error {
	n := gate.rows * gate.cols
	if up.rows*up.cols != n || out.rows*out.cols != n {
		return fmt.Errorf("cuda: SwiGLUForward shape mismatch")
	}
	if rc := C.cu_swiglu_out_f32(out.ptr, gate.ptr, up.ptr, C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: SwiGLUForward rc=%d", int(rc))
	}
	return nil
}
