//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "fmt"

// SoftmaxBackward computes the VJP of a row-wise softmax p = softmax(s): given the probabilities p and
// the upstream gradient dp, it writes ds_j = p_j*(dp_j - Σ_k p_k*dp_k) (the diag(p)-p·pᵀ Jacobian-vector
// product), all device-resident. This is the core of the attention backward and of any softmax VJP.
func SoftmaxBackward(ds, p, dp *DeviceF32) error {
	rows, cols := p.rows, p.cols
	if dp.rows != rows || dp.cols != cols || ds.rows != rows || ds.cols != cols {
		return fmt.Errorf("cuda: SoftmaxBackward shape mismatch")
	}
	if rc := C.cu_softmax_backward_f32(p.ptr, dp.ptr, ds.ptr, C.int(rows), C.int(cols)); rc != 0 {
		return fmt.Errorf("cuda: SoftmaxBackward rc=%d", int(rc))
	}
	return nil
}
