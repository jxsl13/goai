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

// CrossEntropyBackward computes the gradient of the mean softmax-cross-entropy loss with respect to the
// logits: dlogits[m,j] = scale*(softmax(logits[m])[j] - [j==targets[m]]). Pass scale = 1/rows for the
// standard mean reduction (1 for sum). logits and dlogits are device-resident [rows, vocab]; targets is
// a host slice of per-row class indices (uploaded internally). This is the language-model loss VJP — the
// gradient that seeds the whole backward pass in GPU training.
func CrossEntropyBackward(dlogits, logits *DeviceF32, targets []int32, scale float32) error {
	rows, cols := logits.rows, logits.cols
	if dlogits.rows != rows || dlogits.cols != cols {
		return fmt.Errorf("cuda: CrossEntropyBackward dlogits[%d,%d] != logits[%d,%d]", dlogits.rows, dlogits.cols, rows, cols)
	}
	if len(targets) != rows {
		return fmt.Errorf("cuda: CrossEntropyBackward %d targets != %d rows", len(targets), rows)
	}
	dTgt := C.cu_upload_i32((*C.int)(unsafe.Pointer(&targets[0])), C.int(rows))
	if dTgt == nil {
		return fmt.Errorf("cuda: CrossEntropyBackward target upload failed")
	}
	defer C.cu_free_f32(dTgt)
	if rc := C.cu_cross_entropy_backward_f32(logits.ptr, dTgt, dlogits.ptr, C.int(rows), C.int(cols), C.float(scale)); rc != 0 {
		return fmt.Errorf("cuda: CrossEntropyBackward rc=%d", int(rc))
	}
	return nil
}
