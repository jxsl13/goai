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

// EmbedBackward computes the VJP of the input-embedding row gather out[i] = table[ids[i]]. Given the
// gradient of the gathered rows dOut [seq, d] and the token ids, it accumulates each row's gradient back
// into its source table row: dTable[ids[i]] += dOut[i]. dTable [vocab, d] is zeroed first, then scatter-
// accumulated (tokens sharing an id sum via atomicAdd). This is the embedding-weight gradient for GPU
// training. ids is a host slice (uploaded internally); dOut and dTable are device-resident.
func EmbedBackward(dTable, dOut *DeviceF32, ids []int32) error {
	seq, d := dOut.rows, dOut.cols
	if len(ids) != seq {
		return fmt.Errorf("cuda: EmbedBackward %d ids != %d rows", len(ids), seq)
	}
	if dTable.cols != d {
		return fmt.Errorf("cuda: EmbedBackward dTable cols %d != d %d", dTable.cols, d)
	}
	if rc := C.cu_zero_f32(dTable.ptr, C.int(dTable.rows*dTable.cols)); rc != 0 {
		return fmt.Errorf("cuda: EmbedBackward zero dTable rc=%d", int(rc))
	}
	dIds := C.cu_upload_i32((*C.int)(unsafe.Pointer(&ids[0])), C.int(seq))
	if dIds == nil {
		return fmt.Errorf("cuda: EmbedBackward id upload failed")
	}
	defer C.cu_free_f32(dIds)
	if rc := C.cu_embed_backward_f32(dOut.ptr, dIds, dTable.ptr, C.int(seq), C.int(d)); rc != 0 {
		return fmt.Errorf("cuda: EmbedBackward rc=%d", int(rc))
	}
	return nil
}
