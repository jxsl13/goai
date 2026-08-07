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

// EmbedForward gathers rows of a device-resident embedding table into out: out[i] = table[ids[i]].
// The table is a *DeviceF32 (a trainable parameter) rather than a resident weight — the forward
// companion to EmbedBackward for GPU training. out is [seq, d]; table is [vocab, d]; ids is a host slice
// of token indices (uploaded internally).
func EmbedForward(out, table *DeviceF32, ids []int32) error {
	seq, d := out.rows, out.cols
	if table.cols != d {
		return fmt.Errorf("cuda: EmbedForward table cols %d != d %d", table.cols, d)
	}
	if len(ids) != seq {
		return fmt.Errorf("cuda: EmbedForward %d ids != %d rows", len(ids), seq)
	}
	dIds := C.cu_upload_i32((*C.int)(unsafe.Pointer(&ids[0])), C.int(seq))
	if dIds == nil {
		return fmt.Errorf("cuda: EmbedForward id upload failed")
	}
	defer C.cu_free_f32(dIds)
	if rc := C.cu_embed_f32(table.ptr, dIds, out.ptr, C.int(seq), C.int(d)); rc != 0 {
		return fmt.Errorf("cuda: EmbedForward rc=%d", int(rc))
	}
	return nil
}
