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

// PagedDecodeAttnGQAI8 is the int8-KV twin of the paged f16 decode attention: poolK8/poolV8 are int8
// device buffers ([phys, kvHeads*hd]) with per-(token,kvHead) f32 scales poolKs/poolVs. Half the KV
// VRAM of f16 (a quarter of f32) — more concurrent sequences / longer context on a fixed budget — with
// a vectorized int32 dequant so the per-key compute stays competitive with the f16 path. Low-level
// (raw device pointers) so the caller owns the int8 pool layout. Output o is [batch, qHeads*hd] f32.
func PagedDecodeAttnGQAI8(q *DeviceF32, poolK8, poolV8, poolKs, poolVs, blockTables, seqLens unsafe.Pointer,
	o *DeviceF32, batch, qHeads, kvHeads, hd, blockSize, maxBlocks int, scale float32) error {
	if q.ptr == nil || o.ptr == nil || poolK8 == nil || poolV8 == nil || poolKs == nil || poolVs == nil {
		return fmt.Errorf("cuda: PagedDecodeAttnGQAI8 on a nil buffer")
	}
	if rc := C.cu_paged_decode_attn_gqa_i8(q.ptr, poolK8, poolV8, poolKs, poolVs, blockTables, seqLens, o.ptr,
		C.int(batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(blockSize), C.int(maxBlocks), C.float(scale)); rc != 0 {
		return fmt.Errorf("cuda: cu_paged_decode_attn_gqa_i8 rc=%d", int(rc))
	}
	return nil
}
