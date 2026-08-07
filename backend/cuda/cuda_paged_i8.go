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

// PagedAppendBatchedI8 quantizes one fresh f32 K/V token per sequence (dk/dv are [batch, wkv]) into the
// int8 pool (poolK8/poolV8) with a per-(physical-token, kvHead) symmetric f32 scale (poolKs/poolVs) —
// the quantizing device-side append for int8-primary paged serving. It writes at each sequence's current
// length slot (from seqLens); the caller advances the lengths (as with the f16 AppendBatchedDev).
func PagedAppendBatchedI8(poolK8, poolV8, poolKs, poolVs, blockTables, seqLens unsafe.Pointer, dk, dv *DeviceF32,
	batch, wkv, kvHeads, hd, blockSize, maxBlocks int) error {
	if poolK8 == nil || poolV8 == nil || poolKs == nil || poolVs == nil || dk.ptr == nil || dv.ptr == nil {
		return fmt.Errorf("cuda: PagedAppendBatchedI8 on a nil buffer")
	}
	if rc := C.cu_paged_append_batched_i8(poolK8, poolV8, poolKs, poolVs, blockTables, seqLens, dk.ptr, dv.ptr,
		C.int(batch), C.int(wkv), C.int(kvHeads), C.int(hd), C.int(blockSize), C.int(maxBlocks)); rc != 0 {
		return fmt.Errorf("cuda: cu_paged_append_batched_i8 rc=%d", int(rc))
	}
	return nil
}
