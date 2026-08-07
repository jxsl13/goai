//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"math"
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

// AppendBatchedDevI8 quantizes one fresh f32 K/V token per sequence (dk/dv are [batch, wkv]) into the
// int8-primary pool at each sequence's current length (from view), exactly as AppendBatchedDev does for
// the f32 pool — pair it with SeqKV.Advance. The pool must come from NewPagedKVPoolI8.
func (p *PagedKVPool) AppendBatchedDevI8(dk, dv *DeviceF32, view *PagedBatchView) error {
	if p.k8 == nil {
		return fmt.Errorf("cuda: AppendBatchedDevI8 on a non-int8 pool (use NewPagedKVPoolI8)")
	}
	if view.pool != p {
		return fmt.Errorf("cuda: AppendBatchedDevI8 view not from this pool")
	}
	if dk.rows != view.batch || dv.rows != view.batch || dk.cols != p.wkv || dv.cols != p.wkv {
		return fmt.Errorf("cuda: AppendBatchedDevI8 shape mismatch")
	}
	hd := p.wkv / p.i8Heads
	return PagedAppendBatchedI8(p.k8, p.v8, p.ks.DevPtr(), p.vs.DevPtr(), view.dbt, view.dsl, dk, dv,
		view.batch, p.wkv, p.i8Heads, hd, p.blockSize, view.maxBlocks)
}

// BatchedDecodeAttnViewGQAI8 runs the int8-KV paged decode attention over the int8-primary pool for a
// batch (one query token per sequence), returning [batch, qHeads*hd] f32. The int8 twin of
// BatchedDecodeAttnViewGQAf16 — same view, half the KV bytes of f16, already faster per the #1049 bench.
func (p *PagedKVPool) BatchedDecodeAttnViewGQAI8(q *DeviceF32, view *PagedBatchView, qHeads, kvHeads int) (*DeviceF32, error) {
	if p.k8 == nil {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAI8 on a non-int8 pool (use NewPagedKVPoolI8)")
	}
	if view.pool != p {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAI8 view not from this pool")
	}
	if q.rows != view.batch {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAI8 q rows %d != batch %d", q.rows, view.batch)
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || q.cols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAI8 bad head config q=%d kv=%d width=%d", qHeads, kvHeads, q.cols)
	}
	hd := q.cols / qHeads
	if hd != 64 && hd != 128 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAI8 requires hd==64 or 128 (got %d)", hd)
	}
	if kvHeads != p.i8Heads || kvHeads*hd != p.wkv {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAI8 head layout kv=%d hd=%d != pool wkv=%d/i8Heads=%d", kvHeads, hd, p.wkv, p.i8Heads)
	}
	out, err := NewDeviceF32(view.batch, q.cols)
	if err != nil {
		return nil, err
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	if err := PagedDecodeAttnGQAI8(q, p.k8, p.v8, p.ks.DevPtr(), p.vs.DevPtr(), view.dbt, view.dsl, out,
		view.batch, qHeads, kvHeads, hd, p.blockSize, view.maxBlocks, scale); err != nil {
		out.Free()
		return nil, err
	}
	return out, nil
}
