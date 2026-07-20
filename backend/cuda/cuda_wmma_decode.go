//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	_ "embed"
	"fmt"
	"math"
	"unsafe"
)

//go:embed kernels/wmma_paged_decode.fatbin
var wmmaPagedDecodeFatbin []byte

// BatchedDecodeAttnViewWMMA runs the tensor-core (mma.h) paged decode attention — QK and PV as WMMA
// GEMMs over the GQA group's queries (M=16 tile), breaking the warp-primitive kernel's latency-bound
// GEMV (~0.64 TFLOP/s, 34% of the A1 step). q is [batch, qHeads*hd] f32; returns o[batch, qHeads*hd].
// hd==64, group≤8, blockSize≤16, seqLen≤128.
func (p *PagedKVPool) BatchedDecodeAttnViewWMMA(q *DeviceF32, view *PagedBatchView, qHeads, kvHeads int) (*DeviceF32, error) {
	if view.pool != p {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewWMMA view not from this pool")
	}
	if q.rows != view.batch {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewWMMA q rows %d != batch %d", q.rows, view.batch)
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || q.cols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewWMMA bad head config")
	}
	hd := q.cols / qHeads
	if hd != 64 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewWMMA requires hd==64 (got %d)", hd)
	}
	if kvHeads*hd != p.wkv {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewWMMA kvHeads*hd=%d != pool wkv=%d", kvHeads*hd, p.wkv)
	}
	out := C.cu_alloc_f32(C.int(view.batch * q.cols))
	if out == nil {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewWMMA device alloc failed")
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_wmma_paged_decode(unsafe.Pointer(&wmmaPagedDecodeFatbin[0]), C.int(len(wmmaPagedDecodeFatbin)),
		q.ptr, p.k.ptr, p.v.ptr, view.dbt, view.dsl, out,
		C.int(view.batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(p.blockSize), C.int(view.maxBlocks), C.float(scale))
	if rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewWMMA failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: view.batch, cols: q.cols}, nil
}

// BatchedDecodeAttnViewWMMAFlash runs the TILED FlashDecoding WMMA paged decode: online softmax across
// K/V tiles so shared is O(tile) not O(seqLen). Unlike BatchedDecodeAttnViewWMMA (seqLen≤128, all-K/V
// in shared, 1 block/SM) this handles arbitrary seqLen at better occupancy — the long-context regime
// where the warp-GQA kernel degrades and vLLM's Flash-Decoding wins. hd==64, group≤8, blockSize≤16.
func (p *PagedKVPool) BatchedDecodeAttnViewWMMAFlash(q *DeviceF32, view *PagedBatchView, qHeads, kvHeads int) (*DeviceF32, error) {
	if view.pool != p {
		return nil, fmt.Errorf("cuda: WMMAFlash view not from this pool")
	}
	if q.rows != view.batch {
		return nil, fmt.Errorf("cuda: WMMAFlash q rows %d != batch %d", q.rows, view.batch)
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || q.cols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: WMMAFlash bad head config")
	}
	hd := q.cols / qHeads
	if hd != 64 {
		return nil, fmt.Errorf("cuda: WMMAFlash requires hd==64 (got %d)", hd)
	}
	if kvHeads*hd != p.wkv {
		return nil, fmt.Errorf("cuda: WMMAFlash kvHeads*hd=%d != pool wkv=%d", kvHeads*hd, p.wkv)
	}
	out := C.cu_alloc_f32(C.int(view.batch * q.cols))
	if out == nil {
		return nil, fmt.Errorf("cuda: WMMAFlash device alloc failed")
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_wmma_paged_decode_flash(unsafe.Pointer(&wmmaPagedDecodeFatbin[0]), C.int(len(wmmaPagedDecodeFatbin)),
		q.ptr, p.k.ptr, p.v.ptr, view.dbt, view.dsl, out,
		C.int(view.batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(p.blockSize), C.int(view.maxBlocks), C.float(scale))
	if rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: WMMAFlash failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: view.batch, cols: q.cols}, nil
}
