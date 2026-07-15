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

	"github.com/jxsl13/goai/backend"
)

// DevicePos is a device-resident int holding the current decode position. The
// per-token position is updated with Set (a tiny H2D copy) BETWEEN captured-graph
// replays, so the graph itself — which reads the position through this pointer —
// stays structurally constant across tokens.
type DevicePos struct{ ptr unsafe.Pointer }

// NewDevicePos allocates the device position int (initialized to 0).
func NewDevicePos() (*DevicePos, error) {
	zero := C.int(0)
	d := C.cu_upload_i32(&zero, 1)
	if d == nil {
		return nil, fmt.Errorf("cuda: DevicePos alloc failed")
	}
	return &DevicePos{ptr: d}, nil
}

// Set writes the current position (not captured into a graph — call between replays).
func (p *DevicePos) Set(pos int) error {
	if rc := C.cu_set_i32(p.ptr, C.int(pos)); rc != 0 {
		return fmt.Errorf("cuda: DevicePos.Set failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the device int.
func (p *DevicePos) Free() {
	if p.ptr != nil {
		C.cu_free_f32(p.ptr)
		p.ptr = nil
	}
}

// RoPEDpos applies RoPE reading the absolute position from a DevicePos — the
// graph-capturable twin of RoPE.
func (d *DeviceF32) RoPEDpos(attrs backend.RoPEAttrs, pos *DevicePos) error {
	heads := attrs.Heads
	if heads < 1 {
		heads = 1
	}
	hd := d.cols / heads
	base := attrs.Base
	if base == 0 {
		base = 10000
	}
	inv := make([]float32, hd/2)
	for i := range inv {
		inv[i] = float32(1.0 / math.Pow(base, float64(2*i)/float64(hd)))
	}
	invPtr := C.cu_upload_f32((*C.float)(&inv[0]), C.int(len(inv)))
	if invPtr == nil {
		return fmt.Errorf("cuda: RoPEDpos inv upload failed")
	}
	defer C.cu_free_f32(invPtr)
	posDiv := attrs.PosScale
	if posDiv == 0 {
		posDiv = 1
	}
	if rc := C.cu_rope_f32_dpos(d.ptr, invPtr, C.int(d.rows), C.int(heads), C.int(hd), pos.ptr, C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: RoPEDpos failed (code %d)", int(rc))
	}
	return nil
}

// BuildRoPEInv precomputes and uploads the RoPE inverse-frequency table [hd/2]
// for head dim hd and frequency base, ONCE (persistent). Pass it to RoPEDposInv so
// the RoPE op does no per-call host upload — required for the op to be inside a
// captured CUDA graph. Free when done.
func BuildRoPEInv(hd int, base float64) (*DeviceF32, error) {
	if base == 0 {
		base = 10000
	}
	inv := make([]float32, hd/2)
	for i := range inv {
		inv[i] = float32(1.0 / math.Pow(base, float64(2*i)/float64(hd)))
	}
	p := C.cu_upload_f32((*C.float)(&inv[0]), C.int(len(inv)))
	if p == nil {
		return nil, fmt.Errorf("cuda: BuildRoPEInv upload failed")
	}
	return &DeviceF32{ptr: p, rows: 1, cols: hd / 2}, nil
}

// RoPEDposInv applies device-position RoPE using a pre-uploaded inv table (no
// per-call upload) — the graph-capturable RoPE. heads treats d as [seq, heads·hd].
func (d *DeviceF32) RoPEDposInv(heads int, inv *DeviceF32, pos *DevicePos, posScale float64) error {
	if d.ptr == nil || inv.ptr == nil {
		return fmt.Errorf("cuda: RoPEDposInv on a freed handle")
	}
	if heads < 1 {
		heads = 1
	}
	hd := d.cols / heads
	posDiv := posScale
	if posDiv == 0 {
		posDiv = 1
	}
	if rc := C.cu_rope_f32_dpos(d.ptr, inv.ptr, C.int(d.rows), C.int(heads), C.int(hd), pos.ptr, C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: RoPEDposInv failed (code %d)", int(rc))
	}
	return nil
}

// AppendDpos writes one token's k and v ([1,wkv] each) into the cache at row
// *pos — the graph-capturable KV append (does NOT advance c.n; the host tracks
// length separately for the fixed-size graph).
func (c *KVCache) AppendDpos(k, v *DeviceF32, pos *DevicePos) error {
	if k.cols != c.wkv || v.cols != c.wkv || k.rows != 1 || v.rows != 1 {
		return fmt.Errorf("cuda: AppendDpos needs [1,%d] k/v, got k%v v%v", c.wkv, k.shape(), v.shape())
	}
	if rc := C.cu_append_dpos(c.dK, k.ptr, pos.ptr, C.int(c.wkv)); rc != 0 {
		return fmt.Errorf("cuda: AppendDpos K failed (code %d)", int(rc))
	}
	if rc := C.cu_append_dpos(c.dV, v.ptr, pos.ptr, C.int(c.wkv)); rc != 0 {
		return fmt.Errorf("cuda: AppendDpos V failed (code %d)", int(rc))
	}
	return nil
}

// GroupedQueryAttentionKVDpos is GroupedQueryAttentionKV with the causal-mask
// offset read from a DevicePos (the graph-capturable attention). k/v span the
// FULL fixed cache [seqKV,·]; masked positions (key > *pos) contribute nothing.
func GroupedQueryAttentionKVDpos(q, k, v *DeviceF32, qHeads, kvHeads int, off *DevicePos) (*DeviceF32, error) {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil {
		return nil, fmt.Errorf("cuda: GQA-KV-dpos on a freed handle")
	}
	seqQ, wq := q.rows, q.cols
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || wq%qHeads != 0 {
		return nil, fmt.Errorf("cuda: GQA-KV-dpos bad head config")
	}
	hd := wq / qHeads
	seqKV, wkv := k.rows, kvHeads*hd
	if k.cols != wkv || v.rows != seqKV || v.cols != wkv {
		return nil, fmt.Errorf("cuda: GQA-KV-dpos k/v shape")
	}
	scores := C.cu_alloc_f32(C.int(qHeads * seqQ * seqKV))
	if scores == nil {
		return nil, fmt.Errorf("cuda: GQA-KV-dpos scores alloc failed")
	}
	defer C.cu_free_f32(scores)
	if rc := C.cu_gqa_scores(q.ptr, k.ptr, scores, C.int(seqQ), C.int(seqKV), C.int(qHeads), C.int(kvHeads), C.int(hd)); rc != 0 {
		return nil, fmt.Errorf("cuda: GQA-KV-dpos scores failed (code %d)", int(rc))
	}
	if rc := C.cu_attn_softmax_dpos(scores, C.int(qHeads*seqQ), C.int(seqKV), C.float(1/math.Sqrt(float64(hd))), off.ptr, C.int(seqQ)); rc != 0 {
		return nil, fmt.Errorf("cuda: GQA-KV-dpos softmax failed (code %d)", int(rc))
	}
	out := C.cu_alloc_f32(C.int(seqQ * wq))
	if out == nil {
		return nil, fmt.Errorf("cuda: GQA-KV-dpos out alloc failed")
	}
	if rc := C.cu_gqa_out(scores, v.ptr, out, C.int(seqQ), C.int(seqKV), C.int(qHeads), C.int(kvHeads), C.int(hd)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: GQA-KV-dpos out failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: seqQ, cols: wq}, nil
}
