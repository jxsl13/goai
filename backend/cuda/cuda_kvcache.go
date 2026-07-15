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

// KVCache holds one attention layer's resident key/value history for
// autoregressive decode. K and V are contiguous [maxSeq, wkv] device buffers
// filled row-by-row (or in a prefill block); each new token appends its keys and
// values and then attends the whole accumulated cache via GroupedQueryAttentionKV
// — O(N) work per token instead of re-forwarding the entire prefix (O(N²)).
//
// Keys/values must be stored POST-RoPE (rotated at their absolute position), so
// the cached rows never need re-rotating on later steps.
type KVCache struct {
	dK, dV         unsafe.Pointer
	maxSeq, wkv, n int
}

// NewKVCache allocates the K and V buffers for up to maxSeq tokens of width wkv
// (= kvHeads·headDim). Free when the sequence is done.
func NewKVCache(maxSeq, wkv int) (*KVCache, error) {
	if maxSeq <= 0 || wkv <= 0 {
		return nil, fmt.Errorf("cuda: KVCache bad dims maxSeq=%d wkv=%d", maxSeq, wkv)
	}
	dK := C.cu_alloc_f32(C.int(maxSeq * wkv))
	if dK == nil {
		return nil, fmt.Errorf("cuda: KVCache alloc K failed")
	}
	dV := C.cu_alloc_f32(C.int(maxSeq * wkv))
	if dV == nil {
		C.cu_free_f32(dK)
		return nil, fmt.Errorf("cuda: KVCache alloc V failed")
	}
	return &KVCache{dK: dK, dV: dV, maxSeq: maxSeq, wkv: wkv}, nil
}

// Append copies the k and v rows ([nNew, wkv] each, already RoPE'd) into the
// cache just past the stored rows and advances the length by nNew.
func (c *KVCache) Append(k, v *DeviceF32) error {
	if c.dK == nil {
		return fmt.Errorf("cuda: KVCache append on a freed cache")
	}
	if k.ptr == nil || v.ptr == nil {
		return fmt.Errorf("cuda: KVCache append on a freed handle")
	}
	if k.cols != c.wkv || v.cols != c.wkv || k.rows != v.rows {
		return fmt.Errorf("cuda: KVCache append shape k%v v%v (want cols %d, equal rows)", k.shape(), v.shape(), c.wkv)
	}
	nNew := k.rows
	if c.n+nNew > c.maxSeq {
		return fmt.Errorf("cuda: KVCache overflow (%d+%d > %d)", c.n, nNew, c.maxSeq)
	}
	off := c.n * c.wkv
	if rc := C.cu_copy_rows(c.dK, k.ptr, C.int(off), C.int(nNew*c.wkv)); rc != 0 {
		return fmt.Errorf("cuda: KVCache append K failed (code %d)", int(rc))
	}
	if rc := C.cu_copy_rows(c.dV, v.ptr, C.int(off), C.int(nNew*c.wkv)); rc != 0 {
		return fmt.Errorf("cuda: KVCache append V failed (code %d)", int(rc))
	}
	c.n += nNew
	return nil
}

// K and V return NON-owning views of the first Len() cached rows, for use as the
// key/value operands of GroupedQueryAttentionKV. Do NOT Free the returned views —
// they alias the cache buffers, which Free (below) owns.
func (c *KVCache) K() *DeviceF32 { return &DeviceF32{ptr: c.dK, rows: c.n, cols: c.wkv} }
func (c *KVCache) V() *DeviceF32 { return &DeviceF32{ptr: c.dV, rows: c.n, cols: c.wkv} }

// Len is the number of tokens currently cached.
func (c *KVCache) Len() int { return c.n }

// SetLen sets the logical length directly — used with AppendDpos, where the
// device kernel writes at a device-side position and the host tracks length
// separately (the fixed-size graph decode path).
func (c *KVCache) SetLen(n int) { c.n = n }

// Free releases the K/V buffers.
func (c *KVCache) Free() {
	if c.dK != nil {
		C.cu_free_f32(c.dK)
		c.dK = nil
	}
	if c.dV != nil {
		C.cu_free_f32(c.dV)
		c.dV = nil
	}
	c.n = 0
}

// KVCacheF16 is KVCache with f16 (u16) storage — half the K/V bytes, which is
// the flash decode kernel's bandwidth currency (Tw52: attention is K/V-read-
// bound by construction). Rows are converted f32→f16 (round-to-nearest-even) on
// append; the flash kernel converts tiles back to f32 in shared memory, so the
// compute precision matches the f32-cache kernel given the rounded values.
// llama.cpp stores its KV in f16 by default — this also makes cross-engine
// comparisons same-format. Decode-only: usable via the flash attention path
// (GroupedQueryAttentionKVF16DposFlashInto), not the cuBLAS chain (Sgemm needs
// f32 operands).
type KVCacheF16 struct {
	dK, dV      unsafe.Pointer
	maxSeq, wkv int
}

// NewKVCacheF16 allocates u16 K/V buffers for up to maxSeq tokens of width wkv.
func NewKVCacheF16(maxSeq, wkv int) (*KVCacheF16, error) {
	if maxSeq <= 0 || wkv <= 0 {
		return nil, fmt.Errorf("cuda: KVCacheF16 bad dims maxSeq=%d wkv=%d", maxSeq, wkv)
	}
	dK := C.cu_alloc_u16(C.int(maxSeq * wkv))
	if dK == nil {
		return nil, fmt.Errorf("cuda: KVCacheF16 alloc K failed")
	}
	dV := C.cu_alloc_u16(C.int(maxSeq * wkv))
	if dV == nil {
		C.cu_free_f32(dK)
		return nil, fmt.Errorf("cuda: KVCacheF16 alloc V failed")
	}
	return &KVCacheF16{dK: dK, dV: dV, maxSeq: maxSeq, wkv: wkv}, nil
}

// AppendDpos converts one token's k and v rows ([1,wkv] f32, already RoPE'd) to
// f16 and writes them at row *pos — the graph-capturable f16 append.
func (c *KVCacheF16) AppendDpos(k, v *DeviceF32, pos *DevicePos) error {
	if k.cols != c.wkv || v.cols != c.wkv || k.rows != 1 || v.rows != 1 {
		return fmt.Errorf("cuda: KVCacheF16 AppendDpos needs [1,%d] k/v, got k%v v%v", c.wkv, k.shape(), v.shape())
	}
	if rc := C.cu_append_dpos_f16(c.dK, k.ptr, pos.ptr, C.int(c.wkv)); rc != 0 {
		return fmt.Errorf("cuda: KVCacheF16 AppendDpos K failed (code %d)", int(rc))
	}
	if rc := C.cu_append_dpos_f16(c.dV, v.ptr, pos.ptr, C.int(c.wkv)); rc != 0 {
		return fmt.Errorf("cuda: KVCacheF16 AppendDpos V failed (code %d)", int(rc))
	}
	return nil
}

// ZeroCache zeros both buffers (f16 zero == all-zero bytes).
func (c *KVCacheF16) ZeroCache() error {
	if rc := C.cu_zero_u16(c.dK, C.int(c.maxSeq*c.wkv)); rc != 0 {
		return fmt.Errorf("cuda: KVCacheF16 zero K failed (code %d)", int(rc))
	}
	if rc := C.cu_zero_u16(c.dV, C.int(c.maxSeq*c.wkv)); rc != 0 {
		return fmt.Errorf("cuda: KVCacheF16 zero V failed (code %d)", int(rc))
	}
	return nil
}

// Free releases both buffers.
func (c *KVCacheF16) Free() {
	if c.dK != nil {
		C.cu_free_f32(c.dK)
		C.cu_free_f32(c.dV)
		c.dK, c.dV = nil, nil
	}
}

// GroupedQueryAttentionKVF16DposFlashInto is the flash decode attention over an
// f16 cache: identical organization to GroupedQueryAttentionKVDposFlashInto but
// K/V global reads are half the bytes (tiles converted to f32 in shared during
// staging). q is [1, qHeads·hd]; out [1, qHeads·hd]. Needs hd ≤ 128, group ≤ 8.
func GroupedQueryAttentionKVF16DposFlashInto(q *DeviceF32, c *KVCacheF16, qHeads, kvHeads int, off *DevicePos, out *DeviceF32) error {
	if q.ptr == nil || c.dK == nil || out.ptr == nil {
		return fmt.Errorf("cuda: GQA-f16-flash on a freed handle")
	}
	seqQ, wq := q.rows, q.cols
	if seqQ != 1 {
		return fmt.Errorf("cuda: GQA-f16-flash needs seqQ==1 (decode), got %d", seqQ)
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || wq%qHeads != 0 {
		return fmt.Errorf("cuda: GQA-f16-flash bad head config")
	}
	hd := wq / qHeads
	if kvHeads*hd != c.wkv {
		return fmt.Errorf("cuda: GQA-f16-flash cache width %d, want %d", c.wkv, kvHeads*hd)
	}
	if out.rows != 1 || out.cols != wq {
		return fmt.Errorf("cuda: GQA-f16-flash out shape [%d,%d], want [1,%d]", out.rows, out.cols, wq)
	}
	if rc := C.cu_gqa_flash_f16_dpos(q.ptr, c.dK, c.dV, out.ptr,
		C.int(c.maxSeq), C.int(qHeads), C.int(kvHeads), C.int(hd),
		C.float(1/math.Sqrt(float64(hd))), off.ptr); rc != 0 {
		return fmt.Errorf("cuda: GQA-f16-flash failed (code %d)", int(rc))
	}
	return nil
}
