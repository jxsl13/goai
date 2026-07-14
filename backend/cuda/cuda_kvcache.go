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
