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

// PrefillKV converts m already-RoPE'd f32 K/V rows ([m,wkv]) to f16 at rows 0..m-1 — the batched f16
// prefill (f16 twin of KVCacheI8.PrefillKV); same round-to-nearest f16 conversion as AppendDpos.
func (c *KVCacheF16) PrefillKV(k, v *DeviceF32, m int) error {
	if k.cols != c.wkv || v.cols != c.wkv || k.rows != m || v.rows != m {
		return fmt.Errorf("cuda: KVCacheF16 PrefillKV needs [%d,%d] k/v, got k%v v%v", m, c.wkv, k.shape(), v.shape())
	}
	n := C.long(m * c.wkv)
	if rc := C.cu_cvt_f32_to_f16(c.dK, k.ptr, n); rc != 0 {
		return fmt.Errorf("cuda: KVCacheF16 PrefillKV K failed (code %d)", int(rc))
	}
	if rc := C.cu_cvt_f32_to_f16(c.dV, v.ptr, n); rc != 0 {
		return fmt.Errorf("cuda: KVCacheF16 PrefillKV V failed (code %d)", int(rc))
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

// KVCacheI8 is KVCache with int8 storage + a per-(token,head) f32 scale — a QUARTER of the f32
// bytes (half of f16), the flash decode kernel's bandwidth currency, at the cost of int8 rounding.
// K/V rows are symmetric-quantized per head on append (scale = max|·|/127); the flash kernel
// dequantizes tiles (int8·scale) in shared memory, mirroring the f16 path's h2f. Per-head scales
// (one per head per token) keep accuracy when heads differ in magnitude and are graph-capturable.
// Decode-only, via the int8 flash attention path (added alongside this type).
type KVCacheI8 struct {
	dK, dV      unsafe.Pointer // int8 [maxSeq*kvHeads*hd]
	sK, sV      unsafe.Pointer // f32  [maxSeq*kvHeads] per-(token,head) scales
	maxSeq, wkv int
	kvHeads, hd int
}

// NewKVCacheI8 allocates int8 K/V + f32 per-head scale buffers for maxSeq tokens of kvHeads·hd width.
func NewKVCacheI8(maxSeq, kvHeads, hd int) (*KVCacheI8, error) {
	if maxSeq <= 0 || kvHeads <= 0 || hd <= 0 || hd > 128 {
		return nil, fmt.Errorf("cuda: KVCacheI8 bad dims maxSeq=%d kvHeads=%d hd=%d", maxSeq, kvHeads, hd)
	}
	wkv := kvHeads * hd
	c := &KVCacheI8{maxSeq: maxSeq, wkv: wkv, kvHeads: kvHeads, hd: hd}
	c.dK = C.cu_alloc_i8(C.int(maxSeq * wkv))
	c.dV = C.cu_alloc_i8(C.int(maxSeq * wkv))
	c.sK = C.cu_alloc_f32(C.int(maxSeq * kvHeads))
	c.sV = C.cu_alloc_f32(C.int(maxSeq * kvHeads))
	if c.dK == nil || c.dV == nil || c.sK == nil || c.sV == nil {
		c.Free()
		return nil, fmt.Errorf("cuda: KVCacheI8 alloc failed")
	}
	return c, nil
}

// AppendDpos quantizes one token's k and v rows ([1,wkv] f32, already RoPE'd) to int8 with per-head
// scales and writes them at row *pos — the graph-capturable int8 append.
func (c *KVCacheI8) AppendDpos(k, v *DeviceF32, pos *DevicePos) error {
	if k.cols != c.wkv || v.cols != c.wkv || k.rows != 1 || v.rows != 1 {
		return fmt.Errorf("cuda: KVCacheI8 AppendDpos needs [1,%d] k/v, got k%v v%v", c.wkv, k.shape(), v.shape())
	}
	if rc := C.cu_append_dpos_i8(c.dK, c.sK, k.ptr, pos.ptr, C.int(c.kvHeads), C.int(c.hd)); rc != 0 {
		return fmt.Errorf("cuda: KVCacheI8 AppendDpos K failed (code %d)", int(rc))
	}
	if rc := C.cu_append_dpos_i8(c.dV, c.sV, v.ptr, pos.ptr, C.int(c.kvHeads), C.int(c.hd)); rc != 0 {
		return fmt.Errorf("cuda: KVCacheI8 AppendDpos V failed (code %d)", int(rc))
	}
	return nil
}

// Free releases all four buffers.
func (c *KVCacheI8) Free() {
	if c.dK != nil {
		C.cu_free_f32(c.dK)
	}
	if c.dV != nil {
		C.cu_free_f32(c.dV)
	}
	if c.sK != nil {
		C.cu_free_f32(c.sK)
	}
	if c.sV != nil {
		C.cu_free_f32(c.sV)
	}
	c.dK, c.dV, c.sK, c.sV = nil, nil, nil, nil
}

// downloadKForTest reads the int8 K buffer and its scales back to host and returns the dequantized
// K rows (row-major [maxSeq*wkv]) — used only by the roundtrip parity test.
func (c *KVCacheI8) downloadKForTest() ([]float32, error) {
	q := make([]int8, c.maxSeq*c.wkv)
	sc := make([]float32, c.maxSeq*c.kvHeads)
	if rc := C.cu_download_i8(c.dK, (*C.schar)(unsafe.Pointer(&q[0])), C.int(len(q))); rc != 0 {
		return nil, fmt.Errorf("cuda: download i8 K failed (%d)", int(rc))
	}
	if rc := C.cu_download_f32(c.sK, (*C.float)(unsafe.Pointer(&sc[0])), C.int(len(sc))); rc != 0 {
		return nil, fmt.Errorf("cuda: download K scale failed (%d)", int(rc))
	}
	out := make([]float32, c.maxSeq*c.wkv)
	for tok := 0; tok < c.maxSeq; tok++ {
		for h := 0; h < c.kvHeads; h++ {
			s := sc[tok*c.kvHeads+h]
			for d := 0; d < c.hd; d++ {
				idx := tok*c.wkv + h*c.hd + d
				out[idx] = float32(q[idx]) * s
			}
		}
	}
	return out, nil
}

// GroupedQueryAttentionKVI8DposFlashInto is the flash decode attention over an int8 cache: identical
// organization to the f16 path but K/V global reads are a quarter of f32 (half of f16) and dequant
// (int8·per-token-scale) happens in shared. q is [1, qHeads·hd]; out [1, qHeads·hd]. hd ≤ 128, group ≤ 8.
func GroupedQueryAttentionKVI8DposFlashInto(q *DeviceF32, c *KVCacheI8, qHeads, kvHeads int, off *DevicePos, out *DeviceF32) error {
	if q.ptr == nil || c.dK == nil || out.ptr == nil {
		return fmt.Errorf("cuda: GQA-i8-flash on a freed handle")
	}
	if q.rows != 1 {
		return fmt.Errorf("cuda: GQA-i8-flash needs seqQ==1 (decode), got %d", q.rows)
	}
	wq := q.cols
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || wq%qHeads != 0 {
		return fmt.Errorf("cuda: GQA-i8-flash bad head config")
	}
	hd := wq / qHeads
	if kvHeads*hd != c.wkv {
		return fmt.Errorf("cuda: GQA-i8-flash cache width %d, want %d", c.wkv, kvHeads*hd)
	}
	if out.rows != 1 || out.cols != wq {
		return fmt.Errorf("cuda: GQA-i8-flash out shape [%d,%d], want [1,%d]", out.rows, out.cols, wq)
	}
	if rc := C.cu_gqa_flash_i8_dpos(q.ptr, c.dK, c.dV, c.sK, c.sV, out.ptr,
		C.int(c.maxSeq), C.int(qHeads), C.int(kvHeads), C.int(hd),
		C.float(1/math.Sqrt(float64(hd))), off.ptr); rc != 0 {
		return fmt.Errorf("cuda: GQA-i8-flash failed (code %d)", int(rc))
	}
	return nil
}

// PrefillKV quantizes m tokens of K and V ([m, wkv] f32, RoPE'd) into the int8 cache at rows
// 0..m-1 — the batched prefill counterpart of AppendDpos, replacing the f32 cache's raw K/V blit.
func (c *KVCacheI8) PrefillKV(k, v *DeviceF32, m int) error {
	if k.cols != c.wkv || v.cols != c.wkv || k.rows != m || v.rows != m {
		return fmt.Errorf("cuda: KVCacheI8 PrefillKV needs [%d,%d] k/v, got k%v v%v", m, c.wkv, k.shape(), v.shape())
	}
	if rc := C.cu_quant_batch_i8(c.dK, c.sK, k.ptr, C.int(m), C.int(c.kvHeads), C.int(c.hd)); rc != 0 {
		return fmt.Errorf("cuda: KVCacheI8 PrefillKV K failed (code %d)", int(rc))
	}
	if rc := C.cu_quant_batch_i8(c.dV, c.sV, v.ptr, C.int(m), C.int(c.kvHeads), C.int(c.hd)); rc != 0 {
		return fmt.Errorf("cuda: KVCacheI8 PrefillKV V failed (code %d)", int(rc))
	}
	return nil
}
