//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"math"
)

// NewDeviceF32 allocates an uninitialized resident [rows, cols] buffer — a
// persistent scratch buffer reused across decode steps (fixed pointer, the
// precondition for capturing the decode into a CUDA graph). Free when done.
func NewDeviceF32(rows, cols int) (*DeviceF32, error) {
	p := C.cu_alloc_f32(C.int(rows * cols))
	if p == nil {
		return nil, fmt.Errorf("cuda: NewDeviceF32 alloc [%d,%d] failed", rows, cols)
	}
	return &DeviceF32{ptr: p, rows: rows, cols: cols}, nil
}

// Argmax returns the index of the maximum element (greedy next token) computed on
// device — only the 4-byte index is downloaded, not the whole [1,vocab] logit
// vector. Returns -1 on error.
func (d *DeviceF32) Argmax() int {
	return int(C.cu_argmax_f32(d.ptr, C.int(d.rows*d.cols)))
}

// LayerNormInto writes LayerNorm(d)·gamma + beta into out over the last axis
// (torch LayerNorm), leaving d unchanged. gamma/beta are resident [cols] vectors.
func (d *DeviceF32) LayerNormInto(gamma, beta *ResidentVec, eps float32, out *DeviceF32) error {
	if d.ptr == nil || gamma.ptr == nil || beta.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: LayerNormInto on a freed handle")
	}
	if gamma.n != d.cols || beta.n != d.cols || out.rows != d.rows || out.cols != d.cols {
		return fmt.Errorf("cuda: LayerNormInto shape mismatch")
	}
	if eps <= 0 {
		eps = 1e-5
	}
	if rc := C.cu_layernorm_f32(d.ptr, out.ptr, gamma.ptr, beta.ptr, C.int(d.rows), C.int(d.cols), C.float(eps)); rc != 0 {
		return fmt.Errorf("cuda: layernorm failed (code %d)", int(rc))
	}
	return nil
}

// AddBias adds a resident [cols] bias vector to every row of d, in place
// (row-broadcast) — the QKV-projection bias for Qwen and GPT linear biases.
func (d *DeviceF32) AddBias(bias *ResidentVec) error {
	if d.ptr == nil || bias.ptr == nil {
		return fmt.Errorf("cuda: AddBias on a freed handle")
	}
	if bias.n != d.cols {
		return fmt.Errorf("cuda: AddBias bias [%d] != cols %d", bias.n, d.cols)
	}
	if rc := C.cu_addbias_f32(d.ptr, bias.ptr, d.ptr, C.int(d.rows), C.int(d.cols)); rc != 0 {
		return fmt.Errorf("cuda: addbias failed (code %d)", int(rc))
	}
	return nil
}

// CopyFrom overwrites this buffer's contents with src's (device→device, same
// size). Used to reset a captured graph's fixed input buffer between replays.
func (d *DeviceF32) CopyFrom(src *DeviceF32) error {
	if d.ptr == nil || src.ptr == nil {
		return fmt.Errorf("cuda: CopyFrom on a freed handle")
	}
	if d.rows*d.cols != src.rows*src.cols {
		return fmt.Errorf("cuda: CopyFrom size mismatch %d vs %d", d.rows*d.cols, src.rows*src.cols)
	}
	if rc := C.cu_copy_rows(d.ptr, src.ptr, C.int(0), C.int(d.rows*d.cols)); rc != 0 {
		return fmt.Errorf("cuda: CopyFrom failed (code %d)", int(rc))
	}
	return nil
}

// Blit copies n floats from src[srcOff:] into d[dstOff:] (contiguous device→device)
// — the llamagpu recorder Blit for offset band moves.
func (d *DeviceF32) Blit(dstOff int, src *DeviceF32, srcOff, n int) error {
	if d.ptr == nil || src.ptr == nil {
		return fmt.Errorf("cuda: Blit on a freed handle")
	}
	if rc := C.cu_blit(d.ptr, C.int(dstOff), src.ptr, C.int(srcOff), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: blit failed (code %d)", int(rc))
	}
	return nil
}

// Copy2D copies a rows×rowFloats sub-matrix from src into d, each with its own row
// stride/offset — the llamagpu recorder Copy2D (fused-QKV band extraction).
func (d *DeviceF32) Copy2D(dstOff, dstStride int, src *DeviceF32, srcOff, srcStride, rows, rowFloats int) error {
	if d.ptr == nil || src.ptr == nil {
		return fmt.Errorf("cuda: Copy2D on a freed handle")
	}
	if rc := C.cu_copy2d(d.ptr, C.int(dstOff), C.int(dstStride), src.ptr, C.int(srcOff), C.int(srcStride), C.int(rows), C.int(rowFloats)); rc != 0 {
		return fmt.Errorf("cuda: copy2d failed (code %d)", int(rc))
	}
	return nil
}

// Zero sets the buffer to all zeros (on the stream).
func (d *DeviceF32) Zero() error {
	if rc := C.cu_zero_f32(d.ptr, C.int(d.rows*d.cols)); rc != 0 {
		return fmt.Errorf("cuda: Zero failed (code %d)", int(rc))
	}
	return nil
}

// MatMulInto computes out = a·B into the caller's fixed out buffer (no allocation)
// — the persistent-buffer form of MatMulDevice for the graph decode path.
func (r *ResidentB) MatMulInto(a, out *DeviceF32) error {
	if r.ptr == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: MatMulInto on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: MatMulInto shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_matmul_f32_ddd(a.ptr, r.ptr, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n)); rc != 0 {
		return fmt.Errorf("cuda: MatMulInto failed (code %d)", int(rc))
	}
	return nil
}

// RMSNormInto writes RMSNorm(d)·gamma into out (d unchanged) — persistent-buffer
// out-of-place RMSNorm.
func (d *DeviceF32) RMSNormInto(gamma *ResidentVec, eps float32, out *DeviceF32) error {
	if d.ptr == nil || gamma.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: RMSNormInto on a freed handle")
	}
	if gamma.n != d.cols || out.rows != d.rows || out.cols != d.cols {
		return fmt.Errorf("cuda: RMSNormInto shape mismatch")
	}
	if eps <= 0 {
		eps = 1e-5
	}
	if rc := C.cu_rmsnorm_f32(d.ptr, out.ptr, gamma.ptr, C.int(d.rows), C.int(d.cols), C.float(eps)); rc != 0 {
		return fmt.Errorf("cuda: RMSNormInto failed (code %d)", int(rc))
	}
	return nil
}

// EmbedInto gathers the embedding rows for ids into the caller's out buffer.
func (r *ResidentB) EmbedInto(ids []int32, out *DeviceF32) error {
	if r.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: EmbedInto on a freed handle")
	}
	if len(ids) == 0 || out.rows != len(ids) || out.cols != r.n {
		return fmt.Errorf("cuda: EmbedInto shape: ids %d, out [%d,%d], n %d", len(ids), out.rows, out.cols, r.n)
	}
	dIds := C.cu_upload_i32((*C.int)(&ids[0]), C.int(len(ids)))
	if dIds == nil {
		return fmt.Errorf("cuda: EmbedInto id upload failed")
	}
	defer C.cu_free_f32(dIds)
	if rc := C.cu_embed_f32(r.ptr, dIds, out.ptr, C.int(len(ids)), C.int(r.n)); rc != 0 {
		return fmt.Errorf("cuda: EmbedInto failed (code %d)", int(rc))
	}
	return nil
}

// GroupedQueryAttentionKVDposInto is GroupedQueryAttentionKVDpos writing into
// caller-provided scores [qHeads·seqQ·seqKV] and out [seqQ·wq] buffers (no
// allocation) — the fixed-size, fixed-buffer attention for the graph decode.
func GroupedQueryAttentionKVDposInto(q, k, v *DeviceF32, qHeads, kvHeads int, off *DevicePos, scores, out *DeviceF32) error {
	seqQ, wq := q.rows, q.cols
	hd := wq / qHeads
	seqKV := k.rows
	if rc := C.cu_gqa_scores(q.ptr, k.ptr, scores.ptr, C.int(seqQ), C.int(seqKV), C.int(qHeads), C.int(kvHeads), C.int(hd)); rc != 0 {
		return fmt.Errorf("cuda: GQA-dpos-into scores failed (code %d)", int(rc))
	}
	if rc := C.cu_attn_softmax_dpos(scores.ptr, C.int(qHeads*seqQ), C.int(seqKV), C.float(1/math.Sqrt(float64(hd))), off.ptr, C.int(seqQ)); rc != 0 {
		return fmt.Errorf("cuda: GQA-dpos-into softmax failed (code %d)", int(rc))
	}
	if rc := C.cu_gqa_out(scores.ptr, v.ptr, out.ptr, C.int(seqQ), C.int(seqKV), C.int(qHeads), C.int(kvHeads), C.int(hd)); rc != 0 {
		return fmt.Errorf("cuda: GQA-dpos-into out failed (code %d)", int(rc))
	}
	return nil
}

// FullView returns a [rows, wkv] view of the whole cache buffer (for fixed-size
// attention over the padded cache; masked positions past the length contribute
// nothing). Do NOT Free the view.
func (c *KVCache) FullView() (kk, vv *DeviceF32) {
	return &DeviceF32{ptr: c.dK, rows: c.maxSeq, cols: c.wkv},
		&DeviceF32{ptr: c.dV, rows: c.maxSeq, cols: c.wkv}
}

// ZeroCache zeros both K and V buffers (fixed-size init).
func (c *KVCache) ZeroCache() error {
	if rc := C.cu_zero_f32(c.dK, C.int(c.maxSeq*c.wkv)); rc != 0 {
		return fmt.Errorf("cuda: ZeroCache K failed (code %d)", int(rc))
	}
	if rc := C.cu_zero_f32(c.dV, C.int(c.maxSeq*c.wkv)); rc != 0 {
		return fmt.Errorf("cuda: ZeroCache V failed (code %d)", int(rc))
	}
	return nil
}
