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

// Recorder is the CUDA implementation of the llamagpu backend-agnostic recorder: it
// records a transformer decode step's ops (RMSNorm, projections, RoPE, attention,
// SwiGLU, residuals) so ONE Decoder core can drive metal, vulkan and CUDA alike.
//
// Unlike the metal/vulkan command-buffer recorders — which BATCH ops and dispatch
// them at Commit — this one submits EAGERLY: every method enqueues its kernel on the
// shared work stream as it is called (the stream already serialises them in order).
// So Commit is a no-op and Wait/Finish sync the stream (the model the interface doc
// prescribes for "backends without async submit"). The device-side throughput lever
// on CUDA is graph capture, orthogonal to this batching model.
//
// Not safe for concurrent use — one Recorder per goroutine. Free when done.
type Recorder struct {
	scores    unsafe.Pointer // grow-only MHA scores scratch [heads·sq·sk]
	scoresCap int            // its capacity in floats
}

// NewRecorder returns a CUDA recorder (fails if no GPU is present).
func NewRecorder() (*Recorder, error) {
	if C.cu_available() != 1 {
		return nil, fmt.Errorf("cuda: no GPU for Recorder")
	}
	return &Recorder{}, nil
}

// llamagpu unary/binary op selectors (must match llamagpu/decoder.go's constants,
// which in turn match the metal/vulkan kernel switch tables).
const (
	recUnarySiLU    = 6
	recUnaryGELU    = 9
	recUnaryReLU2   = 10 // squared ReLU (Nemotron); cuda-only, so no metal/vulkan mapping needed
	recBinaryAdd    = 0
	recBinaryMul    = 2
	recBinarySwiGLU = 6
)

func (rec *Recorder) RMSNorm(x, g, o *DeviceF32, rows, dim int, eps float32) error {
	if x.ptr == nil || g.ptr == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec RMSNorm on a freed handle")
	}
	if eps <= 0 {
		eps = 1e-5
	}
	if rc := C.cu_rmsnorm_f32(x.ptr, o.ptr, g.ptr, C.int(rows), C.int(dim), C.float(eps)); rc != 0 {
		return fmt.Errorf("cuda: rec RMSNorm failed (code %d)", int(rc))
	}
	return nil
}

func (rec *Recorder) LayerNorm(x, g, b, o *DeviceF32, rows, dim int, eps float32) error {
	if x.ptr == nil || g.ptr == nil || b.ptr == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec LayerNorm on a freed handle")
	}
	if eps <= 0 {
		eps = 1e-5
	}
	if rc := C.cu_layernorm_f32(x.ptr, o.ptr, g.ptr, b.ptr, C.int(rows), C.int(dim), C.float(eps)); rc != 0 {
		return fmt.Errorf("cuda: rec LayerNorm failed (code %d)", int(rc))
	}
	return nil
}

func (rec *Recorder) AddBias(x, b, o *DeviceF32, rows, n int) error {
	if x.ptr == nil || b.ptr == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec AddBias on a freed handle")
	}
	if rc := C.cu_addbias_f32(x.ptr, b.ptr, o.ptr, C.int(rows), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: rec AddBias failed (code %d)", int(rc))
	}
	return nil
}

func (rec *Recorder) MatMul(a, b, c *DeviceF32, m, k, n int) error {
	if a.ptr == nil || b.ptr == nil || c.ptr == nil {
		return fmt.Errorf("cuda: rec MatMul on a freed handle")
	}
	if rc := C.cu_matmul_f32_ddd(a.ptr, b.ptr, c.ptr, C.int(m), C.int(k), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: rec MatMul failed (code %d)", int(rc))
	}
	return nil
}

func (rec *Recorder) MatMulAcc(a, b, c *DeviceF32, m, k, n int) error {
	if a.ptr == nil || b.ptr == nil || c.ptr == nil {
		return fmt.Errorf("cuda: rec MatMulAcc on a freed handle")
	}
	if rc := C.cu_matmul_f32_ddd_acc(a.ptr, b.ptr, c.ptr, C.int(m), C.int(k), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: rec MatMulAcc failed (code %d)", int(rc))
	}
	return nil
}

// ropeInto copies q→o when they differ (RoPE writes in place on the band), then
// rotates the `heads` band at column `off` of rows `width` wide.
func (rec *Recorder) ropeInto(q, inv, o *DeviceF32, off, seq, width, heads, hd, pos int, posDiv float32) error {
	if q.ptr == nil || inv.ptr == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec RoPE on a freed handle")
	}
	if o.ptr != q.ptr {
		if rc := C.cu_blit(o.ptr, C.int(0), q.ptr, C.int(0), C.int(seq*width)); rc != 0 {
			return fmt.Errorf("cuda: rec RoPE copy failed (code %d)", int(rc))
		}
	}
	if rc := C.cu_rope_f32_band(o.ptr, inv.ptr, C.int(seq), C.int(width), C.int(off), C.int(heads), C.int(hd), C.int(pos), C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: rec RoPE failed (code %d)", int(rc))
	}
	return nil
}

func (rec *Recorder) RoPE(q, inv, o *DeviceF32, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return rec.ropeInto(q, inv, o, 0, seq, width, heads, hd, pos, posDiv)
}

func (rec *Recorder) RoPEAt(q, inv, o *DeviceF32, off, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return rec.ropeInto(q, inv, o, off, seq, width, heads, hd, pos, posDiv)
}

func (rec *Recorder) RoPEPair(qkv, inv *DeviceF32, seq, stride, headsQ, offQ, headsK, offK, hd, half, pos int, posDiv float32) error {
	if err := rec.ropeInto(qkv, inv, qkv, offQ, seq, stride, headsQ, hd, pos, posDiv); err != nil {
		return fmt.Errorf("cuda: rec RoPEPair q: %w", err)
	}
	if err := rec.ropeInto(qkv, inv, qkv, offK, seq, stride, headsK, hd, pos, posDiv); err != nil {
		return fmt.Errorf("cuda: rec RoPEPair k: %w", err)
	}
	return nil
}

// RoPEPartialPair is the partial-rotary sibling of [Recorder.RoPEPair]: it rotates ONLY the
// first rotaryDim channels of each head in the q band (headsQ heads at offQ) and the k band
// (headsK at offK) of a fused [seq,stride] QKV buffer in place, leaving the head tails and
// the v band untouched — the fused-QKV RoPE for GPT-NeoX/Phi/StableLM. inv is [rotaryDim/2].
func (rec *Recorder) RoPEPartialPair(qkv, inv *DeviceF32, seq, stride, headsQ, offQ, headsK, offK, hd, rotaryDim, pos int, posDiv float32) error {
	if qkv.ptr == nil || inv.ptr == nil {
		return fmt.Errorf("cuda: rec RoPEPartialPair on a freed handle")
	}
	if rc := C.cu_rope_partial_band(qkv.ptr, inv.ptr, C.int(seq), C.int(stride), C.int(offQ), C.int(headsQ), C.int(hd), C.int(rotaryDim), C.int(pos), C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: rec RoPEPartialPair q failed (code %d)", int(rc))
	}
	if rc := C.cu_rope_partial_band(qkv.ptr, inv.ptr, C.int(seq), C.int(stride), C.int(offK), C.int(headsK), C.int(hd), C.int(rotaryDim), C.int(pos), C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: rec RoPEPartialPair k failed (code %d)", int(rc))
	}
	return nil
}

func (rec *Recorder) Blit(src *DeviceF32, srcOff int, dst *DeviceF32, dstOff, n int) error {
	if src.ptr == nil || dst.ptr == nil {
		return fmt.Errorf("cuda: rec Blit on a freed handle")
	}
	if rc := C.cu_blit(dst.ptr, C.int(dstOff), src.ptr, C.int(srcOff), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: rec Blit failed (code %d)", int(rc))
	}
	return nil
}

func (rec *Recorder) Copy2D(src *DeviceF32, srcOff, srcStride int, dst *DeviceF32, dstOff, dstStride, rows, rowFloats int) error {
	if src.ptr == nil || dst.ptr == nil {
		return fmt.Errorf("cuda: rec Copy2D on a freed handle")
	}
	if rc := C.cu_copy2d(dst.ptr, C.int(dstOff), C.int(dstStride), src.ptr, C.int(srcOff), C.int(srcStride), C.int(rows), C.int(rowFloats)); rc != 0 {
		return fmt.Errorf("cuda: rec Copy2D failed (code %d)", int(rc))
	}
	return nil
}

// ensureScores grows the MHA scores scratch to at least n floats (grow-only).
func (rec *Recorder) ensureScores(n int) error {
	if rec.scoresCap >= n && rec.scores != nil {
		return nil
	}
	if rec.scores != nil {
		C.cu_free_f32(rec.scores)
		rec.scores = nil
		rec.scoresCap = 0
	}
	p := C.cu_alloc_f32(C.int(n))
	if p == nil {
		return fmt.Errorf("cuda: rec MHA scores alloc %d failed", n)
	}
	rec.scores = p
	rec.scoresCap = n
	return nil
}

// MHA runs grouped-query attention (heads query heads sharing kvHeads kv heads):
// scores = Q·Kᵀ → scaled causal softmax → scores·V into o. sq new query rows attend
// to sk cached keys/values. Sliding-window (window>0 and < sk) is not yet supported.
func (rec *Recorder) MHA(q, k, v, o *DeviceF32, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec MHA on a freed handle")
	}
	if window > 0 && window < sk {
		return fmt.Errorf("cuda: rec MHA sliding-window (window=%d < sk=%d) not supported", window, sk)
	}
	if err := rec.ensureScores(heads * sq * sk); err != nil {
		return err
	}
	if rc := C.cu_gqa_scores(q.ptr, k.ptr, rec.scores, C.int(sq), C.int(sk), C.int(heads), C.int(kvHeads), C.int(dk), 0); rc != 0 {
		return fmt.Errorf("cuda: rec MHA scores failed (code %d)", int(rc))
	}
	// Causal: query row i (global position sk-sq+i) attends to key j ≤ i+(sk-sq).
	// Non-causal: offset ≥ sk unmasks every key.
	offset := sk - sq
	if causal == 0 {
		offset = sk
	}
	if rc := C.cu_attn_softmax(rec.scores, C.int(heads*sq), C.int(sk), C.float(scale), C.int(offset), C.int(sq)); rc != 0 {
		return fmt.Errorf("cuda: rec MHA softmax failed (code %d)", int(rc))
	}
	if rc := C.cu_gqa_out(rec.scores, v.ptr, o.ptr, C.int(sq), C.int(sk), C.int(heads), C.int(kvHeads), C.int(dk), 0); rc != 0 {
		return fmt.Errorf("cuda: rec MHA out failed (code %d)", int(rc))
	}
	return nil
}

func (rec *Recorder) Unary(x, o *DeviceF32, op int) error {
	if x.ptr == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec Unary on a freed handle")
	}
	n := o.rows * o.cols
	if o.ptr != x.ptr {
		if rc := C.cu_blit(o.ptr, C.int(0), x.ptr, C.int(0), C.int(n)); rc != 0 {
			return fmt.Errorf("cuda: rec Unary copy failed (code %d)", int(rc))
		}
	}
	var rc C.int
	switch op {
	case recUnarySiLU:
		rc = C.cu_silu_f32(o.ptr, C.int(n))
	case recUnaryGELU:
		rc = C.cu_gelu_f32(o.ptr, C.int(n))
	case recUnaryReLU2:
		rc = C.cu_relu2_f32(o.ptr, C.int(n))
	default:
		return fmt.Errorf("cuda: rec Unary op %d unsupported", op)
	}
	if rc != 0 {
		return fmt.Errorf("cuda: rec Unary(%d) failed (code %d)", op, int(rc))
	}
	return nil
}

// Binary computes o = a OP b. The CUDA kernels mutate their first argument in place,
// so o first receives a (copy when o≠a), then the op folds b in.
func (rec *Recorder) Binary(a, b, o *DeviceF32, op int) error {
	if a.ptr == nil || b.ptr == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec Binary on a freed handle")
	}
	n := o.rows * o.cols
	if o.ptr != a.ptr {
		if rc := C.cu_blit(o.ptr, C.int(0), a.ptr, C.int(0), C.int(n)); rc != 0 {
			return fmt.Errorf("cuda: rec Binary copy failed (code %d)", int(rc))
		}
	}
	var rc C.int
	switch op {
	case recBinaryAdd:
		rc = C.cu_add_f32(o.ptr, b.ptr, C.int(n))
	case recBinaryMul:
		rc = C.cu_mul_f32(o.ptr, b.ptr, C.int(n))
	case recBinarySwiGLU:
		rc = C.cu_swiglu_f32(o.ptr, b.ptr, C.int(n)) // o = silu(a)·b
	default:
		return fmt.Errorf("cuda: rec Binary op %d unsupported", op)
	}
	if rc != 0 {
		return fmt.Errorf("cuda: rec Binary(%d) failed (code %d)", op, int(rc))
	}
	return nil
}

// QMatMulResident records o[m, w.n] = x[m, w.k]·dequant(w) — the resident Q8 matmul
// (beta=0, overwrite o). w's transposed int8 [n,k] + per-32-block f32 scales are
// consumed by the warp-per-output GEMV kernel; the flat buffer shapes are ignored in
// favour of the explicit m / w.k / w.n dims (as MatMul does).
func (rec *Recorder) QMatMulResident(x *DeviceF32, w *ResidentBQ8, o *DeviceF32, m int) error {
	if x.ptr == nil || w.q == nil || w.scales == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec QMatMulResident on a freed handle")
	}
	if rc := C.cu_qmatmul_q8(x.ptr, w.q, w.scales, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.int(w.nb), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: rec QMatMulResident failed (code %d)", int(rc))
	}
	return nil
}

// Commit is a no-op: ops are already enqueued on the stream as recorded.
func (rec *Recorder) Commit() error { return nil }

// Wait blocks until every enqueued op has completed (stream sync).
func (rec *Recorder) Wait() error {
	if rc := C.cu_graph_sync(); rc != 0 {
		return fmt.Errorf("cuda: rec Wait failed (code %d)", int(rc))
	}
	return nil
}

// Finish is Commit+Wait (eager submit ⇒ just a sync).
func (rec *Recorder) Finish() error { return rec.Wait() }

// Free releases the recorder's scratch.
func (rec *Recorder) Free() {
	if rec.scores != nil {
		C.cu_free_f32(rec.scores)
		rec.scores = nil
		rec.scoresCap = 0
	}
}
