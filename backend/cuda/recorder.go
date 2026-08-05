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
	recUnarySiLU     = 6
	recUnaryGELU     = 9
	recUnaryReLU2    = 10 // squared ReLU (Nemotron); cuda-only
	recUnarySigmoid  = 11 // plain sigmoid (Qwen2-MoE shared-expert gate); cuda-only
	recUnarySoftplus = 12 // softplus (Mamba Δ); cuda-only
	recUnaryReLU     = 13 // plain ReLU max(x,0) (T5 v1.0 FFN); cuda-only
	recBinaryAdd     = 0
	recBinaryMul     = 2
	recBinarySwiGLU  = 6
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
	// Fused tensor-core flash-prefill fast path: causal-SQUARE prefill (sq==sk ⇒ the triple's
	// offset sk-sq is 0, exactly the wmma kernel's built-in causal mask: query row i attends
	// keys ≤ i), hd==64, seq a multiple of 16 within the shared-mem bound (wmmaMaxSeq). QKᵀ →
	// warp-parallel scaled softmax → PV in ONE kernel with NO materialized HBM scores — 2.69-3.0×
	// the cu_gqa_scores/softmax/gqa_out triple below (which writes the full [heads,sq,sk] scores
	// to HBM via un-accelerated f32 batched GEMMs). This is the SAME kernel gqaCore already routes
	// this exact shape to (cuda.go), so its f16-accum numerics vs the triple's f32 are already
	// tol-validated (the incumbent llama.cpp/vLLM prefill-attention precision, PERF-INCUMBENT-
	// TOLERANCE-001). Safe on the eager Recorder: this path fires only for square prefill, never
	// decode (sq=1 fails sq==sk and sq%16==0), and prefill is not graph-captured, so the kernel's
	// internal cudaMallocAsync f16 scratch is legal. Any miss (hd≠64, non-causal, sq≠sk, odd/
	// oversized seq, GQA misconfig, or a kernel error) falls through to the triple.
	if causal != 0 && dk == 64 && sq == sk && sq%16 == 0 && sq <= wmmaMaxSeq &&
		kvHeads > 0 && heads%kvHeads == 0 {
		if rc := C.cu_wmma_attn_gqa(unsafe.Pointer(&wmmaAttnGqaFatbin[0]), C.int(len(wmmaAttnGqaFatbin)),
			q.ptr, k.ptr, v.ptr, o.ptr, C.int(sq), C.int(heads), C.int(kvHeads), C.int(dk), C.float(scale)); rc == 0 {
			return nil
		}
		// non-zero rc: fall through to the always-correct triple below.
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

// MoEGate writes Mixtral-style top-k routing weights: for each of `rows` tokens, given `e` raw router
// logits, weights[e] = softmax over the top-`k` logits (0 for unselected experts) — the renormalized
// combine weights. logits and weights are [rows·e] device buffers and must be distinct.
func (rec *Recorder) MoEGate(logits, weights *DeviceF32, rows, e, k, raw int, scale float32) error {
	if logits.ptr == nil || weights.ptr == nil {
		return fmt.Errorf("cuda: rec MoEGate on a freed handle")
	}
	if rc := C.cu_moe_gate(logits.ptr, weights.ptr, C.int(rows), C.int(e), C.int(k), C.int(raw), C.float(scale)); rc != 0 {
		return fmt.Errorf("cuda: rec MoEGate failed (code %d)", int(rc))
	}
	return nil
}

// Conv1DStep advances one decode step of Mamba's causal depthwise conv: reads this channel's state
// (its last K-1 inputs) plus the new x, writes out[D], and shifts the new input into the state. x/b/
// out are [D], w is [D,K], state is [D,K-1] (starts zeroed for the causal left-pad).
func (rec *Recorder) Conv1DStep(x, w, b, state, out *DeviceF32, d, k int) error {
	if x.ptr == nil || w.ptr == nil || b.ptr == nil || state.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: rec Conv1DStep on a freed handle")
	}
	if rc := C.cu_conv1d_step(x.ptr, w.ptr, b.ptr, state.ptr, out.ptr, C.int(d), C.int(k)); rc != 0 {
		return fmt.Errorf("cuda: rec Conv1DStep failed (code %d)", int(rc))
	}
	return nil
}

// WKVStep advances one timestep of the RWKV-4 WKV recurrence for decode: the stabilized linear-
// attention scan. k/v/w/u/out are [D] (w = per-channel decay exp(WLog), u = current-token bonus);
// the running state aa/bb/pp [D] (numerator/denominator/max-exponent) is updated in place and must
// start aa=bb=0, pp=−1e38 for a fresh sequence. No KV cache — RWKV's O(1) recurrent inference.
func (rec *Recorder) WKVStep(k, v, w, u, aa, bb, pp, out *DeviceF32, d int) error {
	if k.ptr == nil || v.ptr == nil || w.ptr == nil || u.ptr == nil || aa.ptr == nil || bb.ptr == nil || pp.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: rec WKVStep on a freed handle")
	}
	if rc := C.cu_wkv_step(k.ptr, v.ptr, w.ptr, u.ptr, aa.ptr, bb.ptr, pp.ptr, out.ptr, C.int(d)); rc != 0 {
		return fmt.Errorf("cuda: rec WKVStep failed (code %d)", int(rc))
	}
	return nil
}

// SSMStep advances one timestep of the Mamba selective-scan recurrence for decode: updates the
// per-channel state h[D,N] in place and writes y[D]. u/delta/dskip/y are [D]; A is [D,N]; B/C are
// [N] (this token's input-dependent matrices, shared across channels). The state persists across
// calls (the caller keeps h between steps).
func (rec *Recorder) SSMStep(u, delta, a, b, c, dskip, h, y *DeviceF32, d, n int) error {
	if u.ptr == nil || delta.ptr == nil || a.ptr == nil || b.ptr == nil || c.ptr == nil || dskip.ptr == nil || h.ptr == nil || y.ptr == nil {
		return fmt.Errorf("cuda: rec SSMStep on a freed handle")
	}
	if rc := C.cu_ssm_step(u.ptr, delta.ptr, a.ptr, b.ptr, c.ptr, dskip.ptr, h.ptr, y.ptr, C.int(d), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: rec SSMStep failed (code %d)", int(rc))
	}
	return nil
}

// SSDStep advances one timestep of the Mamba-2 SSD scan for decode: H heads with a SCALAR per-head
// decay a = exp(Δ[h]·A[h]) and B/C shared across a group (g = h/(H/G)). It updates the per-head state
// state[H,N,P] in place and writes y[H·P]. x/y/dskip are per head·head_dim or per head; A/delta/dskip
// are [H]; B/C are [G·N]. The state persists across calls (the caller keeps it between steps).
func (rec *Recorder) SSDStep(x, delta, a, b, c, dskip, state, y *DeviceF32, heads, headDim, groups, n int) error {
	if x.ptr == nil || delta.ptr == nil || a.ptr == nil || b.ptr == nil || c.ptr == nil || dskip.ptr == nil || state.ptr == nil || y.ptr == nil {
		return fmt.Errorf("cuda: rec SSDStep on a freed handle")
	}
	if rc := C.cu_ssd_step(x.ptr, delta.ptr, a.ptr, b.ptr, c.ptr, dskip.ptr, state.ptr, y.ptr, C.int(heads), C.int(headDim), C.int(groups), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: rec SSDStep failed (code %d)", int(rc))
	}
	return nil
}

// RowAxpy accumulates dst += diag(arow)·src: dst[r,:] += arow[r]·src[r,:], the MoE combine that
// scales an expert's [rows,cols] output by each token's routing weight (arow is a [rows] vector).
func (rec *Recorder) RowAxpy(dst, src, arow *DeviceF32, rows, cols int) error {
	if dst.ptr == nil || src.ptr == nil || arow.ptr == nil {
		return fmt.Errorf("cuda: rec RowAxpy on a freed handle")
	}
	if rc := C.cu_row_axpy(dst.ptr, src.ptr, arow.ptr, C.int(rows), C.int(cols)); rc != 0 {
		return fmt.Errorf("cuda: rec RowAxpy failed (code %d)", int(rc))
	}
	return nil
}

// MHAALiBi is MHA with an ALiBi position bias (MPT): each scaled score gets slopeₕ·(j−qabs) added
// before the mask+softmax, where slopes is a device array of `heads` per-head slopes. No RoPE.
func (rec *Recorder) MHAALiBi(q, k, v, o, slopes *DeviceF32, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil || o.ptr == nil || slopes.ptr == nil {
		return fmt.Errorf("cuda: rec MHAALiBi on a freed handle")
	}
	if window > 0 && window < sk {
		return fmt.Errorf("cuda: rec MHAALiBi sliding-window (window=%d < sk=%d) not supported", window, sk)
	}
	if err := rec.ensureScores(heads * sq * sk); err != nil {
		return err
	}
	if rc := C.cu_gqa_scores(q.ptr, k.ptr, rec.scores, C.int(sq), C.int(sk), C.int(heads), C.int(kvHeads), C.int(dk), 0); rc != 0 {
		return fmt.Errorf("cuda: rec MHAALiBi scores failed (code %d)", int(rc))
	}
	offset := sk - sq
	if causal == 0 {
		offset = sk
	}
	if rc := C.cu_attn_softmax_alibi(rec.scores, C.int(heads*sq), C.int(sk), C.float(scale), C.int(offset), C.int(sq), slopes.ptr); rc != 0 {
		return fmt.Errorf("cuda: rec MHAALiBi softmax failed (code %d)", int(rc))
	}
	if rc := C.cu_gqa_out(rec.scores, v.ptr, o.ptr, C.int(sq), C.int(sk), C.int(heads), C.int(kvHeads), C.int(dk), 0); rc != 0 {
		return fmt.Errorf("cuda: rec MHAALiBi out failed (code %d)", int(rc))
	}
	return nil
}

// MHABias is MHA with a PRECOMPUTED per-head additive bias added to the scaled scores before the
// mask+softmax — the T5 relative-position bias, a learned [heads,sq,sk] table (laid out identically
// to the scores buffer). Unlike ALiBi's computed slope it is an arbitrary per-(head,i,j) value. No
// RoPE (T5 has no rotary/absolute position). causal==0 (T5's bidirectional encoder) unmasks all keys.
func (rec *Recorder) MHABias(q, k, v, o, bias *DeviceF32, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil || o.ptr == nil || bias.ptr == nil {
		return fmt.Errorf("cuda: rec MHABias on a freed handle")
	}
	if window > 0 && window < sk {
		return fmt.Errorf("cuda: rec MHABias sliding-window (window=%d < sk=%d) not supported", window, sk)
	}
	if err := rec.ensureScores(heads * sq * sk); err != nil {
		return err
	}
	if rc := C.cu_gqa_scores(q.ptr, k.ptr, rec.scores, C.int(sq), C.int(sk), C.int(heads), C.int(kvHeads), C.int(dk), 0); rc != 0 {
		return fmt.Errorf("cuda: rec MHABias scores failed (code %d)", int(rc))
	}
	offset := sk - sq
	if causal == 0 {
		offset = sk
	}
	if rc := C.cu_attn_softmax_bias(rec.scores, C.int(heads*sq), C.int(sk), C.float(scale), C.int(offset), C.int(sq), bias.ptr); rc != 0 {
		return fmt.Errorf("cuda: rec MHABias softmax failed (code %d)", int(rc))
	}
	if rc := C.cu_gqa_out(rec.scores, v.ptr, o.ptr, C.int(sq), C.int(sk), C.int(heads), C.int(kvHeads), C.int(dk), 0); rc != 0 {
		return fmt.Errorf("cuda: rec MHABias out failed (code %d)", int(rc))
	}
	return nil
}

// MHARect is multi-head attention with RECTANGULAR head dims: the query/key share a head width dqk
// (so the scores are Q·Kᵀ over dqk) but the value has a DIFFERENT head width dv, so the output rows
// are heads·dv wide. This is the shape DeepSeek-V2 MLA needs — query/key are concat(nope, pe) of
// width QKNope+QKRope while the value is VHead-wide. No soft-cap; standard scale + causal mask.
func (rec *Recorder) MHARect(q, k, v, o *DeviceF32, sq, sk, heads, kvHeads, dqk, dv, causal, window int, scale float32) error {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec MHARect on a freed handle")
	}
	if window > 0 && window < sk {
		return fmt.Errorf("cuda: rec MHARect sliding-window (window=%d < sk=%d) not supported", window, sk)
	}
	if err := rec.ensureScores(heads * sq * sk); err != nil {
		return err
	}
	if rc := C.cu_gqa_scores(q.ptr, k.ptr, rec.scores, C.int(sq), C.int(sk), C.int(heads), C.int(kvHeads), C.int(dqk), 0); rc != 0 {
		return fmt.Errorf("cuda: rec MHARect scores failed (code %d)", int(rc))
	}
	offset := sk - sq
	if causal == 0 {
		offset = sk
	}
	if rc := C.cu_attn_softmax(rec.scores, C.int(heads*sq), C.int(sk), C.float(scale), C.int(offset), C.int(sq)); rc != 0 {
		return fmt.Errorf("cuda: rec MHARect softmax failed (code %d)", int(rc))
	}
	if rc := C.cu_gqa_out(rec.scores, v.ptr, o.ptr, C.int(sq), C.int(sk), C.int(heads), C.int(kvHeads), C.int(dv), 0); rc != 0 {
		return fmt.Errorf("cuda: rec MHARect out failed (code %d)", int(rc))
	}
	return nil
}

// MHACap is MHA with Gemma 2's attention-logit soft-cap: the scaled scores are passed through
// cap·tanh(·/cap) before the causal mask and softmax. cap must be > 0.
func (rec *Recorder) MHACap(q, k, v, o *DeviceF32, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale, cap float32) error {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec MHACap on a freed handle")
	}
	if window > 0 && window < sk {
		return fmt.Errorf("cuda: rec MHACap sliding-window (window=%d < sk=%d) not supported", window, sk)
	}
	if err := rec.ensureScores(heads * sq * sk); err != nil {
		return err
	}
	if rc := C.cu_gqa_scores(q.ptr, k.ptr, rec.scores, C.int(sq), C.int(sk), C.int(heads), C.int(kvHeads), C.int(dk), 0); rc != 0 {
		return fmt.Errorf("cuda: rec MHACap scores failed (code %d)", int(rc))
	}
	offset := sk - sq
	if causal == 0 {
		offset = sk
	}
	if rc := C.cu_attn_softmax_cap(rec.scores, C.int(heads*sq), C.int(sk), C.float(scale), C.int(offset), C.int(sq), C.float(cap)); rc != 0 {
		return fmt.Errorf("cuda: rec MHACap softmax failed (code %d)", int(rc))
	}
	if rc := C.cu_gqa_out(rec.scores, v.ptr, o.ptr, C.int(sq), C.int(sk), C.int(heads), C.int(kvHeads), C.int(dk), 0); rc != 0 {
		return fmt.Errorf("cuda: rec MHACap out failed (code %d)", int(rc))
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
	case recUnaryReLU:
		rc = C.cu_relu_f32(o.ptr, C.int(n))
	case recUnarySigmoid:
		rc = C.cu_sigmoid_f32(o.ptr, C.int(n))
	case recUnarySoftplus:
		rc = C.cu_softplus_f32(o.ptr, C.int(n))
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
	// PREFILL (M>1): cu_qmatmul_q8 is a GEMV — one warp per output element re-reads the whole [N,K] weight
	// for EVERY output row, so it streams the weight M× (M× the dominant weight bandwidth). When the int8
	// tensor-core MMQ GEMM applies (N%64==0, K%32==0), route there: it reads the weight ONCE across all M
	// rows. ResidentBQ8's q + scales ARE already the MMQ weight format (int8 [N,K] + per-32-block f32
	// amax/127 scales), so no requantization — only the activation is quantized per row. Decode (M=1) keeps
	// the GEMV (optimal — MMQ would pad M to 64). Any error falls back to the always-correct GEMV.
	// Threshold m>=6 (measured crossover, cuda_q8_mmq_crossover_bench_test.go): MMQ pads M to 64
	// so its cost is ~flat while the GEMV streams the weight M×, so MMQ wins once M amortizes the
	// pad. m=6 wins at every tested shape (2048² 1.40×, 2048×5632 1.87×, 4096² 2.3×); m=4 only
	// wins at large n (ties at 2048²), so 6 is the no-regression floor — this captures
	// speculative-decode / small-batch (m=6-7), previously stuck on the slow GEMV.
	if m >= 6 && w.n%64 == 0 && w.k%32 == 0 {
		if err := rec.q8PrefillMMQ(x, w, o, m); err == nil {
			return nil
		}
	}
	if rc := C.cu_qmatmul_q8(x.ptr, w.q, w.scales, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.int(w.nb), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: rec QMatMulResident failed (code %d)", int(rc))
	}
	return nil
}

// QMatMulResidentAcc records dst += x·dequant(w) (beta=1) — the residual-add-FUSED Q8 GEMV. It folds
// the transformer residual add into the projection kernel (the same beta=1 fusion the f32 path gets
// via MatMulAcc and the graph decoder via QMatMulAccInto), eliminating the separate cu_add_f32 launch
// and its scratch HBM round-trip in the decode o/down-projection epilogue. Bit-identical to
// QMatMulResident-into-scratch followed by a Binary add. Intended for the GEMV (decode/small-m)
// regime; the flat buffer shapes are ignored in favour of the explicit m/w.k/w.n (as QMatMulResident).
func (rec *Recorder) QMatMulResidentAcc(x *DeviceF32, w *ResidentBQ8, dst *DeviceF32, m int) error {
	if x.ptr == nil || w.q == nil || w.scales == nil || dst.ptr == nil {
		return fmt.Errorf("cuda: rec QMatMulResidentAcc on a freed handle")
	}
	if rc := C.cu_qmatmul_q8(x.ptr, w.q, w.scales, dst.ptr, C.int(m), C.int(w.k), C.int(w.n), C.int(w.nb), C.float(1)); rc != 0 {
		return fmt.Errorf("cuda: rec QMatMulResidentAcc failed (code %d)", int(rc))
	}
	return nil
}

// q4kWMMAThreshold is the m (row count) at/above which QMatMulResidentQ4K routes prefill to the
// tensor-core WMMA path instead of the scalar M-tiled GEMV. Below it the fixed O(K·N) weight-dequant
// pass (a full f16 [K,N] scratch round-trip — ~64 MB of extra traffic at 4096², dominating the runtime
// at small m) isn't amortized over enough rows and the MT decode-GEMV wins; at/above it the f16
// tensor-core GEMM dominates. From the MT-vs-WMMA crossover bench (4096²,
// cuda_q4k_wmma_crossover_bench_internal_test.go), with the coalesced-write dequant
// (cu_dequant_q4k_to_f16 at 62 GB/s vs the old warp-per-column 29 GB/s): m=16 MT wins, m=32 a tie
// (WMMA ~3%, within bench noise), m=48 WMMA 1.35×, m=64 1.61×, m=128 2.42×, m=256 2.71×. 48 is the
// no-regression floor. (The 29 GB/s dequant pinned this at 128; halving its ~64 MB f16-scratch
// round-trip cost with coalesced writes dropped the crossover to 48 and raised every win. The
// remaining lever is a Marlin-style dequant-IN-tile GEMM that erases the round-trip entirely.)
//
//perfscan:ignore PS6023 cgo GPU-dispatch wrapper; no host wall-clock loop
const q4kWMMAThreshold = 48

// QMatMulResidentQ4K records o[m, w.n] = x[m, w.k]·dequant(w) for a resident Q4_K (4-bit k-quant
// super-block) weight — the aggressive-quant decode path. Q4_K is 2× smaller than Q8 (enables models that
// don't fit at Q8, matching the ggml Q4_K_M accuracy) but has NO int-tensor-core prefill GEMM, so this is
// the DECODE-oriented GEMV for all m (M>1 prefill re-reads the weight; use Q8 when prefill throughput matters).
func (rec *Recorder) QMatMulResidentQ4K(x *DeviceF32, w *ResidentBQ4K, o *DeviceF32, m int) error {
	if x.ptr == nil || w.q == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec QMatMulResidentQ4K on a freed handle")
	}
	// Explicit m / w.k / w.n (like the Q8 recorder path) — the llamagpu scratch buffers are flat, so their
	// DeviceF32 rows/cols don't carry the logical [m,k]; QMatMulInto's shape check would reject them.
	//
	// PREFILL/BATCH (m>=8): route to the weight-read-once M-tiled GEMM so column n's Q4_K block is decoded
	// ONCE across all M rows, not re-read per row (the dominant weight-bandwidth cost in this bandwidth-bound
	// path). BIT-IDENTICAL to the GEMV (same fp ops/order — TestCUDAQ4KMatMulMTParity). Mirrors the routing
	// ResidentBQ4K.qmatmul already has (cuda_quant_q4k.go); this recorder method is the path llamagpu's decoder
	// actually calls for Q4_K weights, so it was silently stuck on the GEMV for all M. Falls back on error.
	// Threshold m>=4 (measured crossover, cuda_q4k_mt_crossover_bench_internal_test.go): the MT
	// GEMM decodes each Q4_K block ONCE across all M rows while the GEMV re-decodes it per row,
	// so MT wins once M amortizes the tiling. m=4 wins at every tested shape (2048² 1.42×,
	// 2048×5632 1.44×; m=8 1.76×); m=2 is a tie (GEMV ~1-5% ahead — the pad/tiling isn't
	// amortized yet), so 4 is the no-regression floor. This captures speculative-decode /
	// small-batch (m=4-7) on the common Q4_K_M format, previously stuck on the GEMV. MT is
	// BIT-IDENTICAL to the GEMV (TestCUDAQ4KMatMulMTParity), so this is golden-safe.
	// PREFILL (m>=q4kWMMAThreshold, N%16==0): route to the tensor-core WMMA path — dequant the Q4_K
	// weight to f16 ONCE and run the f16 tensor-core GEMM, several× faster than the scalar MT decode
	// GEMV once M amortizes the O(K·N) dequant pass (M512/K4096/N4096: MT 13.98ms → WMMA 4.32ms = 3.24×).
	// f16-accum (~1e-4 rel vs the bit-identical MT path), the incumbent Q4_K prefill tolerance. Falls back
	// to MT on any error (N%16!=0, alloc fail) — always-correct.
	if m >= q4kWMMAThreshold && w.n%16 == 0 && w.k%16 == 0 {
		if err := rec.q4kPrefillWMMA(x, w, o, m); err == nil {
			return nil
		}
	}
	if m >= 4 {
		if rc := C.cu_qmatmul_q4k_mt(x.ptr, w.q, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(0)); rc == 0 {
			return nil
		}
	}
	if rc := C.cu_qmatmul_q4k(x.ptr, w.q, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: rec QMatMulResidentQ4K failed (code %d)", int(rc))
	}
	return nil
}

// QMatMulResidentAccQ4K records dst += x·dequant(w) (beta=1) — the residual-add-FUSED Q4_K decode
// GEMV, the Q4_K twin of QMatMulResidentAcc. One launch instead of the GEMV(beta=0)→scratch + a
// separate Binary add, saving the scratch HBM round-trip in the decode o/down-projection epilogue.
// Bit-identical to QMatMulResidentQ4K-into-scratch + add; used for the decode/small-m GEMV regime.
func (rec *Recorder) QMatMulResidentAccQ4K(x *DeviceF32, w *ResidentBQ4K, dst *DeviceF32, m int) error {
	if x.ptr == nil || w.q == nil || dst.ptr == nil {
		return fmt.Errorf("cuda: rec QMatMulResidentAccQ4K on a freed handle")
	}
	if rc := C.cu_qmatmul_q4k(x.ptr, w.q, dst.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(1)); rc != 0 {
		return fmt.Errorf("cuda: rec QMatMulResidentAccQ4K failed (code %d)", int(rc))
	}
	return nil
}

// QMatMulResidentAccQ6K fuses the residual add into the Q6_K decode GEMV (dst += x·dequant(w), beta=1),
// the Q6_K twin of QMatMulResidentAccQ4K. Q4_K_M models store attn_v/ffn_down as Q6_K, so a non-Llama
// Q4_K_M model served via the generic Decoder has un-fused Q6_K projections at decode without this.
// qmatmul_q6k's epilogue is out[warp] = beta*out[warp] + acc, so beta=1 is exactly the residual add.
func (rec *Recorder) QMatMulResidentAccQ6K(x *DeviceF32, w *ResidentBQ6K, dst *DeviceF32, m int) error {
	if x.ptr == nil || w.q == nil || dst.ptr == nil {
		return fmt.Errorf("cuda: rec QMatMulResidentAccQ6K on a freed handle")
	}
	if rc := C.cu_qmatmul_q6k(x.ptr, w.q, dst.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(1)); rc != 0 {
		return fmt.Errorf("cuda: rec QMatMulResidentAccQ6K failed (code %d)", int(rc))
	}
	return nil
}

// q6kWMMAThreshold is the m at/above which QMatMulResidentQ6K routes prefill to the tensor-core WMMA
// path. Q6_K's coalesced dequant (#880, ~57 GB/s) matches Q4_K's, and Q6_K's scalar MT decode is
// HEAVIER (6-bit unpack) so WMMA overtakes it no later than Q4_K's benched floor of 48 — a
// conservative, no-regression threshold (Q6_K WMMA M512/K4096/N4096 = 3.72ms vs MT 16.47ms = 4.43×).
//
//perfscan:ignore PS6023 inside Free() cleanup; cold teardown path
const q6kWMMAThreshold = 48

// QMatMulResidentQ6K records o[m,N] = x[m,K]·dequant(w) for a resident Q6_K (6-bit k-quant) weight —
// the Q6_K twin of QMatMulResidentQ4K. Decode (m<2) uses the GEMV; prefill/batch (m>=2) the
// weight-read-once M-tiled GEMM (bit-identical); large prefill (m>=q6kWMMAThreshold, N%16==0) the
// tensor-core WMMA path (dequant→f16→cu_wmma_gemm, f16-accum ~1e-4 rel, several× faster). Any error
// falls back to the always-correct GEMV.
func (rec *Recorder) QMatMulResidentQ6K(x *DeviceF32, w *ResidentBQ6K, o *DeviceF32, m int) error {
	if x.ptr == nil || w.q == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec QMatMulResidentQ6K on a freed handle")
	}
	if m >= q6kWMMAThreshold && w.n%16 == 0 && w.k%16 == 0 {
		if err := rec.q6kPrefillWMMA(x, w, o, m); err == nil {
			return nil
		}
	}
	if m >= 2 {
		if rc := C.cu_qmatmul_q6k_mt(x.ptr, w.q, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(0)); rc == 0 {
			return nil
		}
	}
	if rc := C.cu_qmatmul_q6k(x.ptr, w.q, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: rec QMatMulResidentQ6K failed (code %d)", int(rc))
	}
	return nil
}

// q8PrefillMMQ computes o[m,N] = x[m,K]·dequant(w) via the per-row-scale int8 MMQ GEMM, reusing the
// resident Q8 weight (w.q/w.scales) directly as the MMQ weight. M is padded to 64 (pad rows produce
// ignored outputs — the GEMM is row-independent), the activation is int8-quantized per row, and the
// first m rows of the [mPad,N] result are copied out. Mirrors ResidentMMQ.MatMulDevice's plumbing.
func (rec *Recorder) q8PrefillMMQ(x *DeviceF32, w *ResidentBQ8, o *DeviceF32, m int) error {
	mPad := ((m + 63) / 64) * 64
	dA8 := C.cu_alloc_i8(C.int(mPad * w.k))
	dAs := C.cu_alloc_f32(C.int(mPad))
	dC := C.cu_alloc_f32(C.int(mPad * w.n))
	if dA8 == nil || dAs == nil || dC == nil {
		C.cu_free_f32(unsafe.Pointer(dA8))
		C.cu_free_f32(dAs)
		C.cu_free_f32(dC)
		return fmt.Errorf("cuda: q8 MMQ scratch alloc failed")
	}
	defer C.cu_free_f32(unsafe.Pointer(dA8))
	defer C.cu_free_f32(dAs)
	defer C.cu_free_f32(dC)
	if rc := C.cu_quant_rows_i8(x.ptr, unsafe.Pointer(dA8), unsafe.Pointer(dAs), C.int(m), C.int(w.k)); rc != 0 {
		return fmt.Errorf("cuda: q8 MMQ activation quant failed (code %d)", int(rc))
	}
	if rc := C.cu_matmul_i8_mmq_r(unsafe.Pointer(dA8), w.q, unsafe.Pointer(dAs), w.scales, dC, C.int(mPad), C.int(w.k), C.int(w.n)); rc != 0 {
		return fmt.Errorf("cuda: q8 MMQ matmul failed (code %d)", int(rc))
	}
	if rc := C.cu_blit(o.ptr, C.int(0), dC, C.int(0), C.int(m*w.n)); rc != 0 {
		return fmt.Errorf("cuda: q8 MMQ result copy failed (code %d)", int(rc))
	}
	return nil
}

// q4kPrefillWMMA computes o[m,N] = x[m,K]·dequant(w) on TENSOR CORES for compute-bound Q4_K
// prefill: it dequantizes the resident Q4_K weight to a contiguous f16 [K,N] matrix ONCE, converts
// the f32 activation to f16, and runs the f16 WMMA GEMM (cu_wmma_gemm) — the tensor-core replacement
// for the scalar M-tiled GEMV (cu_qmatmul_q4k_mt), which streams a scalar decode-multiply per output.
// M is padded up to a multiple of 16 (WMMA tiles 16×16×16; pad rows produce ignored outputs — the
// GEMM is row-independent, exactly like q8PrefillMMQ pads to 64), the result lands in an [mPad,N]
// scratch, and the first m rows are copied out. Requires N,K %16==0 (Q4_K K%256==0 always holds);
// returns an error otherwise so the caller falls back to the always-correct MT GEMV. The f16-accum
// deviates from the MT path's bit-identical arithmetic by ~1e-4 rel — the same tolerance llama.cpp's
// tensor-core Q4_K prefill rides (§incumbent-tolerance), and only on the M-large prefill path.
func (rec *Recorder) q4kPrefillWMMA(x *DeviceF32, w *ResidentBQ4K, o *DeviceF32, m int) error {
	bf16 := C.cu_alloc_u16(C.int(w.k * w.n)) // dequantized weight [K,N] f16
	if bf16 == nil {
		return fmt.Errorf("cuda: q4k WMMA weight scratch alloc failed")
	}
	defer C.cu_free_f32(bf16)
	if rc := C.cu_dequant_q4k_to_f16(w.q, bf16, C.int(w.k), C.int(w.n)); rc != 0 {
		return fmt.Errorf("cuda: q4k WMMA dequant failed (code %d)", int(rc))
	}
	// cuBLAS f16 tensor-core GEMM (cublasGemmEx, f16 in; f16 or f32 accum per f16AccEnabled) — ~2.2x the hand cu_wmma_gemm on
	// FFN prefill shapes, no M-pad / activation scratch / output copy. Same f16-input tolerance.
	if rc := matmulF16wPrefill(x.ptr, bf16, o.ptr, m, w.k, w.n, 0); rc != 0 {
		return fmt.Errorf("cuda: q4k prefill cuBLAS f16 gemm failed (code %d)", int(rc))
	}
	return nil
}

// q6kPrefillWMMA is the Q6_K twin of q4kPrefillWMMA: dequant the resident Q6_K weight to a contiguous
// f16 [K,N] scratch ONCE (cu_dequant_q6k_to_f16, coalesced #880), convert the f32 activation to f16,
// run the f16 tensor-core GEMM into an [mPad,N] scratch (M padded to ×16, pad rows ignored), copy the
// first m rows out. Requires N,K %16==0. Returns an error → caller falls back to the MT GEMV.
func (rec *Recorder) q6kPrefillWMMA(x *DeviceF32, w *ResidentBQ6K, o *DeviceF32, m int) error {
	bf16 := C.cu_alloc_u16(C.int(w.k * w.n)) // dequantized weight [K,N] f16
	if bf16 == nil {
		return fmt.Errorf("cuda: q6k WMMA weight scratch alloc failed")
	}
	defer C.cu_free_f32(bf16)
	if rc := C.cu_dequant_q6k_to_f16(w.q, bf16, C.int(w.k), C.int(w.n)); rc != 0 {
		return fmt.Errorf("cuda: q6k WMMA dequant failed (code %d)", int(rc))
	}
	// cuBLAS f16 tensor-core GEMM (cublasGemmEx, f16 in; f16 or f32 accum per f16AccEnabled) — ~2.2x the hand cu_wmma_gemm
	// on FFN prefill shapes, and no M-pad / activation scratch / output copy (#906). Same tolerance.
	if rc := matmulF16wPrefill(x.ptr, bf16, o.ptr, m, w.k, w.n, 0); rc != 0 {
		return fmt.Errorf("cuda: q6k prefill cuBLAS f16 gemm failed (code %d)", int(rc))
	}
	return nil
}

// q5kWMMAThreshold is the m at/above which QMatMulResidentQ5K routes prefill to the tensor-core WMMA
// path. Q5_K's dequant is more ALU-bound (per-element get_sm + qh-bit) so its coalesced dequant runs
// ~34 GB/s (vs Q4_K's 62) → a higher WMMA fixed cost, so this floor is set conservatively above Q4_K's
// 48 (Q5_K WMMA M512/K4096/N4096 = 4.16ms vs MT 17.09ms = 4.11×). No regression below it (MT decode).
//
//perfscan:ignore PS6023 stale line beyond EOF; file only 583 lines
const q5kWMMAThreshold = 64

// QMatMulResidentQ5K records o[m,N] = x[m,K]·dequant(w) for a resident Q5_K (5-bit k-quant) weight —
// the Q5_K twin of QMatMulResidentQ4K/Q6K (GEMV decode m<2 / weight-read-once MT m>=2 / tensor-core
// WMMA prefill m>=q5kWMMAThreshold). Any error falls back to the always-correct GEMV.
func (rec *Recorder) QMatMulResidentQ5K(x *DeviceF32, w *ResidentBQ5K, o *DeviceF32, m int) error {
	if x.ptr == nil || w.q == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec QMatMulResidentQ5K on a freed handle")
	}
	if m >= q5kWMMAThreshold && w.n%16 == 0 && w.k%16 == 0 {
		if err := rec.q5kPrefillWMMA(x, w, o, m); err == nil {
			return nil
		}
	}
	if m >= 2 {
		if rc := C.cu_qmatmul_q5k_mt(x.ptr, w.q, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(0)); rc == 0 {
			return nil
		}
	}
	if rc := C.cu_qmatmul_q5k(x.ptr, w.q, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: rec QMatMulResidentQ5K failed (code %d)", int(rc))
	}
	return nil
}

// q5kPrefillWMMA is the Q5_K twin of q4k/q6kPrefillWMMA: dequant the resident Q5_K weight to a
// contiguous f16 [K,N] scratch once (coalesced cu_dequant_q5k_to_f16), convert the activation to f16,
// run the f16 tensor-core GEMM into an [mPad,N] scratch (M padded ×16), copy the first m rows out.
func (rec *Recorder) q5kPrefillWMMA(x *DeviceF32, w *ResidentBQ5K, o *DeviceF32, m int) error {
	bf16 := C.cu_alloc_u16(C.int(w.k * w.n)) // dequantized weight [K,N] f16
	if bf16 == nil {
		return fmt.Errorf("cuda: q5k WMMA weight scratch alloc failed")
	}
	defer C.cu_free_f32(bf16)
	if rc := C.cu_dequant_q5k_to_f16(w.q, bf16, C.int(w.k), C.int(w.n)); rc != 0 {
		return fmt.Errorf("cuda: q5k WMMA dequant failed (code %d)", int(rc))
	}
	// cuBLAS f16 tensor-core GEMM (cublasGemmEx, f16 in; f16 or f32 accum per f16AccEnabled) — ~2.2x the hand cu_wmma_gemm
	// on FFN prefill shapes, and no M-pad / activation scratch / output copy (#906). Same tolerance.
	if rc := matmulF16wPrefill(x.ptr, bf16, o.ptr, m, w.k, w.n, 0); rc != 0 {
		return fmt.Errorf("cuda: q5k prefill cuBLAS f16 gemm failed (code %d)", int(rc))
	}
	return nil
}

// q2kWMMAThreshold is the m at/above which QMatMulResidentQ2K routes prefill to the tensor-core WMMA
// path. Q2_K's coalesced dequant (57 GB/s, like Q6_K) → the Q6_K floor of 48 applies (Q2_K WMMA
// M512/K4096/N4096 = 3.57ms vs MT 12.62ms = 3.53×). No regression below it (MT decode).
//
//perfscan:ignore PS6023 stale line beyond EOF; file only 583 lines
const q2kWMMAThreshold = 48

// QMatMulResidentQ2K records o[m,N] = x[m,K]·dequant(w) for a resident Q2_K (2-bit k-quant, 0.25 B/w)
// weight — the aggressive-quant path that lets models too big for Q8 fit the 12GB 3060. GEMV decode
// (m<2) / weight-read-once MT (m>=2) / tensor-core WMMA prefill (m>=q2kWMMAThreshold). Falls back to
// the always-correct GEMV on error.
func (rec *Recorder) QMatMulResidentQ2K(x *DeviceF32, w *ResidentBQ2K, o *DeviceF32, m int) error {
	if x.ptr == nil || w.q == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec QMatMulResidentQ2K on a freed handle")
	}
	if m >= q2kWMMAThreshold && w.n%16 == 0 && w.k%16 == 0 {
		if err := rec.q2kPrefillWMMA(x, w, o, m); err == nil {
			return nil
		}
	}
	if m >= 2 {
		if rc := C.cu_qmatmul_q2k_mt(x.ptr, w.q, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(0)); rc == 0 {
			return nil
		}
	}
	if rc := C.cu_qmatmul_q2k(x.ptr, w.q, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: rec QMatMulResidentQ2K failed (code %d)", int(rc))
	}
	return nil
}

// q2kPrefillWMMA is the Q2_K twin of q4k/q5k/q6kPrefillWMMA: dequant the resident Q2_K weight to a
// contiguous f16 [K,N] scratch once (coalesced cu_dequant_q2k_to_f16), convert the activation to f16,
// run the f16 tensor-core GEMM into an [mPad,N] scratch (M padded ×16), copy the first m rows out.
func (rec *Recorder) q2kPrefillWMMA(x *DeviceF32, w *ResidentBQ2K, o *DeviceF32, m int) error {
	bf16 := C.cu_alloc_u16(C.int(w.k * w.n)) // dequantized weight [K,N] f16
	if bf16 == nil {
		return fmt.Errorf("cuda: q2k WMMA weight scratch alloc failed")
	}
	defer C.cu_free_f32(bf16)
	if rc := C.cu_dequant_q2k_to_f16(w.q, bf16, C.int(w.k), C.int(w.n)); rc != 0 {
		return fmt.Errorf("cuda: q2k WMMA dequant failed (code %d)", int(rc))
	}
	// cuBLAS f16 tensor-core GEMM (cublasGemmEx, f16 in; f16 or f32 accum per f16AccEnabled) — ~2.2x the hand cu_wmma_gemm
	// on FFN prefill shapes, and no M-pad / activation scratch / output copy (#906). Same tolerance.
	if rc := matmulF16wPrefill(x.ptr, bf16, o.ptr, m, w.k, w.n, 0); rc != 0 {
		return fmt.Errorf("cuda: q2k prefill cuBLAS f16 gemm failed (code %d)", int(rc))
	}
	return nil
}

// q3kWMMAThreshold is the m at/above which QMatMulResidentQ3K routes prefill to the tensor-core WMMA
// path. Q3_K's dequant is NOT yet coalesced (unlike Q4_K #878 / Q6_K #880 / Q5_K #882 / Q2_K #883),
// so its WMMA fixed cost is higher → the conservative pre-coalescing floor of 128 (Q3_K WMMA
// M512/K4096/N4096 = 4.31ms vs MT 13.09ms = 3.04×, #876). Below it the MT decode wins. A follow-up
// coalescing of the 3-plane meta/qs/hmask dequant would lower this like the other formats.
//
//perfscan:ignore PS6023 stale line beyond EOF; file only 583 lines
const q3kWMMAThreshold = 128

// QMatMulResidentQ3K records o[m,N] = x[m,K]·dequant(w) for a resident Q3_K (3-bit k-quant, repacked
// meta/qs/hmask planes) weight — the Q3_K twin of the other K-quant recorder paths (GEMV decode m<2 /
// weight-read-once MT m>=2 / tensor-core WMMA prefill m>=q3kWMMAThreshold). Falls back to GEMV on error.
func (rec *Recorder) QMatMulResidentQ3K(x *DeviceF32, w *ResidentBQ3K, o *DeviceF32, m int) error {
	if x.ptr == nil || w.meta == nil || o.ptr == nil {
		return fmt.Errorf("cuda: rec QMatMulResidentQ3K on a freed handle")
	}
	if m >= q3kWMMAThreshold && w.n%16 == 0 && w.k%16 == 0 {
		if err := rec.q3kPrefillWMMA(x, w, o, m); err == nil {
			return nil
		}
	}
	if m >= 2 {
		if rc := C.cu_qmatmul_q3k_mt(x.ptr, w.meta, w.qs, w.hm, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(0)); rc == 0 {
			return nil
		}
	}
	if rc := C.cu_qmatmul_q3k(x.ptr, w.meta, w.qs, w.hm, o.ptr, C.int(m), C.int(w.k), C.int(w.n), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: rec QMatMulResidentQ3K failed (code %d)", int(rc))
	}
	return nil
}

// q3kPrefillWMMA is the Q3_K twin of the other *PrefillWMMA helpers: dequant the resident Q3_K weight
// (repacked meta/qs/hmask planes) to a contiguous f16 [K,N] scratch once (cu_dequant_q3k_to_f16),
// convert the activation to f16, run the f16 tensor-core GEMM into an [mPad,N] scratch (M padded ×16),
// copy the first m rows out.
func (rec *Recorder) q3kPrefillWMMA(x *DeviceF32, w *ResidentBQ3K, o *DeviceF32, m int) error {
	bf16 := C.cu_alloc_u16(C.int(w.k * w.n)) // dequantized weight [K,N] f16
	if bf16 == nil {
		return fmt.Errorf("cuda: q3k WMMA weight scratch alloc failed")
	}
	defer C.cu_free_f32(bf16)
	if rc := C.cu_dequant_q3k_to_f16(w.meta, w.qs, w.hm, bf16, C.int(w.k), C.int(w.n)); rc != 0 {
		return fmt.Errorf("cuda: q3k WMMA dequant failed (code %d)", int(rc))
	}
	// cuBLAS f16 tensor-core GEMM (cublasGemmEx, f16 in; f16 or f32 accum per f16AccEnabled) — ~2.2x the hand cu_wmma_gemm
	// on FFN prefill shapes, and no M-pad / activation scratch / output copy (#906). Same tolerance.
	if rc := matmulF16wPrefill(x.ptr, bf16, o.ptr, m, w.k, w.n, 0); rc != 0 {
		return fmt.Errorf("cuda: q3k prefill cuBLAS f16 gemm failed (code %d)", int(rc))
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
