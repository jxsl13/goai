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

	"github.com/jxsl13/goai/tensor"
)

// qBlock is the Q8_0 group size: 32 contiguous weights (along the contracted K
// dimension) share one f32 scale, matching llama.cpp's Q8_0 block layout.
const qBlock = 32

// ResidentBQ8 is a Q8_0-quantized resident weight for a·B matmuls. The f32 weight
// B[K,N] is stored TRANSPOSED and quantized: q[n,k] (int8, each output neuron's K
// weights contiguous) with a per-32-block scale, so B[k,n] ≈ scale[n,k/32]·q[n,k].
// Reading int8 weights is 4× less memory bandwidth than f32 — the decode
// (memory-bound GEMV) win that mirrors llama.cpp's quantized kernels.
//
// DECODE-ONLY: the warp-per-output kernel is a GEMV. For prefill's M=P GEMM it is
// 6–10× SLOWER than cuBLAS f32 (measured, §PERF) — prefill is compute-bound, so it
// stays on the f32 cuBLAS path. Q8 is the memory-bound single-token decode lever.
type ResidentBQ8 struct {
	q      unsafe.Pointer // device int8 [N*K], row n = quantized B[:,n]
	scales unsafe.Pointer // device f32 [N*nb]
	k, n   int
	nb     int           // blocks per output row = ceil(K/32)
	bf16   *ResidentBF16 // optional resident f16 copy for the fast f16-cuBLAS PREFILL path (see Q8PrefillF16)
}

// Q8PrefillF16, when true at ResidentBQ8 construction, makes each Q8 weight ALSO keep a resident f16
// copy so the weight-read-once PREFILL GEMM (QMatMulResident, m>=6) routes through the fast f16-cuBLAS
// path (cu_matmul_f16w) instead of int8-MMQ. MEASURED (BenchmarkGemm_512x2048x*): GoAI's int8-MMQ runs
// at only ~0.5-0.6× its own f16 GEMM on GA106 (13-16 vs 27-31 TFLOP/s), so f16-cuBLAS is ~1.7-2.4×
// faster for prefill. Decode (the M=1 GEMV) still streams the 1-byte Q8 weight (bandwidth-optimal).
// Opt-in because the f16 copy ~doubles weight VRAM (TinyLlama Q8 1.1GB + f16 2.2GB fits 12GB).
var Q8PrefillF16 bool

// NewResidentBQ8 quantizes a host f32 weight B[K,N] to Q8_0 and uploads it. The
// quantization is symmetric per 32-block along K: scale = max|w|/127, q = round(w/scale).
func NewResidentBQ8(b *tensor.Tensor) (*ResidentBQ8, error) {
	if b.Ndim() != 2 {
		return nil, fmt.Errorf("cuda: NewResidentBQ8 needs rank-2, got %dD", b.Ndim())
	}
	bc := b.Cast(tensor.F32).Contiguous()
	k, n := bc.Shape()[0], bc.Shape()[1]
	bf := bc.Storage().F32() // bf[row*n + col] = B[row,col]
	nb := (k + qBlock - 1) / qBlock
	q := make([]int8, n*k)
	scales := make([]float32, n*nb)
	for col := 0; col < n; col++ {
		for blk := 0; blk < nb; blk++ {
			k0 := blk * qBlock
			k1 := k0 + qBlock
			if k1 > k {
				k1 = k
			}
			var amax float32
			for kk := k0; kk < k1; kk++ {
				//perfscan:ignore PS5007,PS6011 load-time weight quantization (NewResidentBQ8)
				if a := float32(math.Abs(float64(bf[kk*n+col]))); a > amax {
					amax = a
				}
			}
			s := amax / 127
			scales[col*nb+blk] = s
			var inv float32
			if s > 0 {
				inv = 1 / s
			}
			for kk := k0; kk < k1; kk++ {
				qi := int32(math.Round(float64(bf[kk*n+col] * inv)))
				if qi > 127 {
					qi = 127
				} else if qi < -127 {
					qi = -127
				}
				q[col*k+kk] = int8(qi)
			}
		}
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&q[0])), C.int(len(q)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: Q8 weight upload failed")
	}
	ds := C.cu_upload_f32((*C.float)(&scales[0]), C.int(len(scales)))
	if ds == nil {
		C.cu_free_f32(dq)
		return nil, fmt.Errorf("cuda: Q8 scales upload failed")
	}
	res := &ResidentBQ8{q: dq, scales: ds, k: k, n: n, nb: nb}
	if Q8PrefillF16 {
		// Keep a resident f16 copy of the ORIGINAL f32 weight [K,N] for the fast prefill GEMM. Best-
		// effort: on failure the prefill path just falls back to int8-MMQ.
		if bf16, e := NewResidentBF16(bc); e == nil {
			res.bf16 = bf16
		}
	}
	return res, nil
}

// QMatMulDevice computes a·dequant(B) with the activation a resident and the
// weight Q8-quantized+resident, leaving the [a.rows, N] result on the GPU.
func (r *ResidentBQ8) QMatMulDevice(a *DeviceF32) (*DeviceF32, error) {
	if r.q == nil || a.ptr == nil {
		return nil, fmt.Errorf("cuda: QMatMulDevice on a freed handle")
	}
	if a.cols != r.k {
		return nil, fmt.Errorf("cuda: QMatMulDevice inner dim mismatch a[%d,%d]·B[%d,%d]", a.rows, a.cols, r.k, r.n)
	}
	out := C.cu_alloc_f32(C.int(a.rows * r.n))
	if out == nil {
		return nil, fmt.Errorf("cuda: QMatMulDevice output alloc failed")
	}
	if rc := C.cu_qmatmul_q8(a.ptr, r.q, r.scales, out, C.int(a.rows), C.int(r.k), C.int(r.n), C.int(r.nb), C.float(0)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: Q8 matmul failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: a.rows, cols: r.n}, nil
}

// QMatMulInto computes out = a·dequant(B) into the caller's fixed buffer (beta=0)
// — the persistent-buffer Q8 matmul for the fixed-buffer / graph decode path.
// QMatMulMoeInto is QMatMulInto with per-token MoE gating: for output row m, if this expert's routing
// weight moeGate[m*gateStride + expertIdx] is 0.0 (the token did not select this expert), the GEMV skips
// the weight-streaming K-loop and writes out[m,:]=0. Bit-identical to the dense eval whose non-selected
// experts are zeroed by RowAxpy(weight=0), but ~E/K faster at decode (only the top-K experts stream
// their weight). All E expert dispatches stay recorded so a captured CUDA graph remains static. beta=0.
// m is passed explicitly (not derived from a.rows) so this works on the flat llamagpu scratch buffers
// whose DeviceF32 rows/cols don't carry the logical [m,k] — same contract as the recorder GEMV path.
func (r *ResidentBQ8) QMatMulMoeInto(a, out, moeGate *DeviceF32, m, gateStride, expertIdx int) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil || moeGate.ptr == nil {
		return fmt.Errorf("cuda: QMatMulMoeInto on a freed handle")
	}
	// &moeGate[expertIdx]; the kernel reads moeGate[row*gateStride]. Device pointer — GC-immovable, so the
	// uintptr offset is safe (unlike a Go heap pointer).
	gp := unsafe.Pointer(uintptr(moeGate.ptr) + uintptr(expertIdx)*4)
	if rc := C.cu_qmatmul_q8_moe(a.ptr, r.q, r.scales, out.ptr, C.int(m), C.int(r.k), C.int(r.n), C.int(r.nb), gp, C.int(gateStride)); rc != 0 {
		return fmt.Errorf("cuda: Q8 moe-gated matmul failed (code %d)", int(rc))
	}
	return nil
}

func (r *ResidentBQ8) QMatMulInto(a, out *DeviceF32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: QMatMulInto on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: QMatMulInto shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q8(a.ptr, r.q, r.scales, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.int(r.nb), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: Q8 matmul-into failed (code %d)", int(rc))
	}
	return nil
}

// QMatMulSwiGLUInto computes out = silu(gate) ⊙ (a·dequant(B)) — the up-projection
// GEMV with SwiGLU applied in the kernel epilogue (Tw55 fusion): replaces the
// up-GEMV + separate SwiGLU launch and the hidden-vector round-trip between them.
// gate must have out's [a.rows, N] layout (the gate projection's output).
func (r *ResidentBQ8) QMatMulSwiGLUInto(a, gate, out *DeviceF32) error {
	if r.q == nil || a.ptr == nil || gate.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: QMatMulSwiGLUInto on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n || gate.rows != out.rows || gate.cols != out.cols {
		return fmt.Errorf("cuda: QMatMulSwiGLUInto shape a[%d,%d]·B[%d,%d] gate[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, gate.rows, gate.cols, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q8_swiglu(a.ptr, r.q, r.scales, gate.ptr, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.int(r.nb)); rc != 0 {
		return fmt.Errorf("cuda: Q8 swiglu matmul failed (code %d)", int(rc))
	}
	return nil
}

// QMatMulAccInto computes c += a·dequant(B) in place (Q8 weight, beta=1), fusing
// the transformer residual add into the quantized projection — the Q8 analogue of
// ResidentB.MatMulAccInto. c must hold the residual and be [a.rows, N].
func (r *ResidentBQ8) QMatMulAccInto(a, c *DeviceF32) error {
	if r.q == nil || a.ptr == nil || c.ptr == nil {
		return fmt.Errorf("cuda: QMatMulAccInto on a freed handle")
	}
	if a.cols != r.k {
		return fmt.Errorf("cuda: QMatMulAccInto inner dim mismatch a[%d,%d]·B[%d,%d]", a.rows, a.cols, r.k, r.n)
	}
	if c.rows != a.rows || c.cols != r.n {
		return fmt.Errorf("cuda: QMatMulAccInto accumulator must be [%d,%d], got [%d,%d]", a.rows, r.n, c.rows, c.cols)
	}
	if rc := C.cu_qmatmul_q8(a.ptr, r.q, r.scales, c.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.int(r.nb), C.float(1)); rc != 0 {
		return fmt.Errorf("cuda: Q8 matmul-acc failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the quantized weight and scale buffers.
func (r *ResidentBQ8) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
	if r.scales != nil {
		C.cu_free_f32(r.scales)
		r.scales = nil
	}
	if r.bf16 != nil {
		r.bf16.Free()
		r.bf16 = nil
	}
}

// Close frees the resident weight — the llamagpu `qweight` interface (alias for Free).
func (r *ResidentBQ8) Close() error {
	r.Free()
	return nil
}
