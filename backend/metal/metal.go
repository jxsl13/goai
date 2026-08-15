//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -fobjc-arc -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation
#include "metal_bridge.h"
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
	"unsafe"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaUseMPSGraph toggles the EXPERIMENTAL MPSGraph attention prefill path (§T621) in place of the
// default per-head MPS-matmul path (mtl_mha_mps). Off by default; flipped only by SetMHAUseMPSGraph
// for the A/B benchmark. Applies to the sq==sk, window==0, non-ALiBi, dk≤128 prefill fast path only.
var mhaUseMPSGraph atomic.Bool

// SetMHAUseMPSGraph selects the MPSGraph attention prefill path (true) or the default per-head MPS
// path (false), returning the previous value. Experiment toggle (§T621) — the custom path is the
// live default. Not goroutine-safe against a concurrent OpMHA on a different backend instance.
func SetMHAUseMPSGraph(v bool) bool { return mhaUseMPSGraph.Swap(v) }

// device is the metal tensor.Device. Tensor memory stays host-side (Apple
// Silicon UMA); the kernel copies through shared MTLBuffers per call — honest
// about the current transfer cost; zero-copy is a later optimization.
type device struct{}

// cpuPrefers reports the optimized CPU backend when it registers its OWN kernel for
// op — the ADR-0008 routing gate (§T534/§T535): per-op memory-bound elementwise on
// host-resident tensors is faster on a single SIMD CPU pass than a GPU round-trip,
// but ONLY where the cpu backend actually has the kernel (otherwise the dispatch
// would fall through to the slow reference and LOSE).
func cpuPrefers(op backend.Op, dt tensor.Dtype) (backend.Backend, bool) {
	cpu, ok := backend.Get(backend.CPU)
	if !ok {
		return nil, false
	}
	if _, has := cpu.Kernel(op, dt); !has {
		return nil, false
	}
	return cpu, true
}

func (device) Kind() tensor.DeviceKind     { return tensor.KindMetal }
func (d device) String() string            { return d.Kind().String() }
func (device) Allocator() tensor.Allocator { return tensor.Heap() }

// Backend implements backend.Backend over Metal/MPS. Synchronous: every kernel
// commits and waits before returning, so Synchronize is a no-op (§V14 permits
// async later without an API break).
type Backend struct{}

func (Backend) Name() backend.Name    { return backend.Metal }
func (Backend) Device() tensor.Device { return device{} }
func (Backend) Synchronize() error    { return nil }

// Available reports whether a Metal device with MPS support exists.
func Available() bool { return C.mtl_available() == 1 }

// AvailableMemory implements backend.MemoryProber using Metal's remaining
// recommended working-set budget, which is safer for placement than total UMA.
func (Backend) AvailableMemory() (bytes int64, ok bool) {
	free := uint64(C.mtl_available_memory())
	if free == 0 {
		return 0, false
	}
	const maxInt64Bytes = uint64(1<<63 - 1)
	if free > maxInt64Bytes {
		free = maxInt64Bytes
	}
	return int64(free), true
}

// ggml quant type codes accelerated by the in-kernel quantized matmuls (§R94/§R100/§R99).
const (
	qtQ4_0 = 2
	qtQ8_0 = 8
	qtQ2_K = 10
	qtQ3_K = 11
	qtQ5_K = 13
	qtQ4_K = 12
	qtQ6_K = 14
)

// UploadQuant implements backend.ResidentQuantMatMuler (§T154): it uploads a Q8_0 weight to a
// device-resident buffer for reuse across matmuls (the decode-loop perf lever), returning
// backend.ErrQuantUnsupported for a quant type without a resident kernel yet so the caller falls
// back to the per-call path.
func (Backend) UploadQuant(weight []byte, quantType uint32, n, k int) (backend.ResidentWeight, error) {
	rw, err := uploadResident(weight, quantType, n, k)
	if err != nil {
		return nil, err // ErrQuantUnsupported for a type without a resident kernel; also avoids
		// wrapping a nil *ResidentQWeight in a non-nil interface.
	}
	return rw, nil
}

// QMatMul implements backend.QuantMatMuler (§T142): it dispatches a quantized linear layer to
// the matching in-kernel Metal matmul (Q8_0/Q4_K/Q6_K), returning backend.ErrQuantUnsupported
// for a ggml type without a GPU kernel so the caller falls back to the CPU path.
func (Backend) QMatMul(x *tensor.Tensor, weight []byte, quantType uint32, n, k int) (*tensor.Tensor, error) {
	switch quantType {
	case qtQ4_0:
		return QMatMulQ4_0(x, weight, n, k)
	case qtQ8_0:
		return QMatMulQ8_0(x, weight, n, k)
	case qtQ2_K:
		return QMatMulQ2_K(x, weight, n, k)
	case qtQ3_K:
		return QMatMulQ3_K(x, weight, n, k)
	case qtQ4_K:
		return QMatMulQ4_K(x, weight, n, k)
	case qtQ5_K:
		return QMatMulQ5_K(x, weight, n, k)
	case qtQ6_K:
		return QMatMulQ6_K(x, weight, n, k)
	default:
		return nil, backend.ErrQuantUnsupported
	}
}

// QMatMulQ8_0 computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q8_0-quantized [N,K] matrix (row-major, K/32 blocks of [f16 scale + 32 int8], §R94)
// dequantized in-kernel — a quantized linear layer run on Metal without materializing the
// full-precision weights, the building block of GPU quantized inference (§T136). K must be a
// multiple of 32 and weight must be N·(K/32)·34 bytes. Accumulation is f32 (the reference
// gguf.QMatMul is f64), so results match it within a crossTol, not bit-exactly (§V11).
func QMatMulQ8_0(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("metal: QMatMulQ8_0 x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: QMatMulQ8_0 is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%32 != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ8_0 K=%d must be a positive multiple of 32", k)
	}
	nb := k / 32
	if len(weight) != n*nb*34 {
		return nil, fmt.Errorf("metal: QMatMulQ8_0 weight %d bytes != %d (N·K/32·34)", len(weight), n*nb*34)
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	xc := x.Contiguous()
	rc := C.mtl_qmatmul_q8_0(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&weight[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ8_0 failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ4_0 computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q4_0-quantized [N,K] matrix (row-major, K/32 blocks of [f16 scale + 16 bytes of 32 4-bit
// nibbles], 18 B/block, §R94): low nibble → element i, high nibble → element i+16, value =
// d·(nibble−8). Dequantized in-kernel so the full-precision weights are never materialized — the
// GPU path for the very common Q4_0 GGUF format (§T166). K must be a multiple of 32 and weight
// must be N·(K/32)·18 bytes. f32 accumulation (the reference gguf.QMatMul is f64), so results
// match within a crossTol, not bit-exactly (§V11).
func QMatMulQ4_0(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("metal: QMatMulQ4_0 x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: QMatMulQ4_0 is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%32 != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ4_0 K=%d must be a positive multiple of 32", k)
	}
	nb := k / 32
	if len(weight) != n*nb*18 {
		return nil, fmt.Errorf("metal: QMatMulQ4_0 weight %d bytes != %d (N·K/32·18)", len(weight), n*nb*18)
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	xc := x.Contiguous()
	rc := C.mtl_qmatmul_q4_0(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&weight[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ4_0 failed (code %d)", int(rc))
	}
	return out, nil
}

// ResidentQWeight is a quantized weight uploaded ONCE to a device-resident GPU buffer and reused
// across many QMatMul calls (§T153/§T155) — so a decode loop, which reuses every weight for every
// token, uploads each weight a single time instead of per step. Any k-quant/Q8_0 format is
// supported (the resident dispatch selects the kernel by quant type). Call Close to free the GPU
// buffer (it is not reclaimed automatically).
type ResidentQWeight struct {
	handle  unsafe.Pointer // retained MTLBuffer (nil after Close)
	n, k    int
	qt      uint32          // ggml quant type code
	cleanup runtime.Cleanup // frees the buffer if the value is GC'd without an explicit Close
}

// residentRowBytes returns the per-row byte size of a quantized [·,k] weight of ggml type qt, and
// the required alignment (block element count) of k. rowBytes*N is the full resident buffer size.
func residentRowBytes(qt uint32, k int) (rowBytes, align int, ok bool) {
	switch qt {
	case qtQ4_0:
		return (k / 32) * 18, 32, true
	case qtQ8_0:
		return (k / 32) * 34, 32, true
	case qtQ2_K:
		return (k / 256) * 84, 256, true
	case qtQ3_K:
		return (k / 256) * 110, 256, true
	case qtQ4_K:
		return (k / 256) * 144, 256, true
	case qtQ5_K:
		return (k / 256) * 176, 256, true
	case qtQ6_K:
		return (k / 256) * 210, 256, true
	}
	return 0, 0, false
}

// uploadResident uploads a quantized [N,K] weight of ggml type qt to a resident GPU buffer.
func uploadResident(weight []byte, qt uint32, n, k int) (*ResidentQWeight, error) {
	rowBytes, align, ok := residentRowBytes(qt, k)
	if !ok {
		return nil, backend.ErrQuantUnsupported
	}
	if k <= 0 || k%align != 0 {
		return nil, fmt.Errorf("metal: resident upload K=%d must be a positive multiple of %d", k, align)
	}
	if len(weight) != n*rowBytes {
		return nil, fmt.Errorf("metal: resident upload weight %d bytes != %d", len(weight), n*rowBytes)
	}
	h := C.mtl_qweight_upload((*C.uchar)(unsafe.Pointer(&weight[0])), C.int(len(weight)))
	if h == nil {
		return nil, fmt.Errorf("metal: qweight upload failed")
	}
	r := &ResidentQWeight{handle: unsafe.Pointer(h), n: n, k: k, qt: qt}
	// free the GPU buffer if r is collected without an explicit Close (the cleanup captures only
	// the handle, never r, so r stays collectible); Close stops it to avoid a double free.
	r.cleanup = runtime.AddCleanup(r, func(p unsafe.Pointer) { C.mtl_qweight_free(p) }, unsafe.Pointer(h))
	return r, nil
}

// UploadQWeightQ8_0 uploads a Q8_0-quantized [N,K] weight (N·(K/32)·34 bytes) to a resident GPU
// buffer for repeated QMatMul. K must be a multiple of 32. Close the result to free it.
func UploadQWeightQ8_0(weight []byte, n, k int) (*ResidentQWeight, error) {
	return uploadResident(weight, qtQ8_0, n, k)
}

// QMatMul computes y[M,N] = x[M,K] · dequant(W)ᵀ using the resident weight — only x is uploaded.
func (r *ResidentQWeight) QMatMul(x *tensor.Tensor) (*tensor.Tensor, error) {
	if r.handle == nil {
		return nil, fmt.Errorf("metal: resident weight already closed")
	}
	if x.Ndim() != 2 || x.Shape()[1] != r.k {
		return nil, fmt.Errorf("metal: resident QMatMul x must be [M,%d], got %v", r.k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: resident QMatMul is f32-only, got %v", x.Dtype())
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, r.n})
	if m == 0 || r.n == 0 {
		return out, nil
	}
	xc := x.Contiguous()
	rc := C.mtl_qmatmul_resident(
		(*C.float)(&xc.Storage().F32()[0]),
		r.handle,
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(r.k), C.int(r.n), C.int(r.qt),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: resident QMatMul failed (code %d)", int(rc))
	}
	return out, nil
}

// Close frees the resident GPU buffer. Safe to call more than once. Optional — an un-closed
// weight is freed by a runtime cleanup when it becomes unreachable — but Close releases the GPU
// memory promptly.
func (r *ResidentQWeight) Close() error {
	if r.handle != nil {
		r.cleanup.Stop() // don't let the cleanup double-free
		C.mtl_qweight_free(r.handle)
		r.handle = nil
	}
	return nil
}

// QMatMulQ4_K computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q4_K-quantized [N,K] matrix (row-major, K/256 super-blocks of 144 bytes, §R100) dequantized
// in-kernel — the DOMINANT real-world quant (the bulk of Q4_K_M models) run on Metal without
// materializing the full-precision weights (§T138). K must be a multiple of 256 and weight
// must be N·(K/256)·144 bytes. Accumulation is f32 (the reference gguf.QMatMul is f64), so
// results match it within a crossTol, not bit-exactly (§V11).
func QMatMulQ4_K(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("metal: QMatMulQ4_K x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: QMatMulQ4_K is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ4_K K=%d must be a positive multiple of 256", k)
	}
	nsb := k / 256
	if len(weight) != n*nsb*144 {
		return nil, fmt.Errorf("metal: QMatMulQ4_K weight %d bytes != %d (N·K/256·144)", len(weight), n*nsb*144)
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	xc := x.Contiguous()
	rc := C.mtl_qmatmul_q4k(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&weight[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ4_K failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ6_K computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q6_K-quantized [N,K] matrix (row-major, K/256 super-blocks of 210 bytes, §R99) dequantized
// in-kernel — the higher-precision tensors of Q4_K_M models (attn_v/ffn_down/output) run on
// Metal (§T140). K must be a multiple of 256 and weight must be N·(K/256)·210 bytes.
// Accumulation is f32 (the reference gguf.QMatMul is f64), so results match within a crossTol.
func QMatMulQ6_K(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("metal: QMatMulQ6_K x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: QMatMulQ6_K is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ6_K K=%d must be a positive multiple of 256", k)
	}
	nsb := k / 256
	if len(weight) != n*nsb*210 {
		return nil, fmt.Errorf("metal: QMatMulQ6_K weight %d bytes != %d (N·K/256·210)", len(weight), n*nsb*210)
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	xc := x.Contiguous()
	rc := C.mtl_qmatmul_q6k(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&weight[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ6_K failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ5_K computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q5_K-quantized [N,K] matrix (row-major, K/256 super-blocks of 176 bytes, §R102) dequantized
// in-kernel — the Q5_K_M weight format run on Metal (§T143). K must be a multiple of 256 and
// weight must be N·(K/256)·176 bytes. Accumulation is f32 (the reference gguf.QMatMul is f64),
// so results match within a crossTol.
func QMatMulQ5_K(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("metal: QMatMulQ5_K x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: QMatMulQ5_K is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ5_K K=%d must be a positive multiple of 256", k)
	}
	nsb := k / 256
	if len(weight) != n*nsb*176 {
		return nil, fmt.Errorf("metal: QMatMulQ5_K weight %d bytes != %d (N·K/256·176)", len(weight), n*nsb*176)
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	xc := x.Contiguous()
	rc := C.mtl_qmatmul_q5k(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&weight[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ5_K failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ2_K computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q2_K-quantized [N,K] matrix (row-major, K/256 super-blocks of 84 bytes, §R104) dequantized
// in-kernel — the smallest quant, run on Metal (§T145). K must be a multiple of 256 and weight
// must be N·(K/256)·84 bytes. Accumulation is f32 (the reference gguf.QMatMul is f64), so
// results match within a crossTol.
func QMatMulQ2_K(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("metal: QMatMulQ2_K x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: QMatMulQ2_K is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ2_K K=%d must be a positive multiple of 256", k)
	}
	nsb := k / 256
	if len(weight) != n*nsb*84 {
		return nil, fmt.Errorf("metal: QMatMulQ2_K weight %d bytes != %d (N·K/256·84)", len(weight), n*nsb*84)
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	xc := x.Contiguous()
	rc := C.mtl_qmatmul_q2k(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&weight[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ2_K failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ3_K computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q3_K-quantized [N,K] matrix (row-major, K/256 super-blocks of 110 bytes, §R103) dequantized
// in-kernel — the Q3_K_M weight format run on Metal (§T147). K must be a multiple of 256 and
// weight must be N·(K/256)·110 bytes. Accumulation is f32 (the reference gguf.QMatMul is f64),
// so results match within a crossTol.
func QMatMulQ3_K(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("metal: QMatMulQ3_K x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: QMatMulQ3_K is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ3_K K=%d must be a positive multiple of 256", k)
	}
	nsb := k / 256
	if len(weight) != n*nsb*110 {
		return nil, fmt.Errorf("metal: QMatMulQ3_K weight %d bytes != %d (N·K/256·110)", len(weight), n*nsb*110)
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	xc := x.Contiguous()
	rc := C.mtl_qmatmul_q3k(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&weight[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: QMatMulQ3_K failed (code %d)", int(rc))
	}
	return out, nil
}

func (Backend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	if dtype == tensor.F32 {
		switch op {
		case backend.OpMatMul:
			return matmulF32, true
		case backend.OpMHA:
			return mhaF32, true
		case backend.OpFlashAttn:
			return flashAttnF32, true
		case backend.OpRetention:
			return retentionF32, true
		case backend.OpRetentionBackward:
			return retentionBackwardF32, true
		case backend.OpMHABackward:
			return mhaBackwardF32, true
		case backend.OpConv2D:
			return conv2dF32, true
		case backend.OpConv2DBackward:
			return conv2dBackwardF32, true
		case backend.OpRMSNorm:
			return rmsnormF32, true
		case backend.OpRMSNormBackward:
			return rmsnormBackwardF32, true
		case backend.OpRoPE:
			return ropeKernelF32(backend.OpRoPE, false), true
		case backend.OpRoPEBackward:
			return ropeKernelF32(backend.OpRoPEBackward, true), true
		case backend.OpSoftmax:
			return softmaxF32, true
		case backend.OpCrossEntropy:
			return crossentropyF32, true
		case backend.OpEmbed:
			return embedF32, true
		case backend.OpCrossEntropyBackward:
			return crossentropyBackwardF32, true
		case backend.OpEmbedBackward:
			return embedBackwardF32, true
		case backend.OpLayerNorm:
			return layernormF32, true
		case backend.OpLayerNormBackward:
			return layernormBackwardF32, true
		case backend.OpNeg:
			return unaryF32(backend.OpNeg, unaryNeg), true
		case backend.OpExp:
			return unaryF32(backend.OpExp, unaryExp), true
		case backend.OpLog:
			return unaryF32(backend.OpLog, unaryLog), true
		case backend.OpTanh:
			return unaryF32(backend.OpTanh, unaryTanh), true
		case backend.OpReLU:
			return unaryF32(backend.OpReLU, unaryReLU), true
		case backend.OpSigmoid:
			return unaryF32(backend.OpSigmoid, unarySigmoid), true
		case backend.OpSiLU:
			return unaryF32(backend.OpSiLU, unarySiLU), true
		case backend.OpSqrt:
			return unaryF32(backend.OpSqrt, unarySqrt), true
		case backend.OpAbs:
			return unaryF32(backend.OpAbs, unaryAbs), true
		case backend.OpGELU:
			return unaryF32(backend.OpGELU, unaryGELU), true
		case backend.OpAddBias:
			return addBiasF32, true
		case backend.OpGELUBackward:
			return geluBackwardF32, true
		case backend.OpSiLUBackward:
			return siluBackwardF32, true
		case backend.OpAddBiasBackward:
			return addBiasBackwardF32, true
		case backend.OpAdd:
			return binaryF32(backend.OpAdd, binaryAdd), true
		case backend.OpSub:
			return binaryF32(backend.OpSub, binarySub), true
		case backend.OpMul:
			return binaryF32(backend.OpMul, binaryMul), true
		case backend.OpDiv:
			return binaryF32(backend.OpDiv, binaryDiv), true
		case backend.OpMaximum:
			return binaryF32(backend.OpMaximum, binaryMax), true
		case backend.OpMinimum:
			return binaryF32(backend.OpMinimum, binaryMin), true
		}
	}
	return nil, false // everything else: fallback to Pure-Go (§I4)
}

// unary op selectors — MUST match the switch in kUnarySource (metal_bridge.m).
const (
	unaryNeg     = 0
	unaryExp     = 1
	unaryLog     = 2
	unaryTanh    = 3
	unaryReLU    = 4
	unarySigmoid = 5
	unarySiLU    = 6
	unarySqrt    = 7
	unaryAbs     = 8
	unaryGELU    = 9
)

// binary op selectors — MUST match the switch in kBinarySource (metal_bridge.m).
const (
	binaryAdd = 0
	binarySub = 1
	binaryMul = 2
	binaryDiv = 3
	binaryMax = 4
	binaryMin = 5
)

// unaryF32 returns a GPU kernel for a simple unary elementwise op (one Metal thread per
// element, selected by `sel`), keeping activations resident on the GPU. f32-only; empty
// tensors fall back to the reference (§I4). GELU (exact erf, §T352) now runs here too —
// it dominated the FFN as a CPU fallback; Metal's erf matches the reference within tol.
func unaryF32(op backend.Op, sel int) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		refFallback := func() ([]*tensor.Tensor, error) {
			return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), op, in, attrs)
		}
		if len(in) != 1 {
			return nil, fmt.Errorf("metal: unary op wants 1 input, got %d", len(in))
		}
		x := in[0]
		n := x.Numel()
		if x.Dtype() != tensor.F32 || n == 0 {
			return refFallback()
		}
		xc := x.Contiguous()
		out := tensor.New(tensor.F32, x.Shape())
		rc := C.mtl_unary_f32(
			(*C.float)(&xc.Storage().F32()[0]),
			(*C.float)(&out.Storage().F32()[0]),
			C.int(n), C.int(sel),
		)
		if rc != 0 {
			return nil, fmt.Errorf("metal: unary op failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{out}, nil
	}
}

// binaryF32 returns a GPU kernel for a same-shape binary elementwise op (one Metal thread
// per element, selected by `sel`). Only equal-shaped f32 inputs run on the GPU —
// broadcasting and non-f32 fall back to the reference (§I4) — covering the FFN's SiLU⊙up
// gate and transformer residual adds without a CPU round-trip.
func binaryF32(op backend.Op, sel int) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		refFallback := func() ([]*tensor.Tensor, error) {
			return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), op, in, attrs)
		}
		if len(in) != 2 {
			return nil, fmt.Errorf("metal: binary op wants 2 inputs, got %d", len(in))
		}
		a, b := in[0], in[1]
		n := a.Numel()
		// ADR-0008 (§T534): per-op memory-bound elementwise NEVER pays on host-resident
		// tensors — the up/down transfer plus dispatch latency dwarfs a single CPU pass
		// (profile: 0.41ms/call for a 512KB add vs ~30µs on cpu; routing binaries to the
		// cpu backend measured +10.8% on the full metal training step). The GPU kernel
		// stays as the fallback when no cpu backend is registered (resident/recorder
		// paths have their own fused ops and never come through here).
		if cpu, ok := cpuPrefers(op, a.Dtype()); ok {
			// strip the recorder: the OUTER Execute records this op onto the tape;
			// re-dispatching with the recorder attached would record it TWICE and
			// double the op's gradients (§B49 — caught by the full sweep's tight
			// trained-model bars, invisible to shape/parity suites).
			return backend.Execute(ctx.WithBackend(cpu).WithRecorder(nil), op, in, attrs)
		}
		if a.Dtype() != tensor.F32 || b.Dtype() != tensor.F32 || !a.Shape().Equal(b.Shape()) || n == 0 {
			return refFallback()
		}
		ac, bc := a.Contiguous(), b.Contiguous()
		out := tensor.New(tensor.F32, a.Shape())
		rc := C.mtl_binary_f32(
			(*C.float)(&ac.Storage().F32()[0]),
			(*C.float)(&bc.Storage().F32()[0]),
			(*C.float)(&out.Storage().F32()[0]),
			C.int(n), C.int(sel),
		)
		if rc != 0 {
			return nil, fmt.Errorf("metal: binary op failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{out}, nil
	}
}

// addBiasF32 computes O[i,j] = X[i,j] + B[j] on the GPU — the FFN's bias add, which
// was a ref/CPU fallback dominating the forward (§T352). x rank-2, bias rank-1 [cols];
// anything else falls back to the reference (§I4).
func addBiasF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpAddBias, in, attrs)
	}
	if len(in) != 2 {
		return nil, fmt.Errorf("metal: addbias wants (x, bias), got %d inputs", len(in))
	}
	x, b := in[0], in[1]
	if x.Dtype() != tensor.F32 || b.Dtype() != tensor.F32 || x.Ndim() != 2 || b.Ndim() != 1 {
		return refFallback()
	}
	rows, n := x.Shape()[0], x.Shape()[1]
	if b.Shape()[0] != n || rows == 0 || n == 0 {
		return refFallback()
	}
	xc, bc := x.Contiguous(), b.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.mtl_addbias_f32(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&bc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(rows), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: addbias failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// addBiasBackwardF32 computes dbias[j] = Σ_i g[i,j] on the GPU (§T354) — the bias-add
// VJP dispatches this so the column-sum doesn't run as a CPU scalar loop. in = (g[rows,n]).
func addBiasBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpAddBiasBackward, in, attrs)
	}
	if len(in) != 1 {
		return nil, fmt.Errorf("metal: addbias-backward wants (g), got %d inputs", len(in))
	}
	g := in[0]
	if g.Dtype() != tensor.F32 || g.Ndim() != 2 {
		return refFallback()
	}
	rows, n := g.Shape()[0], g.Shape()[1]
	if rows == 0 || n == 0 {
		return refFallback()
	}
	gc := g.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{n})
	rc := C.mtl_addbias_backward_f32(
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(rows), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: addbias-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// geluBackwardF32 computes dx = g·gelu'(x) on the GPU (§T353) — the GELU VJP dispatches
// this so training doesn't fall back to the ~30ms CPU scalar loop. in = (x, g), same shape.
func geluBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpGELUBackward, in, attrs)
	}
	if len(in) != 2 {
		return nil, fmt.Errorf("metal: gelu-backward wants (x, g), got %d inputs", len(in))
	}
	x, g := in[0], in[1]
	n := x.Numel()
	if x.Dtype() != tensor.F32 || g.Dtype() != tensor.F32 || !g.Shape().Equal(x.Shape()) || n == 0 {
		return refFallback()
	}
	xc, gc := x.Contiguous(), g.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.mtl_gelu_backward_f32(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: gelu-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// siluBackwardF32 computes dx = g·silu'(x) on the GPU (§T362) — the SiLU VJP dispatches
// this so SwiGLU/Llama training doesn't fall back to the CPU scalar loop. in = (x, g).
func siluBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpSiLUBackward, in, attrs)
	}
	if len(in) != 2 {
		return nil, fmt.Errorf("metal: silu-backward wants (x, g), got %d inputs", len(in))
	}
	x, g := in[0], in[1]
	n := x.Numel()
	if x.Dtype() != tensor.F32 || g.Dtype() != tensor.F32 || !g.Shape().Equal(x.Shape()) || n == 0 {
		return refFallback()
	}
	xc, gc := x.Contiguous(), g.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.mtl_silu_backward_f32(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: silu-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// layernormBackwardF32 computes the LayerNorm gradient on the GPU (§R35): (x,gamma,g) →
// (dx, dgamma, dbeta), one Metal thread per row (dx per-row, dgamma/dbeta via float atomics).
// It makes LayerNorm TRAINING run on the GPU. f32-only; empty tensors fall back to the ref (§I4).
func layernormBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpLayerNormBackward, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("metal: layernorm-backward wants (x, gamma, g), got %d inputs", len(in))
	}
	x, gamma, g := in[0], in[1], in[2]
	if x.Dtype() != tensor.F32 || gamma.Dtype() != tensor.F32 || g.Dtype() != tensor.F32 || x.Ndim() < 1 {
		return refFallback()
	}
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d || !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("metal: layernorm-backward gamma [%d]/g %v mismatch x %v", d, g.Shape(), x.Shape())
	}
	rows := 0
	if d > 0 {
		rows = x.Numel() / d
	}
	if rows == 0 || d == 0 {
		return refFallback()
	}
	pa, _ := attrs.(backend.NormAttrs)
	pa = pa.WithDefaults()
	xc, gc, gupc := x.Contiguous(), gamma.Contiguous(), g.Contiguous()
	dx := tensor.New(tensor.F32, x.Shape())
	dgamma := tensor.New(tensor.F32, gamma.Shape()) // zero-init atomic accumulators
	dbeta := tensor.New(tensor.F32, gamma.Shape())
	rc := C.mtl_layernorm_backward_f32(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&gupc.Storage().F32()[0]),
		(*C.float)(&dx.Storage().F32()[0]),
		(*C.float)(&dgamma.Storage().F32()[0]),
		(*C.float)(&dbeta.Storage().F32()[0]),
		C.int(rows), C.int(d), C.float(pa.Eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: layernorm-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dx, dgamma, dbeta}, nil
}

// rmsnormBackwardF32 computes the RMSNorm gradient on the GPU (§R29/§R35): (x,gamma,g) →
// (dx, dgamma), one Metal thread per row (dx per-row, dgamma via float atomics). It makes
// RMSNorm TRAINING run on the GPU. f32-only; empty tensors fall back to the reference (§I4).
func rmsnormBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpRMSNormBackward, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("metal: rmsnorm-backward wants (x, gamma, g), got %d inputs", len(in))
	}
	x, gamma, g := in[0], in[1], in[2]
	if x.Dtype() != tensor.F32 || gamma.Dtype() != tensor.F32 || g.Dtype() != tensor.F32 || x.Ndim() < 1 {
		return refFallback()
	}
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d || !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("metal: rmsnorm-backward gamma [%d]/g %v mismatch x %v", d, g.Shape(), x.Shape())
	}
	rows := 0
	if d > 0 {
		rows = x.Numel() / d
	}
	if rows == 0 || d == 0 {
		return refFallback()
	}
	pa, _ := attrs.(backend.NormAttrs)
	pa = pa.WithDefaults()
	xc, gc, gupc := x.Contiguous(), gamma.Contiguous(), g.Contiguous()
	dx := tensor.New(tensor.F32, x.Shape())
	dgamma := tensor.New(tensor.F32, gamma.Shape()) // zero-initialized: atomic accumulator
	rc := C.mtl_rmsnorm_backward_f32(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&gupc.Storage().F32()[0]),
		(*C.float)(&dx.Storage().F32()[0]),
		(*C.float)(&dgamma.Storage().F32()[0]),
		C.int(rows), C.int(d), C.float(pa.Eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: rmsnorm-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dx, dgamma}, nil
}

// rmsnormF32 computes RMSNorm on the GPU (§R29): y = x/√(mean(x²)+eps)·γ over the last
// axis, one Metal thread per row. Inputs (x[...,d], gamma[d]); f32-only, falling back to
// the reference (§I4) for empty tensors. Keeps activations on the GPU between matmuls in
// an LLM forward pass instead of bouncing norms through the CPU.
func rmsnormF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpRMSNorm, in, attrs)
	}
	if len(in) != 2 {
		return nil, fmt.Errorf("metal: rmsnorm wants (x, gamma), got %d inputs", len(in))
	}
	x, gamma := in[0], in[1]
	if x.Dtype() != tensor.F32 || gamma.Dtype() != tensor.F32 || x.Ndim() < 1 {
		return refFallback()
	}
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d {
		return nil, fmt.Errorf("metal: rmsnorm gamma must be [%d], got %v", d, gamma.Shape())
	}
	rows := 0
	if d > 0 {
		rows = x.Numel() / d
	}
	if rows == 0 || d == 0 {
		return refFallback() // nothing to launch
	}
	pa, _ := attrs.(backend.NormAttrs)
	pa = pa.WithDefaults()
	xc, gc := x.Contiguous(), gamma.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.mtl_rmsnorm_f32(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(rows), C.int(d), C.float(pa.Eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: rmsnorm failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// ropeKernelF32 returns the GPU kernel for RoPE forward (op=OpRoPE) or its backward
// (op=OpRoPEBackward, the inverse rotation) on the split-half rule (§R28/§R92). Both take
// one rank-2 tensor (q or the upstream gradient g); the per-head inverse frequencies
// (PI/YaRN folded in) and position divisor are precomputed on the host by backend.RoPEFreqs,
// so the forward keeps Q/K resident before attention and the backward keeps the gradient
// resident during backprop. f32-only; the xPos variant (§R125) and empty tensors fall back
// to the reference (§I4).
func ropeKernelF32(op backend.Op, backward bool) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		refFallback := func() ([]*tensor.Tensor, error) {
			return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), op, in, attrs)
		}
		if len(in) != 1 {
			return nil, fmt.Errorf("metal: rope wants 1 input, got %d", len(in))
		}
		q := in[0]
		pr, _ := attrs.(backend.RoPEAttrs)
		if q.Dtype() != tensor.F32 || q.Ndim() != 2 || pr.XPos {
			return refFallback()
		}
		seq, width := q.Shape()[0], q.Shape()[1]
		heads := pr.Heads
		if heads <= 0 {
			heads = 1
		}
		if width%heads != 0 {
			return nil, fmt.Errorf("metal: rope width %d not divisible by heads %d", width, heads)
		}
		hd := width / heads
		if hd%2 != 0 {
			return nil, fmt.Errorf("metal: rope head dim %d must be even", hd)
		}
		half := hd / 2
		if seq == 0 || width == 0 {
			return refFallback()
		}
		inv, posDiv := backend.RoPEFreqs(hd, pr)
		invF32 := make([]float32, half)
		for i := range invF32 {
			invF32[i] = float32(inv[i])
		}
		qc := q.Contiguous()
		out := tensor.New(tensor.F32, q.Shape())
		xp := (*C.float)(&qc.Storage().F32()[0])
		ip := (*C.float)(&invF32[0])
		outp := (*C.float)(&out.Storage().F32()[0])
		var rc C.int
		if backward {
			rc = C.mtl_rope_backward_f32(xp, ip, outp,
				C.int(seq), C.int(width), C.int(heads), C.int(hd), C.int(half), C.int(pr.PosOffset), C.float(posDiv))
		} else {
			rc = C.mtl_rope_f32(xp, ip, outp,
				C.int(seq), C.int(width), C.int(heads), C.int(hd), C.int(half), C.int(pr.PosOffset), C.float(posDiv))
		}
		if rc != 0 {
			return nil, fmt.Errorf("metal: rope failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{out}, nil
	}
}

// embedBackwardF32 computes the embedding gradient on the GPU (§T34): scatter-add g[i,:] into
// dtable[idx[i],:] with float atomics (tokens sharing an index collide). It makes the token/
// position-embedding gradient run on the GPU during training. g must be f32; idx read via
// AtF64 into an f32 buffer (any dtype). Empty/non-f32 g → ref (§I4).
func embedBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpEmbedBackward, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("metal: embed-backward wants (table, idx, g), got %d inputs", len(in))
	}
	table, idx, g := in[0], in[1], in[2]
	if table.Ndim() != 2 || idx.Ndim() != 1 || g.Dtype() != tensor.F32 {
		return refFallback()
	}
	n, d := table.Shape()[0], table.Shape()[1]
	m := idx.Shape()[0]
	if !g.Shape().Equal(tensor.Shape{m, d}) {
		return nil, fmt.Errorf("metal: embed-backward g must be [%d,%d], got %v", m, d, g.Shape())
	}
	if m == 0 || d == 0 || n == 0 {
		return refFallback()
	}
	idxF := make([]float32, m)
	for i := range idxF {
		t := int(idx.AtF64(i))
		if t < 0 || t >= n {
			return nil, fmt.Errorf("metal: embed-backward index %d out of range [0,%d)", t, n)
		}
		idxF[i] = float32(t)
	}
	gc := g.Contiguous()
	dtable := tensor.New(tensor.F32, table.Shape()) // zero-init atomic accumulator
	rc := C.mtl_embed_backward_f32(
		(*C.float)(&idxF[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&dtable.Storage().F32()[0]),
		C.int(n), C.int(d), C.int(m),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: embed-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dtable}, nil
}

// crossentropyBackwardF32 computes the cross-entropy gradient on the GPU (ADR-0007), one
// Metal thread per row: dz = gv·(softmax(z) − q')/b (+z-loss). It makes the loss gradient
// (which seeds the whole backward pass) run on the GPU. Logits z must be f32; targets are
// read via AtF64 into an f32 buffer (any dtype). Empty/non-f32 z → ref (§I4).
// embedF32 is the token-embedding row gather (§T402, host-side since §T410): out[i,:] =
// table[idx[i],:]. §T410's real-workload op profile caught the original GPU version REGRESSING —
// a gather is tiny and memory-bound, and uploading the whole [vocab,dim] table per call (~2.5ms at
// 4096×512) cost more than the entire gather; the old ref fallback (~1ms) was faster. A direct f32
// host copy (~50µs) beats both — no cgo, and no staleness risk when training mutates the table.
func embedF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpEmbed, in, attrs)
	}
	if len(in) != 2 {
		return nil, fmt.Errorf("metal: embed wants (table, idx), got %d inputs", len(in))
	}
	table, idx := in[0], in[1]
	if table.Dtype() != tensor.F32 || table.Ndim() != 2 || idx.Ndim() != 1 {
		return refFallback()
	}
	n, d := table.Shape()[0], table.Shape()[1]
	m := idx.Shape()[0]
	if n == 0 || d == 0 || m == 0 {
		return refFallback()
	}
	ts := table.Contiguous().Storage().F32()
	out := tensor.New(tensor.F32, tensor.Shape{m, d})
	os := out.Storage().F32()
	for i := 0; i < m; i++ {
		r := int(idx.AtF64(i))
		if r < 0 || r >= n {
			return nil, fmt.Errorf("metal: embed index %d out of range [0,%d)", r, n)
		}
		copy(os[i*d:(i+1)*d], ts[r*d:(r+1)*d])
	}
	return []*tensor.Tensor{out}, nil
}

// crossentropyF32 is the fused cross-entropy FORWARD on the GPU (§T401): per row lse−z[target],
// meaned over the batch. Basic case only (label_smoothing=0, z_loss=0) — the smoothed/z-loss forms
// fall back to the reference (§I4). §T401 measured the CPU-fallback forward at ~20ms (~2/3 of the
// GPT forward); this cooperative kernel (one 256-thread threadgroup per [seq,vocab] row) replaces it.
func crossentropyF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpCrossEntropy, in, attrs)
	}
	if len(in) != 2 {
		return nil, fmt.Errorf("metal: crossentropy wants (logits, targets), got %d inputs", len(in))
	}
	z, tg := in[0], in[1]
	if z.Dtype() != tensor.F32 || z.Ndim() != 2 || tg.Ndim() != 1 {
		return refFallback()
	}
	pa, _ := attrs.(backend.CrossEntropyAttrs)
	if !pa.IsBasic() {
		return refFallback() // smoothed / z-loss / masked / non-mean forms stay on the reference
	}
	b, c := z.Shape()[0], z.Shape()[1]
	if tg.Shape()[0] != b {
		return nil, fmt.Errorf("metal: crossentropy targets len %d != batch %d", tg.Shape()[0], b)
	}
	if b == 0 || c == 0 {
		return refFallback()
	}
	tf := make([]float32, b)
	for i := range tf {
		ti := int(tg.AtF64(i))
		if ti < 0 || ti >= c {
			return nil, fmt.Errorf("metal: crossentropy target %d out of range [0,%d)", ti, c)
		}
		tf[i] = float32(ti)
	}
	zc := z.Contiguous()
	losses := make([]float32, b)
	rc := C.mtl_crossentropy_f32(
		(*C.float)(&zc.Storage().F32()[0]),
		(*C.float)(&tf[0]),
		(*C.float)(&losses[0]),
		C.int(b), C.int(c),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: crossentropy failed (code %d)", int(rc))
	}
	var total float64
	for _, l := range losses {
		total += float64(l)
	}
	out := tensor.NewOn(ctx.Device(), z.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

func crossentropyBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpCrossEntropyBackward, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("metal: crossentropy-backward wants (z, targets, g), got %d inputs", len(in))
	}
	z, tg, g := in[0], in[1], in[2]
	if z.Dtype() != tensor.F32 || z.Ndim() != 2 || tg.Ndim() != 1 {
		return refFallback()
	}
	b, c := z.Shape()[0], z.Shape()[1]
	if tg.Shape()[0] != b {
		return nil, fmt.Errorf("metal: crossentropy-backward targets len %d != batch %d", tg.Shape()[0], b)
	}
	if b == 0 || c == 0 {
		return refFallback()
	}
	pa, _ := attrs.(backend.CrossEntropyAttrs)
	if !pa.IsUnmaskedMean() {
		// This kernel folds LabelSmoothing/ZLoss in but has no notion of masked rows or a
		// non-mean reduction (both change the row set AND the divisor) → reference (§I4).
		return refFallback()
	}
	zc := z.Contiguous()
	tf := make([]float32, b) // targets → f32 indices (any tg dtype)
	for i := range tf {
		tf[i] = float32(tg.AtF64(i))
	}
	dz := tensor.New(tensor.F32, z.Shape())
	rc := C.mtl_crossentropy_backward_f32(
		(*C.float)(&zc.Storage().F32()[0]),
		(*C.float)(&tf[0]),
		(*C.float)(&dz.Storage().F32()[0]),
		C.int(b), C.int(c), C.float(g.AtF64()), C.float(pa.LabelSmoothing), C.float(pa.ZLoss),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: crossentropy-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dz}, nil
}

// softmaxF32 computes a numerically-stable softmax over the last axis on the GPU (§V12),
// one Metal thread per row. f32-only; empty tensors fall back to the reference (§I4). Keeps
// logits/scores resident on the GPU (e.g. before sampling) instead of a CPU round-trip.
func softmaxF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpSoftmax, in, attrs)
	}
	if len(in) != 1 {
		return nil, fmt.Errorf("metal: softmax wants 1 input, got %d", len(in))
	}
	x := in[0]
	if x.Dtype() != tensor.F32 || x.Ndim() < 1 {
		return refFallback()
	}
	d := x.Shape()[x.Ndim()-1]
	rows := 0
	if d > 0 {
		rows = x.Numel() / d
	}
	if rows == 0 || d == 0 {
		return refFallback()
	}
	xc := x.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.mtl_softmax_f32(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(rows), C.int(d),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: softmax failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// layernormF32 computes LayerNorm on the GPU (torch semantics): y = (x−mean)/√(var+eps)·γ + β
// over the last axis, one Metal thread per row. Inputs (x[...,d], gamma[d], beta[d]); f32-only,
// falling back to the reference (§I4) for empty tensors. Complements the GPU RMSNorm (§T324).
func layernormF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpLayerNorm, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("metal: layernorm wants (x, gamma, beta), got %d inputs", len(in))
	}
	x, gamma, beta := in[0], in[1], in[2]
	if x.Dtype() != tensor.F32 || gamma.Dtype() != tensor.F32 || beta.Dtype() != tensor.F32 || x.Ndim() < 1 {
		return refFallback()
	}
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d || beta.Ndim() != 1 || beta.Shape()[0] != d {
		return nil, fmt.Errorf("metal: layernorm gamma/beta must be [%d], got %v/%v", d, gamma.Shape(), beta.Shape())
	}
	rows := 0
	if d > 0 {
		rows = x.Numel() / d
	}
	if rows == 0 || d == 0 {
		return refFallback()
	}
	pa, _ := attrs.(backend.NormAttrs)
	pa = pa.WithDefaults()
	xc, gc, bc := x.Contiguous(), gamma.Contiguous(), beta.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.mtl_layernorm_f32(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&bc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(rows), C.int(d), C.float(pa.Eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: layernorm failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// conv2dBackwardF32 is the conv2d backward on the GPU (§T101): (X,W,dO) → (dX,dW,dBias).
// Compute-bound (§C3); one thread per output-gradient element scatters into the shared
// dX/dW/dBias with float atomics. f32-only; the reference serves other dtypes via the
// dispatch fallback (§I4). Dispatched by conv2d's VJP → CV training runs on the GPU.
func conv2dBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("metal: conv2d-backward wants (X,W,dO), got %d inputs", len(in))
	}
	x, w, dO := in[0], in[1], in[2]
	for _, t := range in {
		if t.Ndim() != 4 || t.Dtype() != tensor.F32 {
			return nil, fmt.Errorf("metal: conv2d-backward needs rank-4 f32")
		}
	}
	n, c, h, wd := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	f, wc, kh, kw := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]
	if wc != c {
		return nil, fmt.Errorf("metal: conv2d-backward channel mismatch x C=%d vs w C=%d", c, wc)
	}
	ho, wo := dO.Shape()[2], dO.Shape()[3]
	pa, _ := attrs.(backend.ConvAttrs)
	pa = pa.WithDefaults()

	xc, wcont, dc := x.Contiguous(), w.Contiguous(), dO.Contiguous()
	dX := tensor.New(tensor.F32, x.Shape())
	dW := tensor.New(tensor.F32, w.Shape())
	dB := tensor.New(tensor.F32, tensor.Shape{f})
	rc := C.mtl_conv2d_backward_f32(
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&wcont.Storage().F32()[0]),
		(*C.float)(&dc.Storage().F32()[0]),
		(*C.float)(&dX.Storage().F32()[0]),
		(*C.float)(&dW.Storage().F32()[0]),
		(*C.float)(&dB.Storage().F32()[0]),
		C.int(n), C.int(c), C.int(h), C.int(wd), C.int(f), C.int(kh), C.int(kw),
		C.int(pa.Stride), C.int(pa.Pad), C.int(ho), C.int(wo),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: conv2d-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dX, dW, dB}, nil
}

// conv2dF32 is 2-D convolution on the GPU (§T88): X[N,C,H,W] ⋆ W[F,C,KH,KW] (+
// optional per-filter bias) → [N,F,ho,wo]. Compute-bound (O(N·F·ho·wo·C·KH·KW)),
// so it clears the §C3 offload gate. f32-only; the reference serves other dtypes
// via the dispatch fallback (§I4). Forward only — the conv backward stays on the
// reference for now.
func conv2dF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 && len(in) != 3 {
		return nil, fmt.Errorf("metal: conv2d wants (x, w[, bias]), got %d inputs", len(in))
	}
	x, w := in[0], in[1]
	if x.Ndim() != 4 || w.Ndim() != 4 {
		return nil, fmt.Errorf("metal: conv2d needs x[N,C,H,W] and w[F,C,KH,KW], got %v/%v", x.Shape(), w.Shape())
	}
	if x.Dtype() != tensor.F32 || w.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: conv2d is f32-only")
	}
	n, c, h, wd := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	f, wc, kh, kw := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]
	if wc != c {
		return nil, fmt.Errorf("metal: conv2d channel mismatch x C=%d vs w C=%d", c, wc)
	}
	// per-filter bias: use the caller's, else a zero vector (kernel always adds B[f]).
	bias := make([]float32, f)
	if len(in) == 3 {
		b := in[2]
		if b.Ndim() != 1 || b.Shape()[0] != f || b.Dtype() != tensor.F32 {
			return nil, fmt.Errorf("metal: conv2d bias must be f32 [%d], got %v", f, b.Shape())
		}
		copy(bias, b.Contiguous().Storage().F32())
	}
	pa, _ := attrs.(backend.ConvAttrs)
	pa = pa.WithDefaults()
	s, p := pa.Stride, pa.Pad
	if s < 1 || p < 0 {
		return nil, fmt.Errorf("metal: conv2d invalid stride %d / pad %d", s, p)
	}
	ho := (h+2*p-kh)/s + 1
	wo := (wd+2*p-kw)/s + 1
	if ho < 1 || wo < 1 {
		return nil, fmt.Errorf("metal: conv2d output would be empty (%dx%d)", ho, wo)
	}

	xc, wcont := x.Contiguous(), w.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{n, f, ho, wo})
	xp := (*C.float)(&xc.Storage().F32()[0])
	wp := (*C.float)(&wcont.Storage().F32()[0])
	bp := (*C.float)(&bias[0])
	op := (*C.float)(&out.Storage().F32()[0])
	var rc C.int
	if convUseMPS {
		// Apple's native MPSGraph convolution (§T620) — the live path.
		rc = C.mtl_conv2d_mps_f32(xp, wp, bp, op,
			C.int(n), C.int(c), C.int(h), C.int(wd), C.int(f), C.int(kh), C.int(kw),
			C.int(s), C.int(p), C.int(ho), C.int(wo))
	} else {
		// im2col + MPS-GEMM lowering — kept as a fallback and for A/B measurement.
		rc = C.mtl_conv2d_f32(xp, wp, bp, op,
			C.int(n), C.int(c), C.int(h), C.int(wd), C.int(f), C.int(kh), C.int(kw),
			C.int(s), C.int(p), C.int(ho), C.int(wo))
	}
	if rc != 0 {
		return nil, fmt.Errorf("metal: conv2d failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// convUseMPS routes conv2d forward through Apple's native MPSGraph convolution
// (§T620) instead of the im2col+MPS-GEMM lowering. Native MPS is the default
// (measured faster on the benchmarked shapes); the im2col path stays reachable as
// a fallback and for same-machine A/B measurement (metal_test.go toggles this).
var convUseMPS = true

// SetConvCacheCap sets the effective shape-keyed conv-graph cache cap (§T622), clears all
// resident graphs, and returns the previous cap. cap is clamped to [1,16]; cap==1 reproduces
// the pre-§T622 single last-shape cache (recompile per shape). Benchmark/A-B knob (see the
// Conv2DMultiShape benchmarks) — not part of the stable op surface.
func SetConvCacheCap(cap int) int { return int(C.mtl_conv_cache_cap_set(C.int(cap))) }

// setMatMulNoCopy selects zero-copy operand wrapping (true, default) or the pooled-copy
// path (false) for the standalone matmul, returning the previous value. Interleaved
// same-process A/B measurement hook (internal tests only).
func setMatMulNoCopy(on bool) bool {
	v := 0
	if on {
		v = 1
	}
	return C.mtl_matmul_nocopy_set(C.int(v)) == 1
}

// SetAttnCacheCap sets the effective shape-keyed attention-graph cache cap (§T622), clears all
// resident graphs, and returns the previous cap. cap is clamped to [1,16]; cap==1 reproduces the
// pre-§T622 single last-shape cache (recompile per prefill length). Benchmark/A-B knob.
func SetAttnCacheCap(cap int) int { return int(C.mtl_attn_cache_cap_set(C.int(cap))) }

// SetQ4KCooperative selects the simdgroup-cooperative resident Q4_K matvec
// (true, default) or the historical one-thread-per-output kernel (false), and
// returns the previous setting. It is a same-process forced-off measurement
// hook; devices that cannot compile the cooperative pipeline keep the scalar
// fallback regardless of this setting.
func SetQ4KCooperative(on bool) bool {
	v := 0
	if on {
		v = 1
	}
	return C.mtl_q4k_cooperative_set(C.int(v)) == 1
}

// SetQ4_0Cooperative selects the SIMD-group-cooperative resident Q4_0 M=1 matvec and
// returns the previous setting. Lanes 0-15 take the low nibble of block byte L and
// lanes 16-31 the high nibble of byte L-16, matching Q4_0's x[i]/x[i+16] packing, so
// each lane owns one element per block. Exposed for the A/B control.
func SetQ4_0Cooperative(on bool) bool {
	v := 0
	if on {
		v = 1
	}
	return C.mtl_q4_0_cooperative_set(C.int(v)) == 1
}

// SetQ8_0Cooperative selects the SIMD-group-cooperative resident Q8_0 M=1 matvec and
// returns the previous setting. Exposed for the A/B control; see the kernel comment
// for why Q8_0 is a boundary case for the cooperative transform.
func SetQ8_0Cooperative(on bool) bool {
	v := 0
	if on {
		v = 1
	}
	return C.mtl_q8_0_cooperative_set(C.int(v)) == 1
}

// SetQ2KCooperative selects the SIMD-group-cooperative resident Q2_K M=1 matvec and
// returns the previous setting. Q2_K shares Q3_K's 16-scale-group layout, so the 32
// lanes of a SIMD group pair up two-per-group and each lane stays inside one scale.
// Devices without a 32-lane SIMD width keep the scalar path, and this hook forces it
// off for the A/B control.
//
// Results match the scalar kernel within the 2e-5 relative bar the sibling
// cooperative kernels use: the per-element arithmetic is identical, only the
// summation order differs.
func SetQ2KCooperative(on bool) bool {
	v := 0
	if on {
		v = 1
	}
	return C.mtl_q2k_cooperative_set(C.int(v)) == 1
}

// SetQ3KCooperative selects the SIMD-group-cooperative resident Q3_K M=1 matvec and
// returns the previous setting. Q3_K splits a 256-superblock into 16 scale groups of
// 16 elements, so the 32 lanes of a SIMD group pair up two-per-group and each lane
// stays inside one scale. Devices without a 32-lane SIMD width keep the scalar path,
// and this hook forces it off for the A/B control.
//
// Results match the scalar kernel within the 2e-5 relative bar the sibling
// cooperative kernels use: the per-element arithmetic is identical, only the
// summation order differs.
func SetQ3KCooperative(on bool) bool {
	v := 0
	if on {
		v = 1
	}
	return C.mtl_q3k_cooperative_set(C.int(v)) == 1
}

// SetQ5KCooperative selects the SIMD-group-cooperative resident Q5_K M=1 matvec
// and returns the previous setting. At M=1 the scalar kernel gives work to only N
// threads, each walking all of K; the cooperative kernel gives one output row to a
// SIMD group whose 32 lanes split that row's K. Devices without a 32-lane SIMD width
// keep the scalar path, and this hook forces it off for the A/B control.
//
// Results match the scalar kernel within the 2e-5 relative bar the Q4_K and Q6_K
// cooperative kernels use — the per-element arithmetic is identical, only the
// summation order differs (per-lane partials reduced by simd_sum).
func SetQ5KCooperative(on bool) bool {
	v := 0
	if on {
		v = 1
	}
	return C.mtl_q5k_cooperative_set(C.int(v)) == 1
}

// SetQ6KCooperative selects the SIMD-group-cooperative resident Q6_K M=1
// matvec (true, default) or its historical scalar-K control (false), returning
// the previous setting. M>1 and unsupported devices always retain scalar.
func SetQ6KCooperative(on bool) bool {
	v := 0
	if on {
		v = 1
	}
	return C.mtl_q6k_cooperative_set(C.int(v)) == 1
}

// mhaBackwardF32 is the GPU SDPA backward (§T86): (Q,K,V,dO) → (dQ,dK,dV). It
// serves the same subset as the forward (heads, GQA, causal, sliding window §T128,
// attn-scale) and falls back to the reference (§I4) for ALiBi/dk>128/degenerate
// shapes, non-f32, or a GPU without float-atomic support. Dispatched by the mha VJP,
// so attention training runs on the GPU when the tape's backend is metal.
func mhaBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMHABackward, in, attrs)
	}
	if len(in) != 4 {
		return nil, fmt.Errorf("metal: mha-backward wants (Q,K,V,dO), got %d inputs", len(in))
	}
	q, k, v, dO := in[0], in[1], in[2], in[3]
	for _, x := range in {
		if x.Ndim() != 2 || x.Dtype() != tensor.F32 {
			return nil, fmt.Errorf("metal: mha-backward needs rank-2 f32")
		}
	}
	p, _ := attrs.(backend.AttnAttrs)
	p = p.WithDefaults()

	seq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	if sk != seq {
		return refFallback() // KV-cache is inference-only; ref returns the clear error
	}
	heads := p.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("metal: mha-backward dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := p.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("metal: mha-backward heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	dkv := kvHeads * dk
	if k.Shape()[1] != dkv || !v.Shape().Equal(k.Shape()) || !dO.Shape().Equal(q.Shape()) {
		return nil, fmt.Errorf("metal: mha-backward shape mismatch")
	}
	if p.ALiBi || dk > 128 || seq == 0 {
		return refFallback()
	}

	scale := p.Scale / math.Sqrt(float64(dk))
	causal := 0
	if p.Causal {
		causal = 1
	}
	qc, kc, vc, dc := q.Contiguous(), k.Contiguous(), v.Contiguous(), dO.Contiguous()
	dQ := tensor.New(tensor.F32, q.Shape())
	dK := tensor.New(tensor.F32, k.Shape())
	dV := tensor.New(tensor.F32, v.Shape())
	if p.Window == 0 {
		// MPS-matmul backward (§T399, GQA §T400): dV=Pᵀ·dO, dP=dO·Vᵀ, dS=softmax-jacobian(P,dP),
		// dQ=scale·dS·K, dK=scale·dSᵀ·Q — the backward analog of the §T394 forward. Measured 27.3× over
		// the hand flash-backward (75→2.75ms at 512×8×64), the GPT training-step bottleneck (§T399: bwd
		// was 54× the fwd; wiring → training 2.04×). GQA dK/dV accumulate via MPS beta=1 (§T400).
		// Sliding-window stays on the atomic flash-backward below.
		rc := C.mtl_mha_backward_mps(
			(*C.float)(&qc.Storage().F32()[0]),
			(*C.float)(&kc.Storage().F32()[0]),
			(*C.float)(&vc.Storage().F32()[0]),
			(*C.float)(&dc.Storage().F32()[0]),
			(*C.float)(&dQ.Storage().F32()[0]),
			(*C.float)(&dK.Storage().F32()[0]),
			(*C.float)(&dV.Storage().F32()[0]),
			C.int(seq), C.int(dm), C.int(heads), C.int(dk), C.int(causal), C.int(kvHeads), C.float(scale),
		)
		if rc != 0 {
			return nil, fmt.Errorf("metal: mha-backward (MPS path) failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{dQ, dK, dV}, nil
	}
	rc := C.mtl_mha_backward_f32(
		(*C.float)(&qc.Storage().F32()[0]),
		(*C.float)(&kc.Storage().F32()[0]),
		(*C.float)(&vc.Storage().F32()[0]),
		(*C.float)(&dc.Storage().F32()[0]),
		(*C.float)(&dQ.Storage().F32()[0]),
		(*C.float)(&dK.Storage().F32()[0]),
		(*C.float)(&dV.Storage().F32()[0]),
		C.int(seq), C.int(sk), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk),
		C.int(causal), C.int(p.Window), C.float(scale),
	)
	if int(rc) == -5 {
		return refFallback() // GPU lacks float-atomic pipeline
	}
	if rc != 0 {
		return nil, fmt.Errorf("metal: mha-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dQ, dK, dV}, nil
}

// mhaF32 is fused multi-head scaled-dot-product attention on the GPU (§T85, the
// forward of §T32). It serves the common case — heads, GQA/MQA kv_heads, causal
// mask, sliding-window attention (§T114) and the YaRN attn-scale — and honestly
// falls back to the reference backend (§I4) for the features the kernel does not
// implement (ALiBi bias), for a per-head dim dk>128, or for degenerate shapes.
// Training/prefill shapes (sq==sk, no window) route to the single-pass
// FlashAttention kernel — same math, ~1.6× faster (§T340).
func mhaF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("metal: mha wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("metal: mha needs rank-2 [seq,dmodel]")
	}
	if q.Dtype() != tensor.F32 || k.Dtype() != tensor.F32 || v.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: mha is f32-only")
	}
	p, _ := attrs.(backend.AttnAttrs)
	p = p.WithDefaults()

	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	heads := p.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("metal: mha dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := p.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("metal: mha heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	dkv := kvHeads * dk
	if k.Shape()[1] != dkv || !v.Shape().Equal(k.Shape()) {
		return nil, fmt.Errorf("metal: mha K/V must be [sk,%d], got %v/%v", dkv, k.Shape(), v.Shape())
	}
	if sq > sk {
		return nil, fmt.Errorf("metal: mha query len %d exceeds key len %d", sq, sk)
	}
	// Unsupported feature or shape → reference backend, same result (§I4).
	if p.ALiBi || dk > 128 || sq == 0 || sk == 0 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMHA, in, attrs)
	}

	scale := p.Scale / math.Sqrt(float64(dk))
	causal := 0
	if p.Causal {
		causal = 1
	}
	qc, kc, vc := q.Contiguous(), k.Contiguous(), v.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{sq, dm})
	if sq == sk && p.Window == 0 {
		// Training/prefill fast path: all heads batched in ONE MPSGraph run (Q·Kᵀ → causal
		// softmax → P·V). Same scale/causal/kvHeads, so YaRN attn-scale and GQA carry over.
		// §B56 FIX: the earlier default (mtl_mha_mps, per-head strided MPSMatrix two-matmul with a
		// materialized [seq,seq] scratch) returns NON-DETERMINISTIC WRONG outputs at seq≥≈512 —
		// isolation ruled out scratch reuse AND encoder ordering (per-head/separate scratch buffers,
		// per-head kernels, and fully serialized per-stage command buffers all still reproduced it);
		// the strided-matmul path itself misbehaves at scale. MPSGraph (contiguous, no shared scratch)
		// is exact at every shape (seq 4…1024, verified by TestMetalMHAMPSGraphCrossReference) and
		// race-free, so ALL prefill now routes through it. The mhaUseMPSGraph toggle is retained for
		// the A/B tests but no longer gates correctness here. Decode (sq<sk) and sliding-window stay
		// on the two-pass kernel below.
		rc := C.mtl_mha_mpsgraph(
			(*C.float)(&qc.Storage().F32()[0]),
			(*C.float)(&kc.Storage().F32()[0]),
			(*C.float)(&vc.Storage().F32()[0]),
			(*C.float)(&out.Storage().F32()[0]),
			C.int(sq), C.int(dm), C.int(heads), C.int(dk), C.int(causal), C.int(kvHeads), C.float(scale),
		)
		if rc != 0 {
			return nil, fmt.Errorf("metal: mha (MPSGraph path) failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{out}, nil
	}
	// Cooperative decode/prefill path (§T432, the per-op twin of the recorder's §T428/431 route):
	// the two-pass kernel below runs ONE thread per (head, query row) — ~serial in sk at small sq,
	// which made per-op KV-cache decode collapse at long contexts. Window stays on the two-pass.
	if p.Window == 0 && dk <= 128 && (causal != 0 || sq == 1) {
		rc := C.mtl_mha_decode_host(
			(*C.float)(&qc.Storage().F32()[0]),
			(*C.float)(&kc.Storage().F32()[0]),
			(*C.float)(&vc.Storage().F32()[0]),
			(*C.float)(&out.Storage().F32()[0]),
			C.int(sq), C.int(sk), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk),
			C.int(causal), C.float(scale),
		)
		if rc != 0 {
			return nil, fmt.Errorf("metal: mha (decode path) failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{out}, nil
	}
	rc := C.mtl_mha_f32(
		(*C.float)(&qc.Storage().F32()[0]),
		(*C.float)(&kc.Storage().F32()[0]),
		(*C.float)(&vc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(sq), C.int(sk), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk),
		C.int(causal), C.int(p.Window), C.float(scale),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: mha failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// flashAttnF32 is FlashAttention-2 forward on the GPU (§T109, the GPU backend of the
// reference §T71 / §R72). One thread per (head, query row) streams the keys with the
// online-softmax recurrence, so — unlike mhaF32's two-pass softmax — no [seq,seq]
// score row is ever materialized (the flash memory win, which matters most on the
// GPU). It serves heads, GQA/MQA kv_heads and causal for Q [seq,dmodel], K,V
// [seq,kv_heads·dk] (sq==sk), and falls back to the reference (§I4) for a per-head dim
// dk>128 or degenerate shapes. Exact: equals mhaF32 up to float reassociation (verified
// by V-CROSS, §V3/§V11); block-invariant, so it ignores the Block hint.
func flashAttnF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("metal: flashattn wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("metal: flashattn needs rank-2 [seq,dmodel]")
	}
	if q.Dtype() != tensor.F32 || k.Dtype() != tensor.F32 || v.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: flashattn is f32-only")
	}
	seq, dm := q.Shape()[0], q.Shape()[1]
	p, _ := attrs.(backend.AttnAttrs)
	p = p.WithDefaults()
	heads := p.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("metal: flashattn dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := p.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("metal: flashattn heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	dkv := kvHeads * dk
	// Q [seq,dm]; K,V [seq,kvHeads*dk] (GQA); FlashAttention requires sq==sk.
	if k.Shape()[0] != seq || !v.Shape().Equal(k.Shape()) || k.Shape()[1] != dkv {
		return nil, fmt.Errorf("metal: flashattn needs Q [seq,%d], K,V [seq,%d] (sq==sk), got %v/%v/%v", dm, dkv, q.Shape(), k.Shape(), v.Shape())
	}
	// Unsupported dim or degenerate shape → reference backend, same result (§I4).
	if dk > 128 || seq == 0 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpFlashAttn, in, attrs)
	}
	scale := 1 / math.Sqrt(float64(dk)) // reference flashattn scale (no YaRN attn_scale)
	causal := 0
	if p.Causal {
		causal = 1
	}
	qc, kc, vc := q.Contiguous(), k.Contiguous(), v.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{seq, dm})
	rc := C.mtl_flashattn_f32(
		(*C.float)(&qc.Storage().F32()[0]),
		(*C.float)(&kc.Storage().F32()[0]),
		(*C.float)(&vc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(seq), C.int(dm), C.int(heads), C.int(dk), C.int(causal), C.int(kvHeads), C.float(scale),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: flashattn failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// retentionF32 runs the RetNet retention forward (§T172) on the GPU: O[L,d] = (QKᵀ ⊙ D)·V with the
// γ-decay mask D_nm=γ^(n−m). Q,K,V are [L,d] (a single head). One GPU thread per query row streams
// keys with a register acc[d]; falls back to the reference (§I4) for d>128, non-f32, or empty L. The
// heads/GroupNorm/gate of multi-scale retention are a block-level concern (this is the operator).
func retentionF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("metal: retention wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("metal: retention needs rank-2 [L,d]")
	}
	l, dk := q.Shape()[0], q.Shape()[1] // query/key dim
	dv := v.Shape()[1]                  // value dim (may differ — RetNet value expansion, §T178)
	if k.Shape()[0] != l || v.Shape()[0] != l || k.Shape()[1] != dk {
		return nil, fmt.Errorf("metal: retention needs Q,K [L,dk], V [L,dv]; got Q%v K%v V%v", q.Shape(), k.Shape(), v.Shape())
	}
	pa, _ := attrs.(backend.RetentionAttrs)
	// Unsupported dim / degenerate / non-f32 → reference (§I4). The kernel's register acc is [dv]
	// (must be ≤128); dk is unbounded (streamed, no per-key array). d_v≠d_k is native (§T178).
	if dv > 128 || l == 0 || q.Dtype() != tensor.F32 || k.Dtype() != tensor.F32 || v.Dtype() != tensor.F32 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpRetention, in, attrs)
	}
	qc, kc, vc := q.Contiguous(), k.Contiguous(), v.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{l, dv})
	rc := C.mtl_retention_f32(
		(*C.float)(&qc.Storage().F32()[0]),
		(*C.float)(&kc.Storage().F32()[0]),
		(*C.float)(&vc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(l), C.int(dk), C.int(dv), C.float(pa.Gamma),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: retention failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// retentionBackwardF32 runs the RetNet retention backward (§T174/§T191) on the GPU: (Q,K [L,dk], V,dO
// [L,dv]) → (dQ,dK [L,dk], dV [L,dv]). One thread per query row writes dQ directly and atomically
// accumulates dK/dV — so a Metal tape trains retention on-device, including the value-expanded
// (d_v≠d_k) RetNet block. Falls back to the reference (§I4) for dk>128/empty/non-f32.
func retentionBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("metal: retention-backward wants (Q,K,V,dO), got %d inputs", len(in))
	}
	q, k, v, dO := in[0], in[1], in[2], in[3]
	for _, t := range in {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("metal: retention-backward needs rank-2 tensors")
		}
	}
	l, dk := q.Shape()[0], q.Shape()[1] // query/key dim
	dv := v.Shape()[1]                  // value dim (may differ — RetNet value expansion, §T191)
	if k.Shape()[0] != l || k.Shape()[1] != dk || v.Shape()[0] != l || dO.Shape()[0] != l || dO.Shape()[1] != dv {
		return nil, fmt.Errorf("metal: retention-backward needs Q,K [L,dk], V,dO [L,dv]; got Q%v K%v V%v dO%v", q.Shape(), k.Shape(), v.Shape(), dO.Shape())
	}
	pa, _ := attrs.(backend.RetentionAttrs)
	// Unsupported dim / non-f32 → reference (§I4). The kernel's register dQ accumulator is [dk]
	// (must be ≤128); dv is streamed (unbounded). d_v≠d_k is native (§T191).
	if dk > 128 || l == 0 || q.Dtype() != tensor.F32 || k.Dtype() != tensor.F32 || v.Dtype() != tensor.F32 || dO.Dtype() != tensor.F32 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpRetentionBackward, in, attrs)
	}
	qc, kc, vc, dc := q.Contiguous(), k.Contiguous(), v.Contiguous(), dO.Contiguous()
	gq := tensor.New(tensor.F32, tensor.Shape{l, dk})
	gk := tensor.New(tensor.F32, tensor.Shape{l, dk})
	gv := tensor.New(tensor.F32, tensor.Shape{l, dv})
	rc := C.mtl_retention_backward_f32(
		(*C.float)(&qc.Storage().F32()[0]),
		(*C.float)(&kc.Storage().F32()[0]),
		(*C.float)(&vc.Storage().F32()[0]),
		(*C.float)(&dc.Storage().F32()[0]),
		(*C.float)(&gq.Storage().F32()[0]),
		(*C.float)(&gk.Storage().F32()[0]),
		(*C.float)(&gv.Storage().F32()[0]),
		C.int(l), C.int(dk), C.int(dv), C.float(pa.Gamma),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: retention-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{gq, gk, gv}, nil
}

func matmulF32(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("metal: matmul wants 2 inputs, got %d", len(in))
	}
	a, b := in[0], in[1]
	if a.Ndim() != 2 || b.Ndim() != 2 {
		return nil, fmt.Errorf("metal: matmul needs rank-2, got %dD and %dD", a.Ndim(), b.Ndim())
	}
	if a.Dtype() != tensor.F32 || b.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("metal: matmul is f32-only, got %v/%v", a.Dtype(), b.Dtype())
	}
	m, k := a.Shape()[0], a.Shape()[1]
	k2, n := b.Shape()[0], b.Shape()[1]
	if k != k2 {
		return nil, fmt.Errorf("metal: inner dim mismatch %v · %v", a.Shape(), b.Shape())
	}
	if m == 0 || n == 0 || k == 0 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMatMul, in, nil)
	}
	// A transposed VIEW of a contiguous matrix (the matmul backward's dO·Wᵀ, Xᵀ·dO) is
	// handed to MPS as a transpose FLAG on the contiguous base, instead of materializing
	// it with a CPU strided-gather copy (§T356, ~8ms/operand). transposedBase returns the
	// contiguous base + true when x is exactly a 2-D transpose; otherwise a Contiguous copy.
	ab, transA := transposedBase(a)
	bb, transB := transposedBase(b)
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	rc := C.mtl_matmul_f32(
		(*C.float)(&ab.Storage().F32()[0]),
		(*C.float)(&bb.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
		boolToCInt(transA), boolToCInt(transB),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: matmul failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// transposedBase reports whether x is exactly a 2-D transposed view of a contiguous,
// offset-0 matrix. If so it returns that contiguous base (whose data MPS reads with a
// transpose flag — no copy); otherwise it returns x.Contiguous() and false.
func transposedBase(x *tensor.Tensor) (*tensor.Tensor, bool) {
	if !x.IsContiguous() {
		if xt, err := x.Transpose(0, 1); err == nil && xt.IsContiguous() && xt.Offset() == 0 {
			return xt, true
		}
	}
	return x.Contiguous(), false
}

func boolToCInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// DeviceBuffer is a host-visible GPU-resident buffer (ADR-0019 Phase 1, §T367): f32
// data uploaded into a retained MTLBuffer, read back with DownloadF32, freed with
// Release. On Apple Silicon UMA it is host memory. Nothing dispatches over it yet — it
// is the storage primitive the batched-decode path (Phase 2) will chain ops over so
// intermediates stay on the GPU instead of a host round-trip per op.
type DeviceBuffer struct {
	handle unsafe.Pointer
	n      int // element count (f32)
}

// NewDeviceBufferF32 uploads data into a device-resident buffer.
func NewDeviceBufferF32(data []float32) (*DeviceBuffer, error) {
	if !Available() {
		return nil, fmt.Errorf("metal: no GPU (§V4)")
	}
	if len(data) == 0 {
		return &DeviceBuffer{}, nil
	}
	h := C.mtl_devbuf_upload(unsafe.Pointer(&data[0]), C.int(len(data)*4))
	if h == nil {
		return nil, fmt.Errorf("metal: devbuf upload failed")
	}
	b := &DeviceBuffer{handle: h, n: len(data)}
	runtime.SetFinalizer(b, (*DeviceBuffer).Release) // safety net; callers should Release explicitly
	return b, nil
}

// Len returns the element count.
func (b *DeviceBuffer) Len() int { return b.n }

// DownloadF32 copies the leading len(dst) elements of the buffer into dst (a prefix download —
// over-allocated scratch buffers, §T418, hand back just the rows a step produced). len(dst) must
// be ≤ Len().
func (b *DeviceBuffer) DownloadF32(dst []float32) error {
	if b.handle == nil || b.n == 0 || len(dst) == 0 {
		return nil
	}
	if len(dst) > b.n {
		return fmt.Errorf("metal: devbuf download dst len %d > %d", len(dst), b.n)
	}
	rc := C.mtl_devbuf_download(b.handle, unsafe.Pointer(&dst[0]), C.int(len(dst)*4))
	if rc != 0 {
		return fmt.Errorf("metal: devbuf download failed (code %d)", int(rc))
	}
	return nil
}

// UploadF32 overwrites the buffer contents with src (host→device), reusing the buffer across
// steps without reallocating. src must hold at most Len() elements.
func (b *DeviceBuffer) UploadF32(src []float32) error {
	if b.handle == nil || len(src) == 0 {
		return nil
	}
	if len(src) > b.n {
		return fmt.Errorf("metal: devbuf upload src len %d > %d", len(src), b.n)
	}
	rc := C.mtl_devbuf_upload_into(b.handle, unsafe.Pointer(&src[0]), C.int(len(src)*4))
	if rc != 0 {
		return fmt.Errorf("metal: devbuf upload failed (code %d)", int(rc))
	}
	return nil
}

// Release frees the device buffer. Safe to call more than once.
func (b *DeviceBuffer) Release() {
	if b.handle != nil {
		C.mtl_devbuf_free(b.handle)
		b.handle = nil
	}
}

// batchDispatchBench runs `iters` trivial GPU dispatches, either as `iters` separate
// command buffers (oneBuffer=false, the current per-op model) or one command buffer
// (oneBuffer=true, the batched model). Used by the §T368 measurement to validate
// ADR-0019's premise that batching recovers the per-op round-trip overhead.
func batchDispatchBench(iters int, oneBuffer bool) {
	one := 0
	if oneBuffer {
		one = 1
	}
	C.mtl_batch_dispatch_bench(C.int(iters), C.int(one))
}

// mhaMPS runs the §T394 MPS-matmul attention forward over host slices (unexported; the cross-check
// + A/B live in internal tests). Q,O are [seq,dm]; K,V are [seq,kvHeads*dk].
func mhaMPS(q, k, v []float32, seq, dm, heads, dk, causal, kvHeads int, scale float32) ([]float32, error) {
	o := make([]float32, seq*dm)
	rc := C.mtl_mha_mps(
		(*C.float)(&q[0]), (*C.float)(&k[0]), (*C.float)(&v[0]), (*C.float)(&o[0]),
		C.int(seq), C.int(dm), C.int(heads), C.int(dk), C.int(causal), C.int(kvHeads), C.float(scale))
	if rc != 0 {
		return nil, fmt.Errorf("metal: mha_mps failed (%d)", int(rc))
	}
	return o, nil
}

// mhaBackwardMPS runs the §T399 MPS-matmul attention backward over host slices (unexported; the
// cross-check + A/B live in internal tests). All tensors [seq,dm]; MHA only (kvHeads==heads).
func mhaBackwardMPS(q, k, v, dO []float32, seq, dm, heads, dk, causal, kvHeads int, scale float32) (dQ, dK, dV []float32, err error) {
	kvLen := seq * kvHeads * dk
	dQ = make([]float32, seq*dm)
	dK = make([]float32, kvLen)
	dV = make([]float32, kvLen)
	rc := C.mtl_mha_backward_mps(
		(*C.float)(&q[0]), (*C.float)(&k[0]), (*C.float)(&v[0]), (*C.float)(&dO[0]),
		(*C.float)(&dQ[0]), (*C.float)(&dK[0]), (*C.float)(&dV[0]),
		C.int(seq), C.int(dm), C.int(heads), C.int(dk), C.int(causal), C.int(kvHeads), C.float(scale))
	if rc != 0 {
		return nil, nil, nil, fmt.Errorf("metal: mha_backward_mps failed (%d)", int(rc))
	}
	return dQ, dK, dV, nil
}

// mhaBackwardFlashHost runs the hand-written flash-backward over host slices (unexported; the §T399
// A/B baseline + cross-check oracle, itself ref-validated). All tensors [seq,dm]; kvHeads==heads.
func mhaBackwardFlashHost(q, k, v, dO []float32, seq, dm, heads, dk, causal, kvHeads int, scale float32) (dQ, dK, dV []float32, err error) {
	kvLen := seq * kvHeads * dk
	dQ = make([]float32, seq*dm)
	dK = make([]float32, kvLen)
	dV = make([]float32, kvLen)
	rc := C.mtl_mha_backward_f32(
		(*C.float)(&q[0]), (*C.float)(&k[0]), (*C.float)(&v[0]), (*C.float)(&dO[0]),
		(*C.float)(&dQ[0]), (*C.float)(&dK[0]), (*C.float)(&dV[0]),
		C.int(seq), C.int(seq), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk), C.int(causal), C.int(0), C.float(scale))
	if rc != 0 {
		return nil, nil, nil, fmt.Errorf("metal: mha_backward flash failed (%d)", int(rc))
	}
	return dQ, dK, dV, nil
}

// flashAttnHost runs the hand-written flash attention kernel over host slices (unexported; the
// §T394 A/B baseline). Q,O are [seq,dm]; K,V are [seq,kvHeads*dk].
func flashAttnHost(q, k, v []float32, seq, dm, heads, dk, causal, kvHeads int, scale float32) ([]float32, error) {
	o := make([]float32, seq*dm)
	rc := C.mtl_flashattn_f32(
		(*C.float)(&q[0]), (*C.float)(&k[0]), (*C.float)(&v[0]), (*C.float)(&o[0]),
		C.int(seq), C.int(dm), C.int(heads), C.int(dk), C.int(causal), C.int(kvHeads), C.float(scale))
	if rc != 0 {
		return nil, fmt.Errorf("metal: flashattn failed (%d)", int(rc))
	}
	return o, nil
}

// Recorder wraps one open Metal command buffer that record-mode ops (MatMul, RMSNorm, LayerNorm,
// RoPE, MHA, Unary, Binary, Blit) encode into over DeviceBuffers without a per-op commit/wait;
// Finish submits once. Metal's hazard tracking orders dependent buffer accesses within the buffer,
// so intermediates stay device-resident across a whole decode step (§T370, ADR-0019 Phase 2 — the
// batched-decode engine that is ~3× faster than per-op dispatch at realistic decode sizes, §T379).
type Recorder struct{ handle unsafe.Pointer }

// NewRecorder opens a Recorder (a fresh command buffer). Record ops into it, then Finish to submit
// and wait, and Free to release it. Not safe for concurrent use; drive one Recorder per goroutine.
func NewRecorder() (*Recorder, error) {
	h := C.mtl_recorder_begin()
	if h == nil {
		return nil, fmt.Errorf("metal: Recorder begin failed")
	}
	return &Recorder{handle: h}, nil
}

// Unary encodes o = f(op, x) over x's elements into the Recorder (no commit).
func (r *Recorder) Unary(x, o *DeviceBuffer, op int) error {
	if x.n != o.n {
		return fmt.Errorf("metal: Recorder unary size %d != %d", x.n, o.n)
	}
	rc := C.mtl_recorder_unary(r.handle, x.handle, o.handle, C.int(x.n), C.int(op))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder unary failed (%d)", int(rc))
	}
	return nil
}

// Blit records a copy of n f32 elements from src[srcOff:] to dst[dstOff:] into the command
// buffer (offsets and count in f32 elements). This is the KV-cache append: copy the new
// token's k/v row into cache[cacheLen] each decode step, staying device-resident.
func (r *Recorder) Blit(src *DeviceBuffer, srcOff int, dst *DeviceBuffer, dstOff, n int) error {
	if srcOff < 0 || dstOff < 0 || n <= 0 || srcOff+n > src.n || dstOff+n > dst.n {
		return fmt.Errorf("metal: Recorder blit out of range: src[%d:%d]/%d dst[%d:%d]/%d", srcOff, srcOff+n, src.n, dstOff, dstOff+n, dst.n)
	}
	const f32 = 4
	rc := C.mtl_recorder_blit(r.handle, src.handle, C.int(srcOff*f32), dst.handle, C.int(dstOff*f32), C.int(n*f32))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder blit failed (%d)", int(rc))
	}
	return nil
}

// Copy2D records a strided rows×rowFloats sub-matrix copy: row r moves rowFloats elements
// from src[srcOff+r·srcStride:] to dst[dstOff+r·dstStride:] (all in f32 elements). This is
// the fused-QKV extraction primitive (§T613): after one combined QKV matmul the q/k/v parts
// are column bands of a wider matrix, and prefill needs them as contiguous rows (or written
// row-wise into the KV cache). rows=1 degenerates to Blit.
func (r *Recorder) Copy2D(src *DeviceBuffer, srcOff, srcStride int, dst *DeviceBuffer, dstOff, dstStride, rows, rowFloats int) error {
	if srcOff < 0 || dstOff < 0 || rows < 1 || rowFloats < 1 || srcStride < rowFloats || dstStride < rowFloats ||
		srcOff+(rows-1)*srcStride+rowFloats > src.n || dstOff+(rows-1)*dstStride+rowFloats > dst.n {
		return fmt.Errorf("metal: Recorder copy2d out of range: src off=%d stride=%d n=%d dst off=%d stride=%d n=%d rows=%d rowFloats=%d",
			srcOff, srcStride, src.n, dstOff, dstStride, dst.n, rows, rowFloats)
	}
	rc := C.mtl_recorder_copy2d(r.handle, src.handle, C.int(srcOff), C.int(srcStride),
		dst.handle, C.int(dstOff), C.int(dstStride), C.int(rows), C.int(rowFloats))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder copy2d failed (%d)", int(rc))
	}
	return nil
}

// Binary records O = a (op) b over n elements into the command buffer over device buffers
// (op 0=add, 1=sub, 2=mul, 3=div, 4=max, 5=min).
// BinaryN is Binary bounded to the FIRST n elements. Decoder buffers are allocated for the whole
// context, so an unbounded elementwise op over one decoded row touches ctx x width elements instead
// of width — at ctx=1024 that is a factor of 1024 in memory traffic, and it dominated decode
// (§ the 22-layer profile: 46.8 adds/token over 2,097,152 elements each). Every other recorded op
// already takes a row count; this gives Binary the same.
func (r *Recorder) BinaryN(a, b, o *DeviceBuffer, op, n int) error {
	if n < 0 || n > a.n || n > b.n || n > o.n {
		return fmt.Errorf("metal: Recorder binaryN n=%d out of range (a=%d b=%d o=%d)", n, a.n, b.n, o.n)
	}
	if n == 0 {
		return nil
	}
	rc := C.mtl_recorder_binary(r.handle, a.handle, b.handle, o.handle, C.int(n), C.int(op))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder binaryN failed (%d)", int(rc))
	}
	return nil
}

func (r *Recorder) Binary(a, b, o *DeviceBuffer, op int) error {
	if a.n != o.n || b.n != o.n {
		return fmt.Errorf("metal: Recorder binary size %d/%d != %d", a.n, b.n, o.n)
	}
	rc := C.mtl_recorder_binary(r.handle, a.handle, b.handle, o.handle, C.int(o.n), C.int(op))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder binary failed (%d)", int(rc))
	}
	return nil
}

// MatMul records C = A·B (M×K · K×N → M×N, MPS f32) into the command buffer over
// device-resident buffers. c must be distinct from a and b (MPS forbids aliasing result).
func (r *Recorder) MatMul(a, b, c *DeviceBuffer, m, k, n int) error {
	return r.matmulRec(a, b, c, m, k, n, 0)
}

// MatMulAcc records C += A·B (MPS beta=1) — the residual-add epilogue (§T613): a projection
// whose result feeds an elementwise add (the transformer residual stream) lands directly in
// the running sum, saving the separate add dispatch. C must hold valid input values.
func (r *Recorder) MatMulAcc(a, b, c *DeviceBuffer, m, k, n int) error {
	return r.matmulRec(a, b, c, m, k, n, 1)
}

func (r *Recorder) matmulRec(a, b, c *DeviceBuffer, m, k, n, accumulate int) error {
	// >= not ==: operands may be over-allocated scratch (§T418 StepN); the kernel reads/writes
	// only the leading m*k/k*n/m*n elements.
	if a.n < m*k || b.n < k*n || c.n < m*n {
		return fmt.Errorf("metal: Recorder matmul shape mismatch: a=%d(want ≥%d) b=%d(want ≥%d) c=%d(want ≥%d)", a.n, m*k, b.n, k*n, c.n, m*n)
	}
	rc := C.mtl_recorder_matmul(r.handle, a.handle, b.handle, c.handle, C.int(m), C.int(k), C.int(n), C.int(accumulate))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder matmul failed (%d)", int(rc))
	}
	return nil
}

// RMSNorm records O = rmsnorm(X)·gamma into the command buffer over device buffers.
// x, o are rows×dim; g is the dim-length gamma weight.
func (r *Recorder) RMSNorm(x, g, o *DeviceBuffer, rows, dim int, eps float32) error {
	if x.n < rows*dim || o.n < rows*dim || g.n < dim {
		return fmt.Errorf("metal: Recorder rmsnorm shape mismatch: x=%d o=%d (want %d) g=%d (want %d)", x.n, o.n, rows*dim, g.n, dim)
	}
	rc := C.mtl_recorder_rmsnorm(r.handle, x.handle, g.handle, o.handle, C.int(rows), C.int(dim), C.float(eps))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder rmsnorm failed (%d)", int(rc))
	}
	return nil
}

// flashattn records O = flash-attention(Q,K,V) into the command buffer over device buffers.
// q, o are [seq,dm]; k, v are [seq,kvHeads*dk]. causal!=0 masks future positions.
func (r *Recorder) flashattn(q, k, v, o *DeviceBuffer, seq, dm, heads, dk, causal, kvHeads int, scale float32) error {
	if q.n != seq*dm || o.n != seq*dm || k.n != seq*kvHeads*dk || v.n != seq*kvHeads*dk {
		return fmt.Errorf("metal: Recorder flashattn shape mismatch: q=%d o=%d (want %d) k=%d v=%d (want %d)", q.n, o.n, seq*dm, k.n, v.n, seq*kvHeads*dk)
	}
	rc := C.mtl_recorder_flashattn(r.handle, q.handle, k.handle, v.handle, o.handle,
		C.int(seq), C.int(dm), C.int(heads), C.int(dk), C.int(causal), C.int(kvHeads), C.float(scale))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder flashattn failed (%d)", int(rc))
	}
	return nil
}

// MHA records O = attention(Q,K,V) into the command buffer over device buffers, with
// separate query/key lengths — the decode-against-KV-cache shape (sq=1 query row, sk=cache
// length). q, o are [sq,dm]; k, v are [sk,kvHeads*dk]. causal!=0 masks future positions,
// window>0 limits attention to the last `window` keys.
func (r *Recorder) MHA(q, k, v, o *DeviceBuffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return r.MHAAt(q, k, v, o, 0, sq, sk, dm, heads, kvHeads, dk, causal, window, scale)
}

// MHAAt is MHA with the query rows starting at float-element offset qOff into q — the
// fused-QKV view (§T613): after ONE combined QKV matmul the query part is a sub-range of
// the wider qkv buffer. o still receives its sq×dm rows from offset 0.
func (r *Recorder) MHAAt(q, k, v, o *DeviceBuffer, qOff, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	// k,v may be an over-allocated KV cache; the kernel reads only the first sk rows, so
	// require they are AT LEAST sk*kvHeads*dk (not exactly).
	if qOff < 0 || q.n < qOff+sq*dm || o.n < sq*dm || k.n < sk*kvHeads*dk || v.n < sk*kvHeads*dk {
		return fmt.Errorf("metal: Recorder mha shape mismatch: q=%d(off %d) o=%d (want %d) k=%d v=%d (need >=%d)", q.n, qOff, o.n, sq*dm, k.n, v.n, sk*kvHeads*dk)
	}
	rc := C.mtl_recorder_mha(r.handle, q.handle, k.handle, v.handle, o.handle,
		C.int(sq), C.int(sk), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk), C.int(causal), C.int(window), C.float(scale), C.int(qOff))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder mha failed (%d)", int(rc))
	}
	return nil
}

// RoPE records O = RoPE(Q) into the command buffer over device buffers. q, o are [seq,width];
// inv holds `half` inverse frequencies. posOffset is the query's absolute start position
// (= KV-cache length at a decode step, so the new token rotates by its true position).
func (r *Recorder) RoPE(q, inv, o *DeviceBuffer, seq, width, heads, hd, half, posOffset int, posDiv float32) error {
	return r.RoPEAt(q, inv, o, 0, seq, width, heads, hd, half, posOffset, posDiv)
}

// RoPEAt is RoPE with q AND o addressed from float-element offset `off` — the fused-QKV
// view (§T613): the query/key sub-rows of a combined qkv buffer rotate in place without a
// copy. Both reads and writes shift by off; everything else matches RoPE.
func (r *Recorder) RoPEAt(q, inv, o *DeviceBuffer, off, seq, width, heads, hd, half, posOffset int, posDiv float32) error {
	if off < 0 || q.n < off+seq*width || o.n < off+seq*width || inv.n < half {
		return fmt.Errorf("metal: Recorder rope shape mismatch: q=%d o=%d (want %d at off %d) inv=%d (want %d)", q.n, o.n, seq*width, off, inv.n, half)
	}
	rc := C.mtl_recorder_rope(r.handle, q.handle, inv.handle, o.handle,
		C.int(seq), C.int(width), C.int(heads), C.int(hd), C.int(half), C.int(posOffset), C.float(posDiv), C.int(off))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder rope failed (%d)", int(rc))
	}
	return nil
}

// RoPEPair records the fused two-band rotation (§T613): ONE dispatch rotates the q band
// (headsQ heads at element offset offQ) AND the k band (headsK heads at offK) of a fused
// QKV buffer in place, rows `stride` floats wide — replacing two RoPEAt dispatches.
func (r *Recorder) RoPEPair(qkv, inv *DeviceBuffer, seq, stride, headsQ, offQ, headsK, offK, hd, half, posOffset int, posDiv float32) error {
	maxOff := max(offQ, offK)
	if offQ < 0 || offK < 0 || qkv.n < maxOff+seq*stride || inv.n < half {
		return fmt.Errorf("metal: Recorder rope2 shape mismatch: qkv=%d (want %d at off %d) inv=%d (want %d)", qkv.n, seq*stride, maxOff, inv.n, half)
	}
	rc := C.mtl_recorder_rope2(r.handle, qkv.handle, inv.handle,
		C.int(seq), C.int(stride), C.int(headsQ), C.int(offQ), C.int(headsK), C.int(offK),
		C.int(hd), C.int(half), C.int(posOffset), C.float(posDiv))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder rope2 failed (%d)", int(rc))
	}
	return nil
}

// LayerNorm records O = layernorm(X)·gamma + beta into the command buffer over device buffers
// (GPT-2-style norm; rmsnorm is the Llama variant). x, o are rows×dim; g, b are dim-length.
func (r *Recorder) LayerNorm(x, g, b, o *DeviceBuffer, rows, dim int, eps float32) error {
	if x.n < rows*dim || o.n < rows*dim || g.n < dim || b.n < dim {
		return fmt.Errorf("metal: Recorder layernorm shape mismatch: x=%d o=%d (want %d) g=%d b=%d (want %d)", x.n, o.n, rows*dim, g.n, b.n, dim)
	}
	rc := C.mtl_recorder_layernorm(r.handle, x.handle, g.handle, b.handle, o.handle, C.int(rows), C.int(dim), C.float(eps))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder layernorm failed (%d)", int(rc))
	}
	return nil
}

// QMatMulResident records O = X·dequant(W)ᵀ into the command buffer, where W is a resident
// quantized [N,K] weight (UploadQWeightQ8_0) and X, O are device buffers ([m,K] → [m,N]).
// This is the quantized batched-decode building block (§T413): a Q4/Q8 model's decode step can
// chain qmatmuls with the other record-mode ops in one command buffer.
func (r *Recorder) QMatMulResident(x *DeviceBuffer, w *ResidentQWeight, o *DeviceBuffer, m int) error {
	if w == nil || w.handle == nil {
		return fmt.Errorf("metal: Recorder qmatmul: resident weight is nil/closed")
	}
	if x.n < m*w.k || o.n < m*w.n {
		return fmt.Errorf("metal: Recorder qmatmul shape mismatch: x=%d (want %d) o=%d (want %d)", x.n, m*w.k, o.n, m*w.n)
	}
	rc := C.mtl_recorder_qmatmul(r.handle, x.handle, w.handle, o.handle,
		C.int(m), C.int(w.k), C.int(w.n), C.int(w.qt))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder qmatmul failed (%d)", int(rc))
	}
	return nil
}

// AddBias records O[i,j] = X[i,j] + B[j] into the command buffer over device buffers (x, o are
// rows×n; b is the n-length bias) — the broadcast bias-add multi-token GPT steps need (§T423).
func (r *Recorder) AddBias(x, b, o *DeviceBuffer, rows, n int) error {
	if x.n < rows*n || o.n < rows*n || b.n < n {
		return fmt.Errorf("metal: Recorder addbias shape mismatch: x=%d o=%d (want ≥%d) b=%d (want ≥%d)", x.n, o.n, rows*n, b.n, n)
	}
	rc := C.mtl_recorder_addbias(r.handle, x.handle, b.handle, o.handle, C.int(rows), C.int(n))
	if rc != 0 {
		return fmt.Errorf("metal: Recorder addbias failed (%d)", int(rc))
	}
	return nil
}

// Finish submits the command buffer and waits.
func (r *Recorder) Finish() error {
	rc := C.mtl_recorder_finish(r.handle)
	if rc != 0 {
		return fmt.Errorf("metal: Recorder finish failed (%d)", int(rc))
	}
	return nil
}

// Commit submits the command buffer WITHOUT waiting — the encode-overlap split (§T614):
// the host encodes the next step's command buffer while this one executes on the GPU.
// Pair with Wait; a committed recorder must not record further ops.
func (r *Recorder) Commit() error {
	rc := C.mtl_recorder_commit(r.handle)
	if rc != 0 {
		return fmt.Errorf("metal: Recorder commit failed (%d)", int(rc))
	}
	return nil
}

// Wait blocks until a Commit'ed command buffer completes on the GPU.
func (r *Recorder) Wait() error {
	rc := C.mtl_recorder_wait(r.handle)
	if rc != 0 {
		return fmt.Errorf("metal: Recorder wait failed (%d)", int(rc))
	}
	return nil
}

// LastGPUSeconds returns the GPU-side duration of the most recently waited command buffer,
// from the device timestamps — free of host submit/wake jitter.
func LastGPUSeconds() float64 { return float64(C.mtl_last_gpu_seconds()) }

// SetQ4KMatrixUnit toggles the matrix-unit Q4_K path (M>=8). Test seam: it lets a parity test
// obtain the cooperative kernel's result for the same inputs.
func SetQ4KMatrixUnit(on bool) {
	v := C.int(0)
	if on {
		v = 1
	}
	C.mtl_set_q4k_mm(v)
}

// DequantQ4K records the dequantization of a resident Q4_K weight into a dense f32 [K][N] buffer,
// the operand order MPS GEMM consumes.
func (r *Recorder) DequantQ4K(w *ResidentQWeight, o *DeviceBuffer) error {
	if w == nil || w.handle == nil {
		return fmt.Errorf("metal: DequantQ4K: resident weight is nil/closed")
	}
	if o.n < w.k*w.n {
		return fmt.Errorf("metal: DequantQ4K: output %d < %d", o.n, w.k*w.n)
	}
	rc := C.mtl_recorder_dequant_qk(r.handle, w.handle, o.handle, C.int(w.k), C.int(w.n), C.int(w.qt))
	if rc != 0 {
		return fmt.Errorf("metal: DequantQ4K failed (%d)", int(rc))
	}
	return nil
}

// SetQ4KDequantGemm toggles the prompt-processing path (expand the Q4_K weight once, then a dense
// GEMM) used above the measured batch-size crossover.
func SetQ4KDequantGemm(on bool) {
	v := C.int(0)
	if on {
		v = 1
	}
	C.mtl_set_q4k_dq_gemm(v)
}

func (r *Recorder) Free() {
	if r.handle != nil {
		C.mtl_recorder_free(r.handle)
		r.handle = nil
	}
}

// chainMatMulReLU records an MPS matmul then a ReLU into one command buffer over a
// device-resident intermediate (§T369) — the record-mode chaining proof for ADR-0019
// Phase 2. Returns Out = relu(A·B).
func chainMatMulReLU(a, b []float32, m, k, n int) ([]float32, error) {
	out := make([]float32, m*n)
	rc := C.mtl_chain_matmul_relu(
		(*C.float)(&a[0]), (*C.float)(&b[0]), (*C.float)(&out[0]),
		C.int(m), C.int(k), C.int(n))
	if rc != 0 {
		return nil, fmt.Errorf("metal: chain matmul-relu failed (%d)", int(rc))
	}
	return out, nil
}

func init() {
	if Available() {
		backend.Register(Backend{})
	}
	// no device → not registered → feature detection via backend.Available() (§I4/§V4)
}
