//go:build vulkan && cgo

package vulkan

/*
#cgo pkg-config: vulkan
#include "vk_bridge.h"
*/
import "C"

import (
	_ "embed"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// spirv is the compiled matmul compute shader. It is a BUILD ARTIFACT produced
// from shaders/matmul.comp by `make vulkan-spv` (glslc) — like `-tags cuda`
// needing the CUDA toolkit, `-tags vulkan` needs the Vulkan SDK's glslc to have
// run first. SPIR-V is a stream of 32-bit words; embedding the bytes is fine —
// the bridge passes the length and pointer straight to vkCreateShaderModule.
//
//go:embed shaders/matmul.spv
var spirv []byte

// matmulStridedSpirv + softmaxCausalSpirv back the §T397 MPS-analog attention forward (strided/offset
// GEMM over packed Q/K/V windows + causal row softmax), compiled by `make vulkan-spv`.
//
//go:embed shaders/matmul_strided.spv
var matmulStridedSpirv []byte

//go:embed shaders/softmax_causal.spv
var softmaxCausalSpirv []byte

// mhaDecodeSpirv is the cooperative single-query decode-attention shader (§T429, one subgroup per
// head; SPIR-V 1.3 for subgroup shuffles — the instance already requests Vulkan 1.1).
//
//go:embed shaders/mha_decode.spv
var mhaDecodeSpirv []byte

// rmsnormSpirv is the compiled RMSNorm shader (shaders/rmsnorm.comp → rmsnorm.spv via
// `make vulkan-spv`), embedded like the others.
//
//go:embed shaders/rmsnorm.spv
var rmsnormSpirv []byte

// rmsnormBwdSpirv is the compiled RMSNorm BACKWARD shader (shaders/rmsnorm_bwd.comp →
// rmsnorm_bwd.spv via `make vulkan-spv`), embedded like the others. Needs float atomics.
//
//go:embed shaders/rmsnorm_bwd.spv
var rmsnormBwdSpirv []byte

// ropeSpirv is the compiled RoPE shader (shaders/rope.comp → rope.spv via
// `make vulkan-spv`), embedded like the others.
//
//go:embed shaders/rope.spv
var ropeSpirv []byte

// ropeBwdSpirv is the compiled RoPE BACKWARD shader (shaders/rope_bwd.comp → rope_bwd.spv
// via `make vulkan-spv`) — the inverse rotation. Same buffers/push as the forward, so it
// runs through the same vk_rope_f32 launcher with a different SPIR-V module.
//
//go:embed shaders/rope_bwd.spv
var ropeBwdSpirv []byte

// softmaxSpirv is the compiled softmax shader (shaders/softmax.comp → softmax.spv via
// `make vulkan-spv`), embedded like the others.
//
//go:embed shaders/softmax.spv
var softmaxSpirv []byte

// crossentropyBwdSpirv is the compiled cross-entropy BACKWARD shader
// (shaders/crossentropy_bwd.comp → crossentropy_bwd.spv via `make vulkan-spv`), embedded.
//
//go:embed shaders/crossentropy_bwd.spv
var crossentropyBwdSpirv []byte

// embedBwdSpirv is the compiled embedding BACKWARD shader (shaders/embed_bwd.comp →
// embed_bwd.spv via `make vulkan-spv`), embedded like the others. Needs float atomics.
//
//go:embed shaders/embed_bwd.spv
var embedBwdSpirv []byte

// layernormSpirv is the compiled LayerNorm shader (shaders/layernorm.comp → layernorm.spv
// via `make vulkan-spv`), embedded like the others.
//
//go:embed shaders/layernorm.spv
var layernormSpirv []byte

// layernormBwdSpirv is the compiled LayerNorm BACKWARD shader (shaders/layernorm_bwd.comp →
// layernorm_bwd.spv via `make vulkan-spv`), embedded like the others. Needs float atomics.
//
//go:embed shaders/layernorm_bwd.spv
var layernormBwdSpirv []byte

// unarySpirv is the compiled generic unary-elementwise shader (shaders/unary.comp →
// unary.spv via `make vulkan-spv`), embedded like the others.
//
//go:embed shaders/unary.spv
var unarySpirv []byte

// binarySpirv is the compiled generic same-shape binary-elementwise shader
// (shaders/binary.comp → binary.spv via `make vulkan-spv`), embedded like the others.
//
//go:embed shaders/binary.spv
var binarySpirv []byte

// crossEntropySpirv + embedSpirv back the §T403 forward kernels (parity with metal §T401/T402:
// the loss + token-embedding forwards were silent CPU fallbacks on vulkan too).
//
//go:embed shaders/crossentropy.spv
var crossEntropySpirv []byte

//go:embed shaders/embed.spv
var embedSpirv []byte

// addBiasSpirv is the compiled bias-add shader (shaders/addbias.comp → addbias.spv,
// §T352), embedded like the others.
//
//go:embed shaders/addbias.spv
var addBiasSpirv []byte

// geluBwdSpirv is the compiled GELU-backward shader (shaders/gelu_bwd.comp → gelu_bwd.spv,
// §T353), embedded like the others.
//
//go:embed shaders/gelu_bwd.spv
var geluBwdSpirv []byte

// siluBwdSpirv is the compiled SiLU-backward shader (shaders/silu_bwd.comp → silu_bwd.spv,
// §T362), embedded like the others.
//
//go:embed shaders/silu_bwd.spv
var siluBwdSpirv []byte

// addBiasBwdSpirv is the compiled bias-add-backward shader (shaders/addbias_bwd.comp →
// addbias_bwd.spv, §T354), embedded like the others.
//
//go:embed shaders/addbias_bwd.spv
var addBiasBwdSpirv []byte

// unary op selectors — MUST match the switch in shaders/unary.comp.
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

// binary op selectors — MUST match the switch in shaders/binary.comp.
const (
	binaryAdd = 0
	binarySub = 1
	binaryMul = 2
	binaryDiv = 3
	binaryMax = 4
	binaryMin = 5
)

// qmatmulQ8Spirv is the compiled Q8_0 quantized-matmul shader (shaders/qmatmul_q8.comp →
// qmatmul_q8.spv via `make vulkan-spv`), embedded like the others (§T137).
//
//go:embed shaders/qmatmul_q8.spv
var qmatmulQ8Spirv []byte

// qmatmulQ4_0Spirv is the compiled Q4_0 quantized-matmul shader (shaders/qmatmul_q4_0.comp →
// qmatmul_q4_0.spv via `make vulkan-spv`), embedded like the others (§T167).
//
//go:embed shaders/qmatmul_q4_0.spv
var qmatmulQ4_0Spirv []byte

// qmatmulQ4KSpirv is the compiled Q4_K quantized-matmul shader (shaders/qmatmul_q4k.comp →
// qmatmul_q4k.spv via `make vulkan-spv`), embedded like the others (§T139).
//
//go:embed shaders/qmatmul_q4k.spv
var qmatmulQ4KSpirv []byte

// qmatmulQ6KSpirv is the compiled Q6_K quantized-matmul shader (shaders/qmatmul_q6k.comp →
// qmatmul_q6k.spv via `make vulkan-spv`), embedded like the others (§T141).
//
//go:embed shaders/qmatmul_q6k.spv
var qmatmulQ6KSpirv []byte

// qmatmulQ5KSpirv is the compiled Q5_K quantized-matmul shader (shaders/qmatmul_q5k.comp →
// qmatmul_q5k.spv via `make vulkan-spv`), embedded like the others (§T144).
//
//go:embed shaders/qmatmul_q5k.spv
var qmatmulQ5KSpirv []byte

// qmatmulQ2KSpirv is the compiled Q2_K quantized-matmul shader (shaders/qmatmul_q2k.comp →
// qmatmul_q2k.spv via `make vulkan-spv`), embedded like the others (§T146).
//
//go:embed shaders/qmatmul_q2k.spv
var qmatmulQ2KSpirv []byte

// qmatmulQ3KSpirv is the compiled Q3_K quantized-matmul shader (shaders/qmatmul_q3k.comp →
// qmatmul_q3k.spv via `make vulkan-spv`), embedded like the others (§T148).
//
//go:embed shaders/qmatmul_q3k.spv
var qmatmulQ3KSpirv []byte

// mhaSpirv is the compiled fused-attention shader (shaders/mha.comp → mha.spv via
// `make vulkan-spv`), embedded like the matmul module above (§T89).
//
//go:embed shaders/mha.spv
var mhaSpirv []byte

// flashAttnSpirv is the compiled FlashAttention-2 forward shader
// (shaders/flashattn.comp → flashattn.spv), embedded like the others (§T110).
//
//go:embed shaders/flashattn.spv
var flashAttnSpirv []byte

// retentionSpirv is the compiled RetNet retention forward shader (shaders/retention.comp →
// retention.spv), embedded like the others (§T173).
//
//go:embed shaders/retention.spv
var retentionSpirv []byte

// retentionBwdSpirv is the compiled RetNet retention backward shader (shaders/retention_bwd.comp →
// retention_bwd.spv), embedded like the others (§T175). It uses float atomics
// (VK_EXT_shader_atomic_float) for the shared dK/dV.
//
//go:embed shaders/retention_bwd.spv
var retentionBwdSpirv []byte

// mhaBwdSpirv is the compiled fused-attention backward shader (shaders/mha_bwd.comp
// → mha_bwd.spv), embedded like the others (§T90). It uses float atomics
// (VK_EXT_shader_atomic_float) for the shared dK/dV.
//
// softmaxPackedSpirv / smJacobianSpirv serve the §T528 matmul-decomposed backward
// chain: packed-row softmax (causal via row % seq) and the softmax-jacobian.
//
//go:embed shaders/softmax_packed.spv
var softmaxPackedSpirv []byte

//go:embed shaders/sm_jacobian.spv
var smJacobianSpirv []byte

//go:embed shaders/mha_bwd.spv
var mhaBwdSpirv []byte

// im2colSpirv/coloutSpirv are the conv2d GEMM-lowering stages (§T342): im2col
// unrolls the input into a column matrix, the shared tiled matmul shader
// multiplies the weights against it, and colout scatters into NCHW + bias.
//
//go:embed shaders/im2col.spv
var im2colSpirv []byte

//go:embed shaders/colout.spv
var coloutSpirv []byte

// conv2dBwdSpirv is the compiled conv2d backward shader (shaders/conv2d_bwd.comp →
// conv2d_bwd.spv), embedded like the others (§T102). It uses float atomics
// (VK_EXT_shader_atomic_float) for the shared dX/dW/dBias.
//
//go:embed shaders/conv2d_bwd.spv
var conv2dBwdSpirv []byte

// device is the Vulkan tensor.Device. Tensor memory stays host-side; the kernel
// copies through host-visible buffers per call — honest about transfer cost.
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

func (device) Kind() tensor.DeviceKind     { return tensor.KindVulkan }
func (d device) String() string            { return d.Kind().String() }
func (device) Allocator() tensor.Allocator { return tensor.Heap() }

// Backend implements backend.Backend over Vulkan compute. Synchronous: every
// kernel submits and vkQueueWaitIdle before returning, so Synchronize is a no-op
// (§V14 permits async later without an API break).
type Backend struct{}

func (Backend) Name() backend.Name    { return backend.Vulkan }
func (Backend) Device() tensor.Device { return device{} }
func (Backend) Synchronize() error    { return nil }

// Available reports whether a Vulkan compute-capable device is present.
func Available() bool { return C.vk_available() == 1 }

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

// QMatMul implements backend.QuantMatMuler (§T142): it dispatches a quantized linear layer to
// the matching in-kernel Vulkan matmul (Q8_0/Q4_K/Q6_K), returning backend.ErrQuantUnsupported
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

// residentSpirv returns the compiled shader for a quant type's resident matmul (nil if none).
func residentSpirv(qt uint32) []byte {
	switch qt {
	case qtQ4_0:
		return qmatmulQ4_0Spirv
	case qtQ8_0:
		return qmatmulQ8Spirv
	case qtQ2_K:
		return qmatmulQ2KSpirv
	case qtQ3_K:
		return qmatmulQ3KSpirv
	case qtQ4_K:
		return qmatmulQ4KSpirv
	case qtQ5_K:
		return qmatmulQ5KSpirv
	case qtQ6_K:
		return qmatmulQ6KSpirv
	}
	return nil
}

// residentRowBytes returns the per-row byte size of a quantized [·,k] weight of ggml type qt and
// the required alignment of k (0,0,false for an unsupported type).
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

// ResidentQWeight is a quantized weight uploaded ONCE to a device-resident Vulkan buffer and
// reused across QMatMul calls (§T156) — the decode-loop perf lever, Vulkan twin of the Metal one.
// Close frees the buffer (not reclaimed automatically).
type ResidentQWeight struct {
	handle  unsafe.Pointer // ResidentBuf* (nil after Close)
	n, k    int
	qt      uint32
	wBytes  int             // padded (4-byte-aligned) resident buffer length
	cleanup runtime.Cleanup // frees the buffer if the value is GC'd without an explicit Close
}

// UploadQuant implements backend.ResidentQuantMatMuler (§T156): uploads a quantized weight of any
// resident-supported k-quant/Q8_0 type to a device buffer for reuse, else ErrQuantUnsupported.
func (Backend) UploadQuant(weight []byte, quantType uint32, n, k int) (backend.ResidentWeight, error) {
	spv := residentSpirv(quantType)
	rowBytes, align, ok := residentRowBytes(quantType, k)
	if spv == nil || !ok {
		return nil, backend.ErrQuantUnsupported
	}
	if len(spv) == 0 {
		return nil, fmt.Errorf("vulkan: resident shader for quant %d not embedded", quantType)
	}
	if k <= 0 || k%align != 0 {
		return nil, fmt.Errorf("vulkan: resident upload K=%d must be a positive multiple of %d", k, align)
	}
	if len(weight) != n*rowBytes {
		return nil, fmt.Errorf("vulkan: resident upload weight %d bytes != %d", len(weight), n*rowBytes)
	}
	padded := weight // the shader reads W as a uint word array → pad to a 4-byte boundary
	if r := len(weight) % 4; r != 0 {
		padded = make([]byte, len(weight)+(4-r))
		copy(padded, weight)
	}
	h := C.vk_qweight_upload((*C.uchar)(unsafe.Pointer(&padded[0])), C.int(len(padded)))
	if h == nil {
		return nil, fmt.Errorf("vulkan: qweight upload failed")
	}
	r := &ResidentQWeight{handle: unsafe.Pointer(h), n: n, k: k, qt: quantType, wBytes: len(padded)}
	// free the buffer if r is collected without an explicit Close (the cleanup captures only the
	// handle, never r); Close stops it to avoid a double free.
	r.cleanup = runtime.AddCleanup(r, func(p unsafe.Pointer) { C.vk_qweight_free(p) }, unsafe.Pointer(h))
	return r, nil
}

// QMatMul computes y[M,N] = x[M,K] · dequant(W)ᵀ using the resident weight — only x is uploaded.
func (r *ResidentQWeight) QMatMul(x *tensor.Tensor) (*tensor.Tensor, error) {
	if r.handle == nil {
		return nil, fmt.Errorf("vulkan: resident weight already closed")
	}
	if x.Ndim() != 2 || x.Shape()[1] != r.k {
		return nil, fmt.Errorf("vulkan: resident QMatMul x must be [M,%d], got %v", r.k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: resident QMatMul is f32-only, got %v", x.Dtype())
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, r.n})
	if m == 0 || r.n == 0 {
		return out, nil
	}
	spv := residentSpirv(r.qt)
	xc := x.Contiguous()
	rc := C.vk_qmatmul_resident(
		(*C.uint32_t)(unsafe.Pointer(&spv[0])), C.int(len(spv)),
		(*C.float)(&xc.Storage().F32()[0]),
		r.handle,
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(r.k), C.int(r.n), C.int(r.wBytes),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: resident QMatMul failed (code %d)", int(rc))
	}
	return out, nil
}

// Close frees the resident GPU buffer. Idempotent and optional — an un-closed weight is freed by a
// runtime cleanup when it becomes unreachable — but Close releases the memory promptly.
func (r *ResidentQWeight) Close() error {
	if r.handle != nil {
		r.cleanup.Stop() // don't let the cleanup double-free
		C.vk_qweight_free(r.handle)
		r.handle = nil
	}
	return nil
}

// QMatMulQ8_0 computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q8_0-quantized [N,K] matrix (row-major, K/32 blocks of [f16 scale + 32 int8], §R94)
// dequantized in-kernel — the Vulkan twin of the Metal kernel (§T136/§T137), the building
// block of GPU quantized inference. K must be a multiple of 32 and weight must be N·(K/32)·34
// bytes. The weight is padded to a 4-byte boundary so the shader can read it as a uint word
// array. Accumulation is f32 (the reference gguf.QMatMul is f64), matching within a crossTol.
func QMatMulQ8_0(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("vulkan: QMatMulQ8_0 x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: QMatMulQ8_0 is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%32 != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ8_0 K=%d must be a positive multiple of 32", k)
	}
	nb := k / 32
	if len(weight) != n*nb*34 {
		return nil, fmt.Errorf("vulkan: QMatMulQ8_0 weight %d bytes != %d (N·K/32·34)", len(weight), n*nb*34)
	}
	if len(qmatmulQ8Spirv) == 0 {
		return nil, fmt.Errorf("vulkan: qmatmul_q8.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	// pad the weight to a multiple of 4 bytes so the shader's uint[] indexing never reads OOB
	padded := weight
	if r := len(weight) % 4; r != 0 {
		padded = make([]byte, len(weight)+(4-r))
		copy(padded, weight)
	}
	xc := x.Contiguous()
	rc := C.vk_qmatmul_q8_0(
		(*C.uint32_t)(unsafe.Pointer(&qmatmulQ8Spirv[0])), C.int(len(qmatmulQ8Spirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&padded[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n), C.int(len(padded)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ8_0 failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ4_0 computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q4_0-quantized [N,K] matrix (row-major, K/32 blocks of 18 bytes = [f16 scale + 16 bytes of 32
// 4-bit nibbles], §R94) dequantized in-kernel — the Vulkan twin of the Metal kernel (§T166), the
// GPU path for the very common Q4_0 GGUF format. K must be a multiple of 32 and weight must be
// N·(K/32)·18 bytes. The weight is padded to a 4-byte boundary so the shader can read it as a uint
// word array. Accumulation is f32 (the reference gguf.QMatMul is f64), matching within a crossTol.
func QMatMulQ4_0(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("vulkan: QMatMulQ4_0 x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: QMatMulQ4_0 is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%32 != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ4_0 K=%d must be a positive multiple of 32", k)
	}
	nb := k / 32
	if len(weight) != n*nb*18 {
		return nil, fmt.Errorf("vulkan: QMatMulQ4_0 weight %d bytes != %d (N·K/32·18)", len(weight), n*nb*18)
	}
	if len(qmatmulQ4_0Spirv) == 0 {
		return nil, fmt.Errorf("vulkan: qmatmul_q4_0.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	// pad the weight to a multiple of 4 bytes so the shader's uint[] indexing never reads OOB
	padded := weight
	if r := len(weight) % 4; r != 0 {
		padded = make([]byte, len(weight)+(4-r))
		copy(padded, weight)
	}
	xc := x.Contiguous()
	rc := C.vk_qmatmul_q4_0(
		(*C.uint32_t)(unsafe.Pointer(&qmatmulQ4_0Spirv[0])), C.int(len(qmatmulQ4_0Spirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&padded[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n), C.int(len(padded)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ4_0 failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ4_K computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q4_K-quantized [N,K] matrix (row-major, K/256 super-blocks of 144 bytes, §R100) dequantized
// in-kernel — the Vulkan twin of the Metal kernel (§T138/§T139), the DOMINANT real-world quant
// (the bulk of Q4_K_M models). K must be a multiple of 256 and weight must be N·(K/256)·144
// bytes. The weight is padded to a 4-byte boundary so the shader can read it as a uint word
// array. Accumulation is f32 (the reference gguf.QMatMul is f64), matching within a crossTol.
func QMatMulQ4_K(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("vulkan: QMatMulQ4_K x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: QMatMulQ4_K is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ4_K K=%d must be a positive multiple of 256", k)
	}
	nsb := k / 256
	if len(weight) != n*nsb*144 {
		return nil, fmt.Errorf("vulkan: QMatMulQ4_K weight %d bytes != %d (N·K/256·144)", len(weight), n*nsb*144)
	}
	if len(qmatmulQ4KSpirv) == 0 {
		return nil, fmt.Errorf("vulkan: qmatmul_q4k.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	padded := weight
	if r := len(weight) % 4; r != 0 {
		padded = make([]byte, len(weight)+(4-r))
		copy(padded, weight)
	}
	xc := x.Contiguous()
	rc := C.vk_qmatmul_q4k(
		(*C.uint32_t)(unsafe.Pointer(&qmatmulQ4KSpirv[0])), C.int(len(qmatmulQ4KSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&padded[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n), C.int(len(padded)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ4_K failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ6_K computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q6_K-quantized [N,K] matrix (row-major, K/256 super-blocks of 210 bytes, §R99) dequantized
// in-kernel — the Vulkan twin of the Metal kernel (§T140/§T141), the higher-precision tensors
// of Q4_K_M models. K must be a multiple of 256 and weight must be N·(K/256)·210 bytes. The
// weight is padded to a 4-byte boundary so the shader can read it as a uint word array.
// Accumulation is f32 (the reference gguf.QMatMul is f64), matching within a crossTol.
func QMatMulQ6_K(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("vulkan: QMatMulQ6_K x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: QMatMulQ6_K is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ6_K K=%d must be a positive multiple of 256", k)
	}
	nsb := k / 256
	if len(weight) != n*nsb*210 {
		return nil, fmt.Errorf("vulkan: QMatMulQ6_K weight %d bytes != %d (N·K/256·210)", len(weight), n*nsb*210)
	}
	if len(qmatmulQ6KSpirv) == 0 {
		return nil, fmt.Errorf("vulkan: qmatmul_q6k.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	padded := weight
	if r := len(weight) % 4; r != 0 {
		padded = make([]byte, len(weight)+(4-r))
		copy(padded, weight)
	}
	xc := x.Contiguous()
	rc := C.vk_qmatmul_q6k(
		(*C.uint32_t)(unsafe.Pointer(&qmatmulQ6KSpirv[0])), C.int(len(qmatmulQ6KSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&padded[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n), C.int(len(padded)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ6_K failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ5_K computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q5_K-quantized [N,K] matrix (row-major, K/256 super-blocks of 176 bytes, §R102) dequantized
// in-kernel — the Vulkan twin of the Metal kernel (§T143/§T144), the Q5_K_M weight format. K
// must be a multiple of 256 and weight must be N·(K/256)·176 bytes. The weight is padded to a
// 4-byte boundary so the shader can read it as a uint word array. Accumulation is f32 (the
// reference gguf.QMatMul is f64), matching within a crossTol.
func QMatMulQ5_K(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("vulkan: QMatMulQ5_K x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: QMatMulQ5_K is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ5_K K=%d must be a positive multiple of 256", k)
	}
	nsb := k / 256
	if len(weight) != n*nsb*176 {
		return nil, fmt.Errorf("vulkan: QMatMulQ5_K weight %d bytes != %d (N·K/256·176)", len(weight), n*nsb*176)
	}
	if len(qmatmulQ5KSpirv) == 0 {
		return nil, fmt.Errorf("vulkan: qmatmul_q5k.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	padded := weight
	if r := len(weight) % 4; r != 0 {
		padded = make([]byte, len(weight)+(4-r))
		copy(padded, weight)
	}
	xc := x.Contiguous()
	rc := C.vk_qmatmul_q5k(
		(*C.uint32_t)(unsafe.Pointer(&qmatmulQ5KSpirv[0])), C.int(len(qmatmulQ5KSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&padded[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n), C.int(len(padded)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ5_K failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ2_K computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q2_K-quantized [N,K] matrix (row-major, K/256 super-blocks of 84 bytes, §R104) dequantized
// in-kernel — the Vulkan twin of the Metal kernel (§T145/§T146), the smallest quant. K must be a
// multiple of 256 and weight must be N·(K/256)·84 bytes. The weight is padded to a 4-byte
// boundary so the shader can read it as a uint word array. Accumulation is f32 (the reference
// gguf.QMatMul is f64), matching within a crossTol.
func QMatMulQ2_K(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("vulkan: QMatMulQ2_K x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: QMatMulQ2_K is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ2_K K=%d must be a positive multiple of 256", k)
	}
	nsb := k / 256
	if len(weight) != n*nsb*84 {
		return nil, fmt.Errorf("vulkan: QMatMulQ2_K weight %d bytes != %d (N·K/256·84)", len(weight), n*nsb*84)
	}
	if len(qmatmulQ2KSpirv) == 0 {
		return nil, fmt.Errorf("vulkan: qmatmul_q2k.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	padded := weight
	if r := len(weight) % 4; r != 0 {
		padded = make([]byte, len(weight)+(4-r))
		copy(padded, weight)
	}
	xc := x.Contiguous()
	rc := C.vk_qmatmul_q2k(
		(*C.uint32_t)(unsafe.Pointer(&qmatmulQ2KSpirv[0])), C.int(len(qmatmulQ2KSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&padded[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n), C.int(len(padded)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ2_K failed (code %d)", int(rc))
	}
	return out, nil
}

// QMatMulQ3_K computes y[M,N] = x[M,K] · dequant(weight)ᵀ on the GPU, where weight is a
// Q3_K-quantized [N,K] matrix (row-major, K/256 super-blocks of 110 bytes, §R103) dequantized
// in-kernel — the Vulkan twin of the Metal kernel (§T147/§T148), the Q3_K_M weight format. K
// must be a multiple of 256 and weight must be N·(K/256)·110 bytes. The weight is padded to a
// 4-byte boundary so the shader can read it as a uint word array. Accumulation is f32 (the
// reference gguf.QMatMul is f64), matching within a crossTol.
func QMatMulQ3_K(x *tensor.Tensor, weight []byte, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("vulkan: QMatMulQ3_K x must be [M,%d], got %v", k, x.Shape())
	}
	if x.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: QMatMulQ3_K is f32-only, got %v", x.Dtype())
	}
	if k <= 0 || k%256 != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ3_K K=%d must be a positive multiple of 256", k)
	}
	nsb := k / 256
	if len(weight) != n*nsb*110 {
		return nil, fmt.Errorf("vulkan: QMatMulQ3_K weight %d bytes != %d (N·K/256·110)", len(weight), n*nsb*110)
	}
	if len(qmatmulQ3KSpirv) == 0 {
		return nil, fmt.Errorf("vulkan: qmatmul_q3k.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	m := x.Shape()[0]
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	if m == 0 || n == 0 {
		return out, nil
	}
	padded := weight
	if r := len(weight) % 4; r != 0 {
		padded = make([]byte, len(weight)+(4-r))
		copy(padded, weight)
	}
	xc := x.Contiguous()
	rc := C.vk_qmatmul_q3k(
		(*C.uint32_t)(unsafe.Pointer(&qmatmulQ3KSpirv[0])), C.int(len(qmatmulQ3KSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.uchar)(unsafe.Pointer(&padded[0])),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n), C.int(len(padded)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: QMatMulQ3_K failed (code %d)", int(rc))
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
			return ropeKernelF32(backend.OpRoPE, ropeSpirv), true
		case backend.OpRoPEBackward:
			return ropeKernelF32(backend.OpRoPEBackward, ropeBwdSpirv), true
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

// binaryF32 returns a GPU kernel for a same-shape binary elementwise op (one Vulkan
// invocation per element, selected by `sel`). It handles only equal-shaped f32 inputs —
// broadcasting and non-f32 fall back to the reference (§I4) — which covers the FFN's
// SiLU⊙up gate and transformer residual adds without a CPU round-trip.
func binaryF32(op backend.Op, sel int) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		refFallback := func() ([]*tensor.Tensor, error) {
			return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), op, in, attrs)
		}
		if len(in) != 2 {
			return nil, fmt.Errorf("vulkan: binary op wants 2 inputs, got %d", len(in))
		}
		a, b := in[0], in[1]
		n := a.Numel()
		// ADR-0008 (§T534): per-op memory-bound elementwise loses to a single CPU pass on
		// host-resident tensors (metal twin measured +10.8% training from this routing);
		// the GPU kernel remains the no-cpu fallback.
		if cpu, ok := cpuPrefers(op, a.Dtype()); ok {
			// strip the recorder: the OUTER Execute records this op onto the tape;
			// re-dispatching with the recorder attached would record it TWICE and
			// double the op's gradients (§B49 — caught by the full sweep's tight
			// trained-model bars, invisible to shape/parity suites).
			return backend.Execute(ctx.WithBackend(cpu).WithRecorder(nil), op, in, attrs)
		}
		if a.Dtype() != tensor.F32 || b.Dtype() != tensor.F32 || !a.Shape().Equal(b.Shape()) || n == 0 || len(binarySpirv) == 0 {
			return refFallback() // broadcasting / non-f32 / empty → reference
		}
		ac, bc := a.Contiguous(), b.Contiguous()
		out := tensor.New(tensor.F32, a.Shape())
		rc := C.vk_binary_f32(
			(*C.uint32_t)(unsafe.Pointer(&binarySpirv[0])), C.int(len(binarySpirv)),
			(*C.float)(&ac.Storage().F32()[0]),
			(*C.float)(&bc.Storage().F32()[0]),
			(*C.float)(&out.Storage().F32()[0]),
			C.int(n), C.int(sel),
		)
		if rc != 0 {
			return nil, fmt.Errorf("vulkan: binary op failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{out}, nil
	}
}

// addBiasF32 computes O[i,j] = X[i,j] + B[j] on the GPU — the FFN's bias add, a ref/CPU
// fallback that dominated the forward before it moved here (§T352). x rank-2, bias rank-1
// [cols]; anything else falls back to the reference (§I4).
func addBiasF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpAddBias, in, attrs)
	}
	if len(in) != 2 {
		return nil, fmt.Errorf("vulkan: addbias wants (x, bias), got %d inputs", len(in))
	}
	x, b := in[0], in[1]
	if x.Dtype() != tensor.F32 || b.Dtype() != tensor.F32 || x.Ndim() != 2 || b.Ndim() != 1 || len(addBiasSpirv) == 0 {
		return refFallback()
	}
	rows, n := x.Shape()[0], x.Shape()[1]
	if b.Shape()[0] != n || rows == 0 || n == 0 {
		return refFallback()
	}
	xc, bc := x.Contiguous(), b.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.vk_addbias_f32(
		(*C.uint32_t)(unsafe.Pointer(&addBiasSpirv[0])), C.int(len(addBiasSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&bc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(rows), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: addbias failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// addBiasBackwardF32 computes dbias[j] = Σ_i g[i,j] on the GPU (§T354) — the bias-add VJP
// dispatches this so the column-sum doesn't run as a CPU scalar loop. in = (g[rows,n]).
func addBiasBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpAddBiasBackward, in, attrs)
	}
	if len(in) != 1 {
		return nil, fmt.Errorf("vulkan: addbias-backward wants (g), got %d inputs", len(in))
	}
	g := in[0]
	if g.Dtype() != tensor.F32 || g.Ndim() != 2 || len(addBiasBwdSpirv) == 0 {
		return refFallback()
	}
	rows, n := g.Shape()[0], g.Shape()[1]
	if rows == 0 || n == 0 {
		return refFallback()
	}
	gc := g.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{n})
	rc := C.vk_addbias_backward_f32(
		(*C.uint32_t)(unsafe.Pointer(&addBiasBwdSpirv[0])), C.int(len(addBiasBwdSpirv)),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(rows), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: addbias-backward failed (code %d)", int(rc))
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
		return nil, fmt.Errorf("vulkan: gelu-backward wants (x, g), got %d inputs", len(in))
	}
	x, g := in[0], in[1]
	n := x.Numel()
	if x.Dtype() != tensor.F32 || g.Dtype() != tensor.F32 || !g.Shape().Equal(x.Shape()) || n == 0 || len(geluBwdSpirv) == 0 {
		return refFallback()
	}
	xc, gc := x.Contiguous(), g.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.vk_gelu_backward_f32(
		(*C.uint32_t)(unsafe.Pointer(&geluBwdSpirv[0])), C.int(len(geluBwdSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: gelu-backward failed (code %d)", int(rc))
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
		return nil, fmt.Errorf("vulkan: silu-backward wants (x, g), got %d inputs", len(in))
	}
	x, g := in[0], in[1]
	n := x.Numel()
	if x.Dtype() != tensor.F32 || g.Dtype() != tensor.F32 || !g.Shape().Equal(x.Shape()) || n == 0 || len(siluBwdSpirv) == 0 {
		return refFallback()
	}
	xc, gc := x.Contiguous(), g.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.vk_silu_backward_f32(
		(*C.uint32_t)(unsafe.Pointer(&siluBwdSpirv[0])), C.int(len(siluBwdSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: silu-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// unaryF32 returns a GPU kernel for a simple unary elementwise op (one Vulkan invocation
// per element, selected by `sel`), keeping activations resident on the GPU. f32-only;
// empty tensors fall back to the reference (§I4). GELU (§T352) now runs here too via the
// Abramowitz-Stegun erf approximation (GLSL has no erf).
func unaryF32(op backend.Op, sel int) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		refFallback := func() ([]*tensor.Tensor, error) {
			return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), op, in, attrs)
		}
		if len(in) != 1 {
			return nil, fmt.Errorf("vulkan: unary op wants 1 input, got %d", len(in))
		}
		x := in[0]
		n := x.Numel()
		if x.Dtype() != tensor.F32 || n == 0 || len(unarySpirv) == 0 {
			return refFallback()
		}
		xc := x.Contiguous()
		out := tensor.New(tensor.F32, x.Shape())
		rc := C.vk_unary_f32(
			(*C.uint32_t)(unsafe.Pointer(&unarySpirv[0])), C.int(len(unarySpirv)),
			(*C.float)(&xc.Storage().F32()[0]),
			(*C.float)(&out.Storage().F32()[0]),
			C.int(n), C.int(sel),
		)
		if rc != 0 {
			return nil, fmt.Errorf("vulkan: unary op failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{out}, nil
	}
}

// DeviceBuffer is a persistent host-visible VkBuffer (§T381, mirrors the Metal backend's
// DeviceBuffer §T367): data uploaded once, read back with DownloadF32, overwritten with UploadF32,
// freed with Release. It is the storage the batched-decode Recorder (Vulkan port of ADR-0019
// Phase 2) will chain ops over so intermediates stay device-resident.
type DeviceBuffer struct {
	handle unsafe.Pointer
	n      int // element count (f32)
}

// NewDeviceBufferF32 uploads data into a device-resident buffer.
func NewDeviceBufferF32(data []float32) (*DeviceBuffer, error) {
	if !Available() {
		return nil, fmt.Errorf("vulkan: no GPU (§V4)")
	}
	if len(data) == 0 {
		return &DeviceBuffer{}, nil
	}
	h := C.vk_devbuf_upload(unsafe.Pointer(&data[0]), C.int(len(data)*4))
	if h == nil {
		return nil, fmt.Errorf("vulkan: devbuf upload failed")
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
		return fmt.Errorf("vulkan: devbuf download dst len %d > %d", len(dst), b.n)
	}
	rc := C.vk_devbuf_download(b.handle, unsafe.Pointer(&dst[0]), C.int(len(dst)*4))
	if rc != 0 {
		return fmt.Errorf("vulkan: devbuf download failed (code %d)", int(rc))
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
		return fmt.Errorf("vulkan: devbuf upload src len %d > %d", len(src), b.n)
	}
	rc := C.vk_devbuf_upload_into(b.handle, unsafe.Pointer(&src[0]), C.int(len(src)*4))
	if rc != 0 {
		return fmt.Errorf("vulkan: devbuf upload failed (code %d)", int(rc))
	}
	return nil
}

// Release frees the device buffer. Safe to call more than once.
func (b *DeviceBuffer) Release() {
	if b.handle != nil {
		C.vk_devbuf_free(b.handle)
		b.handle = nil
	}
}

// flashAttnHost runs the Vulkan flash attention kernel over host slices (unexported; the §T396
// A/B baseline). Q,O are [seq,dm]; K,V are [seq,kvHeads*dk].
func flashAttnHost(q, k, v []float32, seq, dm, heads, dk, causal, kvHeads int, scale float32) ([]float32, error) {
	o := make([]float32, seq*dm)
	rc := C.vk_flashattn_f32(
		(*C.uint32_t)(unsafe.Pointer(&flashAttnSpirv[0])), C.int(len(flashAttnSpirv)),
		(*C.float)(&q[0]), (*C.float)(&k[0]), (*C.float)(&v[0]), (*C.float)(&o[0]),
		C.int(seq), C.int(dm), C.int(heads), C.int(dk), C.int(causal), C.int(kvHeads), C.float(scale))
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: flashattn failed (%d)", int(rc))
	}
	return o, nil
}

// Recorder holds one open Vulkan command buffer that record-mode ops (MatMul, RMSNorm, LayerNorm,
// RoPE, MHA, Unary, Binary, Blit) encode dispatches into over DeviceBuffers without a per-op
// submit/waitIdle; Finish submits once with a single waitIdle. Explicit barriers between ops stand
// in for Metal's automatic hazard tracking (§T382, ADR-0019 — the batched-decode engine, ~6× faster
// than per-op dispatch at realistic decode sizes, §T390).
type Recorder struct{ handle unsafe.Pointer }

// NewRecorder opens a Recorder (a fresh command buffer). Record ops into it, then Finish to submit
// and wait, and Free to release it. A single global recorder backs it (reset on each begin), so it
// is not safe for concurrent use — drive one Recorder at a time.
func NewRecorder() (*Recorder, error) {
	h := C.vk_recorder_begin()
	if h == nil {
		return nil, fmt.Errorf("vulkan: Recorder begin failed")
	}
	return &Recorder{handle: h}, nil
}

// Unary records O = f(op, X) over the device buffers x, o into the command buffer.
func (r *Recorder) Unary(x, o *DeviceBuffer, op int) error {
	if x.n != o.n {
		return fmt.Errorf("vulkan: Recorder unary size %d != %d", x.n, o.n)
	}
	rc := C.vk_recorder_unary(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&unarySpirv[0])), C.int(len(unarySpirv)),
		x.handle, o.handle, C.int(x.n), C.int(op))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder unary failed (%d)", int(rc))
	}
	return nil
}

// Binary records O = a (op) b over n elements into the command buffer over device buffers
// (op 0=add, 1=sub, 2=mul, 3=div, 4=max, 5=min).
func (r *Recorder) Binary(a, b, o *DeviceBuffer, op int) error {
	if a.n != o.n || b.n != o.n {
		return fmt.Errorf("vulkan: Recorder binary size %d/%d != %d", a.n, b.n, o.n)
	}
	rc := C.vk_recorder_binary(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&binarySpirv[0])), C.int(len(binarySpirv)),
		a.handle, b.handle, o.handle, C.int(o.n), C.int(op))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder binary failed (%d)", int(rc))
	}
	return nil
}

// MatMul records C = A·B (M×K · K×N → M×N) into the command buffer over device buffers.
// Forward-only (no transpose). c must be distinct from a and b.
func (r *Recorder) MatMul(a, b, c *DeviceBuffer, m, k, n int) error {
	// >= not ==: operands may be over-allocated scratch (§T418 StepN); the kernel reads/writes
	// only the leading m*k/k*n/m*n elements.
	if a.n < m*k || b.n < k*n || c.n < m*n {
		return fmt.Errorf("vulkan: Recorder matmul shape mismatch: a=%d(want ≥%d) b=%d(want ≥%d) c=%d(want ≥%d)", a.n, m*k, b.n, k*n, c.n, m*n)
	}
	rc := C.vk_recorder_matmul(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&spirv[0])), C.int(len(spirv)),
		a.handle, b.handle, c.handle, C.int(m), C.int(k), C.int(n))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder matmul failed (%d)", int(rc))
	}
	return nil
}

// RMSNorm records O = rmsnorm(X)·gamma into the command buffer over device buffers.
// x, o are rows×dim; g is the dim-length gamma weight.
func (r *Recorder) RMSNorm(x, g, o *DeviceBuffer, rows, dim int, eps float32) error {
	if x.n < rows*dim || o.n < rows*dim || g.n < dim {
		return fmt.Errorf("vulkan: Recorder rmsnorm shape mismatch: x=%d o=%d (want %d) g=%d (want %d)", x.n, o.n, rows*dim, g.n, dim)
	}
	rc := C.vk_recorder_rmsnorm(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&rmsnormSpirv[0])), C.int(len(rmsnormSpirv)),
		x.handle, g.handle, o.handle, C.int(rows), C.int(dim), C.float(eps))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder rmsnorm failed (%d)", int(rc))
	}
	return nil
}

// RoPE records O = RoPE(Q) into the command buffer over device buffers. q, o are [seq,width];
// inv holds `half` inverse frequencies. posOffset is the query's absolute start position.
func (r *Recorder) RoPE(q, inv, o *DeviceBuffer, seq, width, heads, hd, half, posOffset int, posDiv float32) error {
	if q.n < seq*width || o.n < seq*width || inv.n < half {
		return fmt.Errorf("vulkan: Recorder rope shape mismatch: q=%d o=%d (want %d) inv=%d (want %d)", q.n, o.n, seq*width, inv.n, half)
	}
	rc := C.vk_recorder_rope(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&ropeSpirv[0])), C.int(len(ropeSpirv)),
		q.handle, inv.handle, o.handle,
		C.int(seq), C.int(width), C.int(heads), C.int(hd), C.int(half), C.int(posOffset), C.float(posDiv))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder rope failed (%d)", int(rc))
	}
	return nil
}

// MHA records O = attention(Q,K,V) into the command buffer with separate query/key lengths —
// the decode-against-KV-cache shape (sq=1 query row, sk=cache length). q, o are [sq,dm]; k, v are
// [sk,kvHeads*dk] and may be over-allocated caches (only the first sk rows are read).
func (r *Recorder) MHA(q, k, v, o *DeviceBuffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	if q.n < sq*dm || o.n < sq*dm || k.n < sk*kvHeads*dk || v.n < sk*kvHeads*dk {
		return fmt.Errorf("vulkan: Recorder mha shape mismatch: q=%d o=%d (want %d) k=%d v=%d (need >=%d)", q.n, o.n, sq*dm, k.n, v.n, sk*kvHeads*dk)
	}
	// Cooperative fast path (§T429 decode, §T431 generalized to sq>1 for prefill windows): one
	// 32-lane subgroup per (query row, head); per-row causal bound jmax = sk-sq+i+1. Window stays
	// on the two-pass kernel. At sq==1 causal is irrelevant, so non-causal decode routes here too.
	if window == 0 && dk <= 128 && (causal != 0 || sq == 1) && len(mhaDecodeSpirv) > 0 {
		rc := C.vk_recorder_mha_decode(r.handle,
			(*C.uint32_t)(unsafe.Pointer(&mhaDecodeSpirv[0])), C.int(len(mhaDecodeSpirv)),
			q.handle, k.handle, v.handle, o.handle,
			C.int(sq), C.int(sk), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk), C.int(causal), C.float(scale))
		if rc != 0 {
			return fmt.Errorf("vulkan: Recorder mha (decode path) failed (%d)", int(rc))
		}
		return nil
	}
	rc := C.vk_recorder_mha(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&mhaSpirv[0])), C.int(len(mhaSpirv)),
		q.handle, k.handle, v.handle, o.handle,
		C.int(sq), C.int(sk), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk), C.int(causal), C.int(window), C.float(scale))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder mha failed (%d)", int(rc))
	}
	return nil
}

// matmulStrided records C = alpha·A·(B/Bᵀ) with per-operand row strides + element offsets (the
// §T397 attention matmuls over windows of packed Q/K/V/O buffers).
func (r *Recorder) matmulStrided(a, b, c *DeviceBuffer, m, k, n, transA, transB, lda, ldb, ldc, offA, offB, offC int, alpha float32) error {
	rc := C.vk_recorder_matmul_strided(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&matmulStridedSpirv[0])), C.int(len(matmulStridedSpirv)),
		a.handle, b.handle, c.handle,
		C.int(m), C.int(k), C.int(n), C.int(transA), C.int(transB),
		C.int(lda), C.int(ldb), C.int(ldc), C.int(offA), C.int(offB), C.int(offC), C.float(alpha))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder matmulStrided failed (%d)", int(rc))
	}
	return nil
}

// softmaxCausal records an in-place row softmax of s[rows,cols] with an optional causal mask.
func (r *Recorder) softmaxCausal(s *DeviceBuffer, rows, cols, causal int) error {
	rc := C.vk_recorder_softmax_causal(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&softmaxCausalSpirv[0])), C.int(len(softmaxCausalSpirv)),
		s.handle, C.int(rows), C.int(cols), C.int(causal))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder softmaxCausal failed (%d)", int(rc))
	}
	return nil
}

// attnBufCache reuses the attention device buffers across calls (§T398): the GPT forward calls
// attention with a fixed shape every layer/step, so per-call NewDeviceBufferF32 (vkCreateBuffer+
// vkAllocateMemory) was pure churn (§T397 root cause). Keyed by the exact shape; single active set
// (attention is driven single-threaded here). Not concurrency-safe — one caller at a time.
type attnBufs struct {
	seq, dm, kvDim int
	q, k, v, s, o  *DeviceBuffer
}

var attnCache *attnBufs

func acquireAttnBufs(seq, dm, kvDim int) (*attnBufs, error) {
	if attnCache != nil && attnCache.seq == seq && attnCache.dm == dm && attnCache.kvDim == kvDim {
		return attnCache, nil
	}
	if attnCache != nil {
		attnCache.q.Release()
		attnCache.k.Release()
		attnCache.v.Release()
		attnCache.s.Release()
		attnCache.o.Release()
		attnCache = nil
	}
	mk := func(n int) (*DeviceBuffer, error) { return NewDeviceBufferF32(make([]float32, n)) }
	q, err := mk(seq * dm)
	if err != nil {
		return nil, err
	}
	k, _ := mk(seq * kvDim)
	v, _ := mk(seq * kvDim)
	s, _ := mk(seq * seq)
	o, _ := mk(seq * dm)
	attnCache = &attnBufs{seq, dm, kvDim, q, k, v, s, o}
	return attnCache, nil
}

// mhaMatmul runs the §T397 MPS-analog attention forward on Vulkan over host slices: per head,
// S=scale·Q_h·K_hᵀ (strided matmul) → causal softmax → O_h=S·V_h (strided matmul), all in one
// Recorder command buffer over CACHED device buffers (§T398). Q,O are [seq,dm]; K,V are [seq,kvHeads*dk].
func mhaMatmul(q, k, v []float32, seq, dm, heads, dk, causal, kvHeads int, scale float32) ([]float32, error) {
	kvDim := kvHeads * dk
	rep := heads / kvHeads
	bufs, err := acquireAttnBufs(seq, dm, kvDim)
	if err != nil {
		return nil, err
	}
	qb, kb, vb, sb, ob := bufs.q, bufs.k, bufs.v, bufs.s, bufs.o
	if err := qb.UploadF32(q); err != nil {
		return nil, err
	}
	if err := kb.UploadF32(k); err != nil {
		return nil, err
	}
	if err := vb.UploadF32(v); err != nil {
		return nil, err
	}

	r, err := NewRecorder()
	if err != nil {
		return nil, err
	}
	for h := 0; h < heads; h++ {
		hh := h / rep
		// S = scale · Q_h · K_hᵀ : A=Q offA=h*dk lda=dm ; B=K offB=hh*dk ldb=kvDim transB=1 ; C=S ldc=seq
		if err := r.matmulStrided(qb, kb, sb, seq, dk, seq, 0, 1, dm, kvDim, seq, h*dk, hh*dk, 0, scale); err != nil {
			r.Free()
			return nil, err
		}
		if err := r.softmaxCausal(sb, seq, seq, causal); err != nil {
			r.Free()
			return nil, err
		}
		// O_h = S · V_h : A=S lda=seq ; B=V offB=hh*dk ldb=kvDim transB=0 ; C=O offC=h*dk ldc=dm
		if err := r.matmulStrided(sb, vb, ob, seq, seq, dk, 0, 0, seq, kvDim, dm, 0, hh*dk, h*dk, 1.0); err != nil {
			r.Free()
			return nil, err
		}
	}
	if err := r.Finish(); err != nil {
		r.Free()
		return nil, err
	}
	r.Free()
	out := make([]float32, seq*dm)
	if err := ob.DownloadF32(out); err != nil {
		return nil, err
	}
	return out, nil
}

// Blit records a copy of n f32 elements from src[srcOff:] to dst[dstOff:] into the command buffer
// (offsets and count in f32 elements) — the KV-cache append: copy the new token's k/v row into
// cache[cacheLen] each decode step, staying device-resident.
func (r *Recorder) Blit(src *DeviceBuffer, srcOff int, dst *DeviceBuffer, dstOff, n int) error {
	if srcOff < 0 || dstOff < 0 || n <= 0 || srcOff+n > src.n || dstOff+n > dst.n {
		return fmt.Errorf("vulkan: Recorder blit out of range: src[%d:%d]/%d dst[%d:%d]/%d", srcOff, srcOff+n, src.n, dstOff, dstOff+n, dst.n)
	}
	const f32 = 4
	rc := C.vk_recorder_blit(r.handle, src.handle, C.int(srcOff*f32), dst.handle, C.int(dstOff*f32), C.int(n*f32))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder blit failed (%d)", int(rc))
	}
	return nil
}

// LayerNorm records O = layernorm(X)·gamma + beta into the command buffer over device buffers
// (GPT-2-style norm; rmsnorm is the Llama variant). x, o are rows×dim; g, b are dim-length.
func (r *Recorder) LayerNorm(x, g, b, o *DeviceBuffer, rows, dim int, eps float32) error {
	if x.n < rows*dim || o.n < rows*dim || g.n < dim || b.n < dim {
		return fmt.Errorf("vulkan: Recorder layernorm shape mismatch: x=%d o=%d (want %d) g=%d b=%d (want %d)", x.n, o.n, rows*dim, g.n, b.n, dim)
	}
	rc := C.vk_recorder_layernorm(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&layernormSpirv[0])), C.int(len(layernormSpirv)),
		x.handle, g.handle, b.handle, o.handle, C.int(rows), C.int(dim), C.float(eps))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder layernorm failed (%d)", int(rc))
	}
	return nil
}

// QMatMulResident records O = X·dequant(W)ᵀ into the command buffer, where W is a resident
// quantized [N,K] weight (UploadQuant) and X, O are device buffers ([m,K] → [m,N]). The quantized
// batched-decode building block (§T414, parity with metal §T413).
func (r *Recorder) QMatMulResident(x *DeviceBuffer, w *ResidentQWeight, o *DeviceBuffer, m int) error {
	if w == nil || w.handle == nil {
		return fmt.Errorf("vulkan: Recorder qmatmul: resident weight is nil/closed")
	}
	spv := residentSpirv(w.qt)
	if len(spv) == 0 {
		return fmt.Errorf("vulkan: Recorder qmatmul: no resident shader for quant type %d", w.qt)
	}
	if x.n < m*w.k || o.n < m*w.n {
		return fmt.Errorf("vulkan: Recorder qmatmul shape mismatch: x=%d (want %d) o=%d (want %d)", x.n, m*w.k, o.n, m*w.n)
	}
	rc := C.vk_recorder_qmatmul(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&spv[0])), C.int(len(spv)),
		x.handle, w.handle, o.handle,
		C.int(m), C.int(w.k), C.int(w.n), C.int(w.wBytes))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder qmatmul failed (%d)", int(rc))
	}
	return nil
}

// AddBias records O[i,j] = X[i,j] + B[j] into the command buffer over device buffers (x, o are
// rows×n; b is the n-length bias) — the broadcast bias-add multi-token GPT steps need (§T423).
func (r *Recorder) AddBias(x, b, o *DeviceBuffer, rows, n int) error {
	if x.n < rows*n || o.n < rows*n || b.n < n {
		return fmt.Errorf("vulkan: Recorder addbias shape mismatch: x=%d o=%d (want ≥%d) b=%d (want ≥%d)", x.n, o.n, rows*n, b.n, n)
	}
	rc := C.vk_recorder_addbias(r.handle,
		(*C.uint32_t)(unsafe.Pointer(&addBiasSpirv[0])), C.int(len(addBiasSpirv)),
		x.handle, b.handle, o.handle, C.int(rows), C.int(n))
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder addbias failed (%d)", int(rc))
	}
	return nil
}

// Finish submits the command buffer and waits.
func (r *Recorder) Finish() error {
	rc := C.vk_recorder_finish(r.handle)
	if rc != 0 {
		return fmt.Errorf("vulkan: Recorder finish failed (%d)", int(rc))
	}
	return nil
}

func (r *Recorder) Free() {
	if r.handle != nil {
		C.vk_recorder_free(r.handle)
		r.handle = nil
	}
}

// batchDispatchBench (§T380) runs the unary ReLU kernel iters× either batched into one command
// buffer (oneBuffer=true) or per-op (false), to measure the Vulkan/MoltenVK command-buffer
// batching win before porting the whole Recorder. Unexported; called from an internal bench test.
func batchDispatchBench(iters int, oneBuffer bool, n int) error {
	one := 0
	if oneBuffer {
		one = 1
	}
	rc := C.vk_batch_dispatch_bench(
		(*C.uint32_t)(unsafe.Pointer(&unarySpirv[0])), C.int(len(unarySpirv)),
		C.int(iters), C.int(one), C.int(n),
	)
	if rc != 0 {
		return fmt.Errorf("vulkan: batch dispatch bench failed (code %d)", int(rc))
	}
	return nil
}

// layernormF32 computes LayerNorm on the GPU (torch semantics): y = (x−mean)/√(var+eps)·γ + β
// over the last axis, one Vulkan invocation per row. Inputs (x[...,d], gamma[d], beta[d]);
// f32-only, falling back to the reference (§I4) for empty tensors. Complements the GPU RMSNorm
// (§T324) so LayerNorm-based models keep activations resident between matmuls too.
func layernormF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpLayerNorm, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("vulkan: layernorm wants (x, gamma, beta), got %d inputs", len(in))
	}
	x, gamma, beta := in[0], in[1], in[2]
	if x.Dtype() != tensor.F32 || gamma.Dtype() != tensor.F32 || beta.Dtype() != tensor.F32 || x.Ndim() < 1 {
		return refFallback()
	}
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d || beta.Ndim() != 1 || beta.Shape()[0] != d {
		return nil, fmt.Errorf("vulkan: layernorm gamma/beta must be [%d], got %v/%v", d, gamma.Shape(), beta.Shape())
	}
	rows := 0
	if d > 0 {
		rows = x.Numel() / d
	}
	if rows == 0 || d == 0 || len(layernormSpirv) == 0 {
		return refFallback()
	}
	pa, _ := attrs.(backend.NormAttrs)
	pa = pa.WithDefaults()
	xc, gc, bc := x.Contiguous(), gamma.Contiguous(), beta.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.vk_layernorm_f32(
		(*C.uint32_t)(unsafe.Pointer(&layernormSpirv[0])), C.int(len(layernormSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&bc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(rows), C.int(d), C.float(pa.Eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: layernorm failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// embedBackwardF32 computes the embedding gradient on the GPU (§T34): scatter-add g[i,:] into
// dtable[idx[i],:] with float atomics (tokens sharing an index collide). It makes the token/
// position-embedding gradient run on the GPU during training. g must be f32; idx is read via
// AtF64 into an f32 buffer (any dtype). Empty/non-f32 g or no float atomics → ref (§I4).
func embedBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpEmbedBackward, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("vulkan: embed-backward wants (table, idx, g), got %d inputs", len(in))
	}
	table, idx, g := in[0], in[1], in[2]
	if table.Ndim() != 2 || idx.Ndim() != 1 || g.Dtype() != tensor.F32 {
		return refFallback()
	}
	n, d := table.Shape()[0], table.Shape()[1]
	m := idx.Shape()[0]
	if !g.Shape().Equal(tensor.Shape{m, d}) {
		return nil, fmt.Errorf("vulkan: embed-backward g must be [%d,%d], got %v", m, d, g.Shape())
	}
	if m == 0 || d == 0 || n == 0 || len(embedBwdSpirv) == 0 {
		return refFallback()
	}
	idxF := make([]float32, m)
	for i := range idxF {
		t := int(idx.AtF64(i))
		if t < 0 || t >= n {
			return nil, fmt.Errorf("vulkan: embed-backward index %d out of range [0,%d)", t, n)
		}
		idxF[i] = float32(t)
	}
	gc := g.Contiguous()
	dtable := tensor.New(tensor.F32, table.Shape()) // zero-init atomic accumulator
	rc := C.vk_embed_backward_f32(
		(*C.uint32_t)(unsafe.Pointer(&embedBwdSpirv[0])), C.int(len(embedBwdSpirv)),
		(*C.float)(&idxF[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&dtable.Storage().F32()[0]),
		C.int(n), C.int(d), C.int(m),
	)
	if int(rc) == -7 {
		return refFallback() // no float atomics
	}
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: embed-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dtable}, nil
}

// crossentropyBackwardF32 computes the cross-entropy gradient on the GPU (ADR-0007), one
// Vulkan invocation per row: dz = gv·(softmax(z) − q')/b (+z-loss). It makes the loss
// gradient — which seeds the whole backward pass — run on the GPU. Logits z must be f32;
// targets are read via AtF64 into an f32 buffer (any dtype). Empty/non-f32 z → ref (§I4).
// crossentropyF32 is the fused cross-entropy FORWARD on the GPU (§T403, parity with metal §T401):
// per row lse−z[target], meaned over the batch. Basic case only; label-smoothing/z-loss → ref.
func crossentropyF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpCrossEntropy, in, attrs)
	}
	if len(in) != 2 {
		return nil, fmt.Errorf("vulkan: crossentropy wants (logits, targets), got %d inputs", len(in))
	}
	z, tg := in[0], in[1]
	if z.Dtype() != tensor.F32 || z.Ndim() != 2 || tg.Ndim() != 1 || len(crossEntropySpirv) == 0 {
		return refFallback()
	}
	pa, _ := attrs.(backend.CrossEntropyAttrs)
	if pa.LabelSmoothing != 0 || pa.ZLoss != 0 {
		return refFallback()
	}
	b, c := z.Shape()[0], z.Shape()[1]
	if tg.Shape()[0] != b {
		return nil, fmt.Errorf("vulkan: crossentropy targets len %d != batch %d", tg.Shape()[0], b)
	}
	if b == 0 || c == 0 {
		return refFallback()
	}
	tf := make([]float32, b)
	for i := range tf {
		ti := int(tg.AtF64(i))
		if ti < 0 || ti >= c {
			return nil, fmt.Errorf("vulkan: crossentropy target %d out of range [0,%d)", ti, c)
		}
		tf[i] = float32(ti)
	}
	zc := z.Contiguous()
	losses := make([]float32, b)
	rc := C.vk_crossentropy_f32(
		(*C.uint32_t)(unsafe.Pointer(&crossEntropySpirv[0])), C.int(len(crossEntropySpirv)),
		(*C.float)(&zc.Storage().F32()[0]), (*C.float)(&tf[0]), (*C.float)(&losses[0]),
		C.int(b), C.int(c))
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: crossentropy failed (code %d)", int(rc))
	}
	var total float64
	for _, l := range losses {
		total += float64(l)
	}
	out := tensor.NewOn(ctx.Device(), z.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

// embedF32 is the token-embedding row gather (§T403, host-side since §T410 — same finding as
// metal: a gather is tiny, and uploading the whole [vocab,dim] table per call cost more than the
// entire gather; a direct f32 host copy beats both the GPU kernel and the ref fallback).
func embedF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpEmbed, in, attrs)
	}
	if len(in) != 2 {
		return nil, fmt.Errorf("vulkan: embed wants (table, idx), got %d inputs", len(in))
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
			return nil, fmt.Errorf("vulkan: embed index %d out of range [0,%d)", r, n)
		}
		copy(os[i*d:(i+1)*d], ts[r*d:(r+1)*d])
	}
	return []*tensor.Tensor{out}, nil
}

func crossentropyBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpCrossEntropyBackward, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("vulkan: crossentropy-backward wants (z, targets, g), got %d inputs", len(in))
	}
	z, tg, g := in[0], in[1], in[2]
	if z.Dtype() != tensor.F32 || z.Ndim() != 2 || tg.Ndim() != 1 {
		return refFallback()
	}
	b, c := z.Shape()[0], z.Shape()[1]
	if tg.Shape()[0] != b {
		return nil, fmt.Errorf("vulkan: crossentropy-backward targets len %d != batch %d", tg.Shape()[0], b)
	}
	if b == 0 || c == 0 || len(crossentropyBwdSpirv) == 0 {
		return refFallback()
	}
	pa, _ := attrs.(backend.CrossEntropyAttrs)
	zc := z.Contiguous()
	tf := make([]float32, b) // targets → f32 (indices), independent of tg's dtype
	for i := range tf {
		tf[i] = float32(tg.AtF64(i))
	}
	dz := tensor.New(tensor.F32, z.Shape())
	rc := C.vk_crossentropy_backward_f32(
		(*C.uint32_t)(unsafe.Pointer(&crossentropyBwdSpirv[0])), C.int(len(crossentropyBwdSpirv)),
		(*C.float)(&zc.Storage().F32()[0]),
		(*C.float)(&tf[0]),
		(*C.float)(&dz.Storage().F32()[0]),
		C.int(b), C.int(c), C.float(g.AtF64()), C.float(pa.LabelSmoothing), C.float(pa.ZLoss),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: crossentropy-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dz}, nil
}

// softmaxF32 computes a numerically-stable softmax over the last axis on the GPU (§V12),
// one Vulkan invocation per row. f32-only; empty tensors fall back to the reference (§I4).
// Keeps logits/scores resident on the GPU (e.g. before sampling) instead of a CPU round-trip.
func softmaxF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpSoftmax, in, attrs)
	}
	if len(in) != 1 {
		return nil, fmt.Errorf("vulkan: softmax wants 1 input, got %d", len(in))
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
	if rows == 0 || d == 0 || len(softmaxSpirv) == 0 {
		return refFallback()
	}
	xc := x.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.vk_softmax_f32(
		(*C.uint32_t)(unsafe.Pointer(&softmaxSpirv[0])), C.int(len(softmaxSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(rows), C.int(d),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: softmax failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// ropeKernelF32 returns the GPU kernel for RoPE forward (OpRoPE, spv=rope.spv) or its
// backward (OpRoPEBackward, spv=rope_bwd.spv, the inverse rotation) on the split-half rule
// (§R28/§R92). Both take one rank-2 tensor (q or the upstream gradient g) and share the
// identical launcher: the per-head inverse frequencies (linear-PI §R64 / YaRN §R66) and
// position divisor are precomputed on the host by backend.RoPEFreqs, so only the rotation
// runs on the GPU — keeping Q/K (forward) or the gradient (backward) resident. f32-only;
// the xPos magnitude variant (§R125) and empty tensors fall back to the reference (§I4).
func ropeKernelF32(op backend.Op, spv []byte) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		refFallback := func() ([]*tensor.Tensor, error) {
			return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), op, in, attrs)
		}
		if len(in) != 1 {
			return nil, fmt.Errorf("vulkan: rope wants 1 input, got %d", len(in))
		}
		q := in[0]
		pr, _ := attrs.(backend.RoPEAttrs)
		if q.Dtype() != tensor.F32 || q.Ndim() != 2 || pr.XPos {
			return refFallback() // non-f32 / non-2-D / xPos → reference
		}
		seq, width := q.Shape()[0], q.Shape()[1]
		heads := pr.Heads
		if heads <= 0 {
			heads = 1
		}
		if width%heads != 0 {
			return nil, fmt.Errorf("vulkan: rope width %d not divisible by heads %d", width, heads)
		}
		hd := width / heads
		if hd%2 != 0 {
			return nil, fmt.Errorf("vulkan: rope head dim %d must be even", hd)
		}
		half := hd / 2
		if seq == 0 || width == 0 || len(spv) == 0 {
			return refFallback()
		}
		inv, posDiv := backend.RoPEFreqs(hd, pr) // host-side freqs (PI/YaRN folded in)
		invF32 := make([]float32, half)
		for i := range invF32 {
			invF32[i] = float32(inv[i])
		}
		qc := q.Contiguous()
		out := tensor.New(tensor.F32, q.Shape())
		rc := C.vk_rope_f32(
			(*C.uint32_t)(unsafe.Pointer(&spv[0])), C.int(len(spv)),
			(*C.float)(&qc.Storage().F32()[0]),
			(*C.float)(&invF32[0]),
			(*C.float)(&out.Storage().F32()[0]),
			C.int(seq), C.int(width), C.int(heads), C.int(hd), C.int(half), C.int(pr.PosOffset), C.float(posDiv),
		)
		if rc != 0 {
			return nil, fmt.Errorf("vulkan: rope failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{out}, nil
	}
}

// layernormBackwardF32 computes the LayerNorm gradient on the GPU (§R35): (x,gamma,g) →
// (dx, dgamma, dbeta), one Vulkan invocation per row (dx per-row, dgamma/dbeta via float
// atomics). It makes LayerNorm TRAINING run on the GPU. f32-only; empty tensors and a device
// without float atomics fall back to the reference (§I4).
func layernormBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpLayerNormBackward, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("vulkan: layernorm-backward wants (x, gamma, g), got %d inputs", len(in))
	}
	x, gamma, g := in[0], in[1], in[2]
	if x.Dtype() != tensor.F32 || gamma.Dtype() != tensor.F32 || g.Dtype() != tensor.F32 || x.Ndim() < 1 {
		return refFallback()
	}
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d || !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("vulkan: layernorm-backward gamma [%d]/g %v mismatch x %v", d, g.Shape(), x.Shape())
	}
	rows := 0
	if d > 0 {
		rows = x.Numel() / d
	}
	if rows == 0 || d == 0 || len(layernormBwdSpirv) == 0 {
		return refFallback()
	}
	pa, _ := attrs.(backend.NormAttrs)
	pa = pa.WithDefaults()
	xc, gc, gupc := x.Contiguous(), gamma.Contiguous(), g.Contiguous()
	dx := tensor.New(tensor.F32, x.Shape())
	dgamma := tensor.New(tensor.F32, gamma.Shape()) // zero-init atomic accumulators
	dbeta := tensor.New(tensor.F32, gamma.Shape())
	rc := C.vk_layernorm_backward_f32(
		(*C.uint32_t)(unsafe.Pointer(&layernormBwdSpirv[0])), C.int(len(layernormBwdSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&gupc.Storage().F32()[0]),
		(*C.float)(&dx.Storage().F32()[0]),
		(*C.float)(&dgamma.Storage().F32()[0]),
		(*C.float)(&dbeta.Storage().F32()[0]),
		C.int(rows), C.int(d), C.float(pa.Eps),
	)
	if int(rc) == -7 {
		return refFallback() // no float atomics
	}
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: layernorm-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dx, dgamma, dbeta}, nil
}

// rmsnormBackwardF32 computes the RMSNorm gradient on the GPU (§R29/§R35): (x,gamma,g) →
// (dx, dgamma), one Vulkan invocation per row (dx per-row, dgamma via float atomics). It
// makes RMSNorm TRAINING run on the GPU. f32-only; empty tensors and a device without
// float atomics fall back to the reference (§I4).
func rmsnormBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpRMSNormBackward, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("vulkan: rmsnorm-backward wants (x, gamma, g), got %d inputs", len(in))
	}
	x, gamma, g := in[0], in[1], in[2]
	if x.Dtype() != tensor.F32 || gamma.Dtype() != tensor.F32 || g.Dtype() != tensor.F32 || x.Ndim() < 1 {
		return refFallback()
	}
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d || !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("vulkan: rmsnorm-backward gamma [%d]/g %v mismatch x %v", d, g.Shape(), x.Shape())
	}
	rows := 0
	if d > 0 {
		rows = x.Numel() / d
	}
	if rows == 0 || d == 0 || len(rmsnormBwdSpirv) == 0 {
		return refFallback()
	}
	pa, _ := attrs.(backend.NormAttrs)
	pa = pa.WithDefaults()
	xc, gc, gupc := x.Contiguous(), gamma.Contiguous(), g.Contiguous()
	dx := tensor.New(tensor.F32, x.Shape())
	dgamma := tensor.New(tensor.F32, gamma.Shape()) // zero-initialized: atomic accumulator
	rc := C.vk_rmsnorm_backward_f32(
		(*C.uint32_t)(unsafe.Pointer(&rmsnormBwdSpirv[0])), C.int(len(rmsnormBwdSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&gupc.Storage().F32()[0]),
		(*C.float)(&dx.Storage().F32()[0]),
		(*C.float)(&dgamma.Storage().F32()[0]),
		C.int(rows), C.int(d), C.float(pa.Eps),
	)
	if int(rc) == -7 {
		return refFallback() // no float atomics
	}
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: rmsnorm-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dx, dgamma}, nil
}

// rmsnormF32 computes RMSNorm on the GPU (§R29): y = x/√(mean(x²)+eps)·γ over the last
// axis, one Vulkan invocation per row. Inputs (x[...,d], gamma[d]); f32-only, falling
// back to the reference (§I4) for empty tensors. Keeps activations on the GPU between
// matmuls in an LLM forward pass instead of bouncing norms through the CPU.
func rmsnormF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpRMSNorm, in, attrs)
	}
	if len(in) != 2 {
		return nil, fmt.Errorf("vulkan: rmsnorm wants (x, gamma), got %d inputs", len(in))
	}
	x, gamma := in[0], in[1]
	if x.Dtype() != tensor.F32 || gamma.Dtype() != tensor.F32 || x.Ndim() < 1 {
		return refFallback()
	}
	d := x.Shape()[x.Ndim()-1]
	if gamma.Ndim() != 1 || gamma.Shape()[0] != d {
		return nil, fmt.Errorf("vulkan: rmsnorm gamma must be [%d], got %v", d, gamma.Shape())
	}
	rows := 0
	if d > 0 {
		rows = x.Numel() / d
	}
	if rows == 0 || d == 0 || len(rmsnormSpirv) == 0 {
		return refFallback() // nothing to launch / no shader embedded
	}
	pa, _ := attrs.(backend.NormAttrs)
	pa = pa.WithDefaults()
	xc, gc := x.Contiguous(), gamma.Contiguous()
	out := tensor.New(tensor.F32, x.Shape())
	rc := C.vk_rmsnorm_f32(
		(*C.uint32_t)(unsafe.Pointer(&rmsnormSpirv[0])), C.int(len(rmsnormSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&gc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(rows), C.int(d), C.float(pa.Eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: rmsnorm failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// conv2dBackwardF32 is the conv2d backward via Vulkan compute (§T102, the portable
// twin of the Metal kernel §T101): (X,W,dO) → (dX,dW,dBias). One invocation per
// output-gradient element scatters into the shared dX/dW/dBias with float atomics.
// f32-only; falls back to the reference (§I4) for non-f32 or a device without float
// atomics. Dispatched by conv2d's VJP → CV training runs on the GPU.
func conv2dBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpConv2DBackward, in, attrs)
	}
	if len(in) != 3 {
		return nil, fmt.Errorf("vulkan: conv2d-backward wants (X,W,dO), got %d inputs", len(in))
	}
	x, w, dO := in[0], in[1], in[2]
	for _, t := range in {
		if t.Ndim() != 4 || t.Dtype() != tensor.F32 {
			return nil, fmt.Errorf("vulkan: conv2d-backward needs rank-4 f32")
		}
	}
	n, c, h, wd := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	f, wc, kh, kw := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]
	if wc != c {
		return nil, fmt.Errorf("vulkan: conv2d-backward channel mismatch x C=%d vs w C=%d", c, wc)
	}
	if C.vk_atomic_float() != 1 || len(conv2dBwdSpirv) == 0 {
		return refFallback()
	}
	ho, wo := dO.Shape()[2], dO.Shape()[3]
	pa, _ := attrs.(backend.ConvAttrs)
	pa = pa.WithDefaults()

	xc, wcont, dc := x.Contiguous(), w.Contiguous(), dO.Contiguous()
	dX := tensor.New(tensor.F32, x.Shape()) // zero-initialized (atomic accumulators)
	dW := tensor.New(tensor.F32, w.Shape())
	dB := tensor.New(tensor.F32, tensor.Shape{f})
	rc := C.vk_conv2d_backward_f32(
		(*C.uint32_t)(unsafe.Pointer(&conv2dBwdSpirv[0])), C.int(len(conv2dBwdSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&wcont.Storage().F32()[0]),
		(*C.float)(&dc.Storage().F32()[0]),
		(*C.float)(&dX.Storage().F32()[0]),
		(*C.float)(&dW.Storage().F32()[0]),
		(*C.float)(&dB.Storage().F32()[0]),
		C.int(n), C.int(c), C.int(h), C.int(wd), C.int(f), C.int(kh), C.int(kw),
		C.int(pa.Stride), C.int(pa.Pad), C.int(ho), C.int(wo),
	)
	if int(rc) == -7 {
		return refFallback()
	}
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: conv2d-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dX, dW, dB}, nil
}

// conv2dF32 is 2-D convolution on the GPU via Vulkan compute (§T91, the portable
// twin of the Metal kernel §T88): X[N,C,H,W] ⋆ W[F,C,KH,KW] (+ optional per-filter
// bias) → [N,F,ho,wo]. Compute-bound (§C3). f32-only; the reference serves other
// dtypes via the dispatch fallback (§I4). Forward only.
func conv2dF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 && len(in) != 3 {
		return nil, fmt.Errorf("vulkan: conv2d wants (x, w[, bias]), got %d inputs", len(in))
	}
	x, w := in[0], in[1]
	if x.Ndim() != 4 || w.Ndim() != 4 {
		return nil, fmt.Errorf("vulkan: conv2d needs x[N,C,H,W] and w[F,C,KH,KW], got %v/%v", x.Shape(), w.Shape())
	}
	if x.Dtype() != tensor.F32 || w.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: conv2d is f32-only")
	}
	n, c, h, wd := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
	f, wc, kh, kw := w.Shape()[0], w.Shape()[1], w.Shape()[2], w.Shape()[3]
	if wc != c {
		return nil, fmt.Errorf("vulkan: conv2d channel mismatch x C=%d vs w C=%d", c, wc)
	}
	bias := make([]float32, f) // kernel always adds B[f]; zeros when no bias
	if len(in) == 3 {
		b := in[2]
		if b.Ndim() != 1 || b.Shape()[0] != f || b.Dtype() != tensor.F32 {
			return nil, fmt.Errorf("vulkan: conv2d bias must be f32 [%d], got %v", f, b.Shape())
		}
		copy(bias, b.Contiguous().Storage().F32())
	}
	pa, _ := attrs.(backend.ConvAttrs)
	pa = pa.WithDefaults()
	s, p := pa.Stride, pa.Pad
	if s < 1 || p < 0 {
		return nil, fmt.Errorf("vulkan: conv2d invalid stride %d / pad %d", s, p)
	}
	ho := (h+2*p-kh)/s + 1
	wo := (wd+2*p-kw)/s + 1
	if ho < 1 || wo < 1 {
		return nil, fmt.Errorf("vulkan: conv2d output would be empty (%dx%d)", ho, wo)
	}
	if len(im2colSpirv) == 0 || len(coloutSpirv) == 0 || len(spirv) == 0 {
		return nil, fmt.Errorf("vulkan: conv2d shaders not embedded — run `make vulkan-spv` (glslc)")
	}

	xc, wcont := x.Contiguous(), w.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{n, f, ho, wo})
	rc := C.vk_conv2d_f32(
		(*C.uint32_t)(unsafe.Pointer(&im2colSpirv[0])), C.int(len(im2colSpirv)),
		(*C.uint32_t)(unsafe.Pointer(&spirv[0])), C.int(len(spirv)),
		(*C.uint32_t)(unsafe.Pointer(&coloutSpirv[0])), C.int(len(coloutSpirv)),
		(*C.float)(&xc.Storage().F32()[0]),
		(*C.float)(&wcont.Storage().F32()[0]),
		(*C.float)(&bias[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(n), C.int(c), C.int(h), C.int(wd), C.int(f), C.int(kh), C.int(kw),
		C.int(s), C.int(p), C.int(ho), C.int(wo),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: conv2d failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// mhaBackwardF32 is the GPU SDPA backward via Vulkan compute (§T90, the portable
// twin of the Metal kernel §T86): (Q,K,V,dO) → (dQ,dK,dV). It serves the same
// subset as the forward — including sliding window (§T129) — and falls back to the
// reference (§I4) for ALiBi / dk>128 / KV-cache (sq≠sk), or when the device lacks
// float atomics (VK_EXT_shader_atomic_float). Dispatched by the mha VJP, so attention
// training
// runs on the GPU when the tape's backend is vulkan.
func mhaBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	refFallback := func() ([]*tensor.Tensor, error) {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMHABackward, in, attrs)
	}
	if len(in) != 4 {
		return nil, fmt.Errorf("vulkan: mha-backward wants (Q,K,V,dO), got %d inputs", len(in))
	}
	q, k, v, dO := in[0], in[1], in[2], in[3]
	for _, x := range in {
		if x.Ndim() != 2 || x.Dtype() != tensor.F32 {
			return nil, fmt.Errorf("vulkan: mha-backward needs rank-2 f32")
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
		return nil, fmt.Errorf("vulkan: mha-backward dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := p.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("vulkan: mha-backward heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	dkv := kvHeads * dk
	if k.Shape()[1] != dkv || !v.Shape().Equal(k.Shape()) || !dO.Shape().Equal(q.Shape()) {
		return nil, fmt.Errorf("vulkan: mha-backward shape mismatch")
	}
	if p.ALiBi || dk > 128 || seq == 0 || C.vk_atomic_float() != 1 {
		return refFallback()
	}

	scale := p.Scale / math.Sqrt(float64(dk))
	causal := 0
	if p.Causal {
		causal = 1
	}
	qc, kc, vc, dc := q.Contiguous(), k.Contiguous(), v.Contiguous(), dO.Contiguous()
	dQ := tensor.New(tensor.F32, q.Shape()) // zero-initialized (atomic accumulators)
	dK := tensor.New(tensor.F32, k.Shape())
	dV := tensor.New(tensor.F32, v.Shape())
	if p.Window == 0 {
		// §T528 matmul-decomposed chain (the vulkan analog of metal's §T399 MPS
		// path): staged strided matmuls + packed softmax + jacobian, one submit
		// per stage. dVt/dKt come back per QUERY head; GQA groups reduce here.
		dVt := make([]float32, heads*seq*dk)
		dKt := make([]float32, heads*seq*dk)
		rc := C.vk_mha_backward_chain(
			(*C.uint32_t)(unsafe.Pointer(&matmulStridedSpirv[0])), C.int(len(matmulStridedSpirv)),
			(*C.uint32_t)(unsafe.Pointer(&softmaxPackedSpirv[0])), C.int(len(softmaxPackedSpirv)),
			(*C.uint32_t)(unsafe.Pointer(&smJacobianSpirv[0])), C.int(len(smJacobianSpirv)),
			(*C.float)(&qc.Storage().F32()[0]),
			(*C.float)(&kc.Storage().F32()[0]),
			(*C.float)(&vc.Storage().F32()[0]),
			(*C.float)(&dc.Storage().F32()[0]),
			(*C.float)(&dQ.Storage().F32()[0]),
			(*C.float)(&dVt[0]), (*C.float)(&dKt[0]),
			C.int(seq), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk),
			C.int(causal), C.float(scale),
		)
		if rc == 0 {
			rep := heads / kvHeads
			dks, dvs := dK.Storage().F32(), dV.Storage().F32()
			for h := 0; h < heads; h++ {
				hh := h / rep
				for i := 0; i < seq; i++ {
					src := (h*seq + i) * dk
					dst := i*dkv + hh*dk
					for d := 0; d < dk; d++ {
						dks[dst+d] += dKt[src+d]
						dvs[dst+d] += dVt[src+d]
					}
				}
			}
			return []*tensor.Tensor{dQ, dK, dV}, nil
		}
		// chain failed: fall through to the atomic kernel below
	}
	rc := C.vk_mha_backward_f32(
		(*C.uint32_t)(unsafe.Pointer(&mhaBwdSpirv[0])), C.int(len(mhaBwdSpirv)),
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
	if int(rc) == -7 {
		return refFallback() // device lacks float atomics
	}
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: mha-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{dQ, dK, dV}, nil
}

// mhaF32 is fused multi-head attention on the GPU via Vulkan compute (§T89, the
// portable twin of the Metal kernel §T85). It serves the common case — heads, GQA
// kv_heads, causal, sliding-window attention (§T115), YaRN attn-scale — and falls back
// to the reference (§I4) for ALiBi, per-head dim dk>128, or degenerate shapes. Forward
// only. Training/prefill shapes (sq==sk, no window) route to the single-pass
// FlashAttention kernel — same math, ~1.6× faster (§T340).
func mhaF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("vulkan: mha wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("vulkan: mha needs rank-2 [seq,dmodel]")
	}
	if q.Dtype() != tensor.F32 || k.Dtype() != tensor.F32 || v.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: mha is f32-only")
	}
	p, _ := attrs.(backend.AttnAttrs)
	p = p.WithDefaults()

	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	heads := p.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("vulkan: mha dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := p.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("vulkan: mha heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	dkv := kvHeads * dk
	if k.Shape()[1] != dkv || !v.Shape().Equal(k.Shape()) {
		return nil, fmt.Errorf("vulkan: mha K/V must be [sk,%d], got %v/%v", dkv, k.Shape(), v.Shape())
	}
	if sq > sk {
		return nil, fmt.Errorf("vulkan: mha query len %d exceeds key len %d", sq, sk)
	}
	// Unsupported feature or shape → reference backend, same result (§I4).
	if p.ALiBi || dk > 128 || sq == 0 || sk == 0 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMHA, in, attrs)
	}
	if len(mhaSpirv) == 0 {
		return nil, fmt.Errorf("vulkan: mha.spv not embedded — run `make vulkan-spv` (glslc)")
	}

	scale := p.Scale / math.Sqrt(float64(dk))
	causal := 0
	if p.Causal {
		causal = 1
	}
	qc, kc, vc := q.Contiguous(), k.Contiguous(), v.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{sq, dm})
	if sq == sk && p.Window == 0 && len(flashAttnSpirv) > 0 {
		// §T531: the matmul-decomposed forward chain (§T528 structure — 3 submits,
		// one dset per stage) is the DEFAULT for the training/prefill shape: A/B/A
		// on the real training step measured flash 1590 → chain 1878 tok/s (+18%;
		// with the §T528 backward chain 2.01× cumulative, metal-class). §T398's
		// earlier rejection measured a costlier per-head-submit structure — its
		// verdict is superseded for THIS shape; flash remains the sliding-window
		// path and the fallback when the chain errors.
		{
			rc := C.vk_mha_forward_chain(
				(*C.uint32_t)(unsafe.Pointer(&matmulStridedSpirv[0])), C.int(len(matmulStridedSpirv)),
				(*C.uint32_t)(unsafe.Pointer(&softmaxPackedSpirv[0])), C.int(len(softmaxPackedSpirv)),
				(*C.float)(&qc.Storage().F32()[0]),
				(*C.float)(&kc.Storage().F32()[0]),
				(*C.float)(&vc.Storage().F32()[0]),
				(*C.float)(&out.Storage().F32()[0]),
				C.int(sq), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk),
				C.int(causal), C.float(scale),
			)
			if rc == 0 {
				return []*tensor.Tensor{out}, nil
			}
		}
		// Training/prefill fast path (§T340): the single-pass FlashAttention kernel. The §T397/T398
		// strided-matmul reformulation (mhaMatmul, MPS-analog) wins in ISOLATION even at the forward's
		// seq (256×8×64: flash 5.12→matmul 0.85ms = 6×, buffers cached §T398) BUT does NOT help the
		// real GPT forward here (§V22 A/B: 3191→2957 tok/s) — UNLIKE metal (§T395 1.87×), on Vulkan
		// attention is NOT the forward bottleneck: the tiled-GEMM FFN matmuls (~300-600 GFLOP/s vs
		// metal MPS ~1186) dominate the wall time, so a faster attention saves little and the per-head
		// multi-dispatch/up-download overhead eats it. Flash stays wired; mhaMatmul + cache + shaders
		// kept + tested (ready for large-seq or if the FFN matmuls are sped up first). §T398.
		rc := C.vk_flashattn_f32(
			(*C.uint32_t)(unsafe.Pointer(&flashAttnSpirv[0])), C.int(len(flashAttnSpirv)),
			(*C.float)(&qc.Storage().F32()[0]),
			(*C.float)(&kc.Storage().F32()[0]),
			(*C.float)(&vc.Storage().F32()[0]),
			(*C.float)(&out.Storage().F32()[0]),
			C.int(sq), C.int(dm), C.int(heads), C.int(dk), C.int(causal), C.int(kvHeads), C.float(scale),
		)
		if rc != 0 {
			return nil, fmt.Errorf("vulkan: mha (flash path) failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{out}, nil
	}
	// Cooperative decode/prefill path (§T432, the per-op twin of the recorder's §T429/431 route).
	if p.Window == 0 && dk <= 128 && (causal != 0 || sq == 1) && len(mhaDecodeSpirv) > 0 {
		rc := C.vk_mha_decode_f32(
			(*C.uint32_t)(unsafe.Pointer(&mhaDecodeSpirv[0])), C.int(len(mhaDecodeSpirv)),
			(*C.float)(&qc.Storage().F32()[0]),
			(*C.float)(&kc.Storage().F32()[0]),
			(*C.float)(&vc.Storage().F32()[0]),
			(*C.float)(&out.Storage().F32()[0]),
			C.int(sq), C.int(sk), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk),
			C.int(causal), C.float(scale),
		)
		if rc != 0 {
			return nil, fmt.Errorf("vulkan: mha (decode path) failed (code %d)", int(rc))
		}
		return []*tensor.Tensor{out}, nil
	}
	rc := C.vk_mha_f32(
		(*C.uint32_t)(unsafe.Pointer(&mhaSpirv[0])), C.int(len(mhaSpirv)),
		(*C.float)(&qc.Storage().F32()[0]),
		(*C.float)(&kc.Storage().F32()[0]),
		(*C.float)(&vc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(sq), C.int(sk), C.int(dm), C.int(heads), C.int(kvHeads), C.int(dk),
		C.int(causal), C.int(p.Window), C.float(scale),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: mha failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// flashAttnF32 is FlashAttention-2 forward on the GPU via Vulkan compute (§T110, the
// portable twin of the Metal kernel §T109; the GPU backend of the reference §T71 /
// §R72). One invocation per (head, query row) streams the keys with the online-softmax
// recurrence, so — unlike mhaF32's two-pass softmax — no [seq,seq] score row is
// materialized (the flash memory win). It serves heads, GQA/MQA kv_heads and causal for
// Q [seq,dmodel], K,V [seq,kv_heads·dk] (sq==sk), and falls back to the reference (§I4)
// for a per-head dim dk>128 or degenerate shapes. Exact: equals mhaF32 up to float
// reassociation (verified by V-CROSS, §V3/§V11); block-invariant, so it ignores the
// Block hint.
func flashAttnF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("vulkan: flashattn wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("vulkan: flashattn needs rank-2 [seq,dmodel]")
	}
	if q.Dtype() != tensor.F32 || k.Dtype() != tensor.F32 || v.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: flashattn is f32-only")
	}
	seq, dm := q.Shape()[0], q.Shape()[1]
	p, _ := attrs.(backend.AttnAttrs)
	p = p.WithDefaults()
	heads := p.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("vulkan: flashattn dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := p.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("vulkan: flashattn heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	dkv := kvHeads * dk
	// Q [seq,dm]; K,V [seq,kvHeads*dk] (GQA); FlashAttention requires sq==sk.
	if k.Shape()[0] != seq || !v.Shape().Equal(k.Shape()) || k.Shape()[1] != dkv {
		return nil, fmt.Errorf("vulkan: flashattn needs Q [seq,%d], K,V [seq,%d] (sq==sk), got %v/%v/%v", dm, dkv, q.Shape(), k.Shape(), v.Shape())
	}
	// Unsupported dim or degenerate shape → reference backend, same result (§I4).
	if dk > 128 || seq == 0 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpFlashAttn, in, attrs)
	}
	if len(flashAttnSpirv) == 0 {
		return nil, fmt.Errorf("vulkan: flashattn.spv not embedded — run `make vulkan-spv` (glslc)")
	}

	scale := 1 / math.Sqrt(float64(dk)) // reference flashattn scale (no YaRN attn_scale)
	causal := 0
	if p.Causal {
		causal = 1
	}
	qc, kc, vc := q.Contiguous(), k.Contiguous(), v.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{seq, dm})
	rc := C.vk_flashattn_f32(
		(*C.uint32_t)(unsafe.Pointer(&flashAttnSpirv[0])), C.int(len(flashAttnSpirv)),
		(*C.float)(&qc.Storage().F32()[0]),
		(*C.float)(&kc.Storage().F32()[0]),
		(*C.float)(&vc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(seq), C.int(dm), C.int(heads), C.int(dk), C.int(causal), C.int(kvHeads), C.float(scale),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: flashattn failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// retentionF32 runs the RetNet retention forward (§T173, the Vulkan twin of Metal §T172) on the GPU:
// O[L,d] = (QKᵀ ⊙ D)·V with the γ-decay mask D_nm=γ^(n−m). Q,K,V are [L,d] (a single head). One
// invocation per query row streams keys with a register acc[d]; falls back to the reference (§I4) for
// d>128, empty L, or non-f32.
func retentionF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("vulkan: retention wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("vulkan: retention needs rank-2 [L,d]")
	}
	l, dk := q.Shape()[0], q.Shape()[1] // query/key dim
	dv := v.Shape()[1]                  // value dim (may differ — RetNet value expansion, §T179)
	if k.Shape()[0] != l || v.Shape()[0] != l || k.Shape()[1] != dk {
		return nil, fmt.Errorf("vulkan: retention needs Q,K [L,dk], V [L,dv]; got Q%v K%v V%v", q.Shape(), k.Shape(), v.Shape())
	}
	pa, _ := attrs.(backend.RetentionAttrs)
	// Unsupported dim / degenerate / non-f32 → reference (§I4). The kernel's register acc is [dv]
	// (must be ≤128); dk is unbounded (streamed). d_v≠d_k is handled natively (§T179).
	if dv > 128 || l == 0 || q.Dtype() != tensor.F32 || k.Dtype() != tensor.F32 || v.Dtype() != tensor.F32 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpRetention, in, attrs)
	}
	if len(retentionSpirv) == 0 {
		return nil, fmt.Errorf("vulkan: retention.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	qc, kc, vc := q.Contiguous(), k.Contiguous(), v.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{l, dv})
	rc := C.vk_retention_f32(
		(*C.uint32_t)(unsafe.Pointer(&retentionSpirv[0])), C.int(len(retentionSpirv)),
		(*C.float)(&qc.Storage().F32()[0]),
		(*C.float)(&kc.Storage().F32()[0]),
		(*C.float)(&vc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(l), C.int(dk), C.int(dv), C.float(pa.Gamma),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: retention failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// retentionBackwardF32 runs the RetNet retention backward (§T175, the Vulkan twin of Metal §T174) on
// the GPU: (Q,K,V,dO) [L,d] → (dQ,dK,dV). One invocation per query row writes dQ directly and float-
// atomically accumulates dK/dV — so a Vulkan tape trains retention on-device. Falls back to the
// reference (§I4) for d>128/empty/non-f32, or when the device lacks VK_EXT_shader_atomic_float.
func retentionBackwardF32(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("vulkan: retention-backward wants (Q,K,V,dO), got %d inputs", len(in))
	}
	q, k, v, dO := in[0], in[1], in[2], in[3]
	for _, t := range in {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("vulkan: retention-backward needs rank-2 tensors")
		}
	}
	l, dk := q.Shape()[0], q.Shape()[1] // query/key dim
	dv := v.Shape()[1]                  // value dim (may differ — RetNet value expansion, §T192)
	if k.Shape()[0] != l || k.Shape()[1] != dk || v.Shape()[0] != l || dO.Shape()[0] != l || dO.Shape()[1] != dv {
		return nil, fmt.Errorf("vulkan: retention-backward needs Q,K [L,dk], V,dO [L,dv]; got Q%v K%v V%v dO%v", q.Shape(), k.Shape(), v.Shape(), dO.Shape())
	}
	// Unsupported dim / non-f32 / no atomic-float → reference (§I4). dQ register acc is [dk] (≤128);
	// dv is streamed (unbounded). d_v≠d_k is native (§T192).
	if dk > 128 || l == 0 || q.Dtype() != tensor.F32 || k.Dtype() != tensor.F32 || v.Dtype() != tensor.F32 || dO.Dtype() != tensor.F32 || C.vk_atomic_float() != 1 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpRetentionBackward, in, attrs)
	}
	if len(retentionBwdSpirv) == 0 {
		return nil, fmt.Errorf("vulkan: retention_bwd.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	pa, _ := attrs.(backend.RetentionAttrs)
	qc, kc, vc, dc := q.Contiguous(), k.Contiguous(), v.Contiguous(), dO.Contiguous()
	gq := tensor.New(tensor.F32, tensor.Shape{l, dk}) // zero-initialized (atomic accumulators)
	gk := tensor.New(tensor.F32, tensor.Shape{l, dk})
	gv := tensor.New(tensor.F32, tensor.Shape{l, dv})
	rc := C.vk_retention_backward_f32(
		(*C.uint32_t)(unsafe.Pointer(&retentionBwdSpirv[0])), C.int(len(retentionBwdSpirv)),
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
		return nil, fmt.Errorf("vulkan: retention-backward failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{gq, gk, gv}, nil
}

func matmulF32(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("vulkan: matmul wants 2 inputs, got %d", len(in))
	}
	a, b := in[0], in[1]
	if a.Ndim() != 2 || b.Ndim() != 2 {
		return nil, fmt.Errorf("vulkan: matmul needs rank-2, got %dD and %dD", a.Ndim(), b.Ndim())
	}
	if a.Dtype() != tensor.F32 || b.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("vulkan: matmul is f32-only, got %v/%v", a.Dtype(), b.Dtype())
	}
	m, k := a.Shape()[0], a.Shape()[1]
	k2, n := b.Shape()[0], b.Shape()[1]
	if k != k2 {
		return nil, fmt.Errorf("vulkan: inner dim mismatch %v · %v", a.Shape(), b.Shape())
	}
	if m == 0 || n == 0 || k == 0 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMatMul, in, nil)
	}
	if len(spirv) == 0 {
		return nil, fmt.Errorf("vulkan: matmul.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	// A transposed VIEW of a contiguous matrix (the matmul backward's dO·Wᵀ / Xᵀ·dO) is
	// read transposed in the shader via a flag, instead of a CPU strided-gather copy (§T356).
	ab, transA := transposedBase(a)
	bb, transB := transposedBase(b)
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	rc := C.vk_matmul_f32(
		(*C.uint32_t)(unsafe.Pointer(&spirv[0])), C.int(len(spirv)),
		(*C.float)(&ab.Storage().F32()[0]),
		(*C.float)(&bb.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
		boolToCInt(transA), boolToCInt(transB),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: matmul failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// transposedBase reports whether x is exactly a 2-D transposed view of a contiguous,
// offset-0 matrix. If so it returns that contiguous base (read transposed in-shader via a
// flag — no copy); otherwise it returns x.Contiguous() and false.
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

func init() {
	if Available() {
		backend.Register(Backend{})
	}
	// no Vulkan device → not registered → feature detection via backend.Available() (§I4/§V4)
}
