//go:build cuda && cgo && (linux || windows)

package cuda

/*
#cgo LDFLAGS: -lcudart -lcublas -lnvrtc -lcuda
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// device is the CUDA tensor.Device. Tensor memory stays host-side; the kernel
// copies through device buffers per call — honest about transfer cost.
// Device-resident tensors are a later optimization (§V14).
type device struct{}

func (device) Kind() tensor.DeviceKind     { return tensor.KindCUDA }
func (d device) String() string            { return d.Kind().String() }
func (device) Allocator() tensor.Allocator { return tensor.Heap() }

// Backend implements backend.Backend over CUDA/cuBLAS. Synchronous: every kernel
// copies D2H and synchronizes before returning, so Synchronize is a no-op (§V14
// permits async later without an API break).
type Backend struct{}

func (Backend) Name() backend.Name    { return backend.CUDA }
func (Backend) Device() tensor.Device { return device{} }
func (Backend) Synchronize() error    { return nil }

// Available reports whether a CUDA-capable GPU is present.
func Available() bool { return C.cu_available() == 1 }

func (Backend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpMatMul && dtype == tensor.F32 {
		return matmulF32, true
	}
	return nil, false // everything else: fallback to Pure-Go (§I4)
}

func matmulF32(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cuda: matmul wants 2 inputs, got %d", len(in))
	}
	a, b := in[0], in[1]
	if a.Ndim() != 2 || b.Ndim() != 2 {
		return nil, fmt.Errorf("cuda: matmul needs rank-2, got %dD and %dD", a.Ndim(), b.Ndim())
	}
	if a.Dtype() != tensor.F32 || b.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("cuda: matmul is f32-only, got %v/%v", a.Dtype(), b.Dtype())
	}
	m, k := a.Shape()[0], a.Shape()[1]
	k2, n := b.Shape()[0], b.Shape()[1]
	if k != k2 {
		return nil, fmt.Errorf("cuda: inner dim mismatch %v · %v", a.Shape(), b.Shape())
	}
	if m == 0 || n == 0 || k == 0 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMatMul, in, nil)
	}
	ac, bc := a.Contiguous(), b.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	rc := C.cu_matmul_f32(
		(*C.float)(&ac.Storage().F32()[0]),
		(*C.float)(&bc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda: matmul failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

// ResidentB is a weight matrix B[K,N] uploaded to the GPU once and reused across
// many matmuls, skipping its per-call H2D transfer (§V14 Phase-1, mirrors the
// metal §T156 resident-weight seed). It is the transfer lever for inference: the
// weight is fixed, only the activation A[M,K] varies. The result C = A·B is
// identical to the per-call cuda matmul (cuBLAS Sgemm, same operands). Call Free
// when done; a ResidentB outliving process exit is reclaimed by the driver.
type ResidentB struct {
	ptr  unsafe.Pointer // device buffer holding B[K,N] row-major f32
	k, n int
}

// NewResidentB uploads a rank-2 f32 B[K,N] to the GPU. Free the result when done.
func NewResidentB(b *tensor.Tensor) (*ResidentB, error) {
	if b.Ndim() != 2 {
		return nil, fmt.Errorf("cuda: ResidentB needs rank-2, got %dD", b.Ndim())
	}
	if b.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("cuda: ResidentB is f32-only, got %v", b.Dtype())
	}
	bc := b.Contiguous()
	k, n := bc.Shape()[0], bc.Shape()[1]
	s := bc.Storage().F32()
	ptr := C.cu_upload_f32((*C.float)(&s[0]), C.int(len(s)))
	if ptr == nil {
		return nil, fmt.Errorf("cuda: ResidentB upload failed (%d×%d)", k, n)
	}
	return &ResidentB{ptr: ptr, k: k, n: n}, nil
}

// MatMul computes A[M,K]·B[K,N] using the resident B (no B upload). A must be a
// rank-2 f32 with inner dim K.
func (r *ResidentB) MatMul(a *tensor.Tensor) (*tensor.Tensor, error) {
	if r.ptr == nil {
		return nil, fmt.Errorf("cuda: ResidentB already freed")
	}
	if a.Ndim() != 2 || a.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("cuda: ResidentB.MatMul needs rank-2 f32 A")
	}
	m, k := a.Shape()[0], a.Shape()[1]
	if k != r.k {
		return nil, fmt.Errorf("cuda: ResidentB inner dim mismatch A[%d,%d]·B[%d,%d]", m, k, r.k, r.n)
	}
	ac := a.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{m, r.n})
	rc := C.cu_matmul_f32_bres(
		(*C.float)(&ac.Storage().F32()[0]),
		r.ptr,
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(r.n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda: resident matmul failed (code %d)", int(rc))
	}
	return out, nil
}

// Embed treats this resident matrix as an embedding table [vocab, d] and gathers
// the rows for the given token ids into a resident [len(ids), d] activation, on
// the GPU — the input embedding of a language model. Free the result when done.
func (r *ResidentB) Embed(ids []int32) (*DeviceF32, error) {
	if r.ptr == nil {
		return nil, fmt.Errorf("cuda: Embed on a freed table")
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("cuda: Embed needs ≥1 id")
	}
	dIds := C.cu_upload_i32((*C.int)(&ids[0]), C.int(len(ids)))
	if dIds == nil {
		return nil, fmt.Errorf("cuda: Embed id upload failed")
	}
	defer C.cu_free_f32(dIds)
	out := C.cu_alloc_f32(C.int(len(ids) * r.n))
	if out == nil {
		return nil, fmt.Errorf("cuda: Embed output alloc failed")
	}
	if rc := C.cu_embed_f32(r.ptr, dIds, out, C.int(len(ids)), C.int(r.n)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: embed failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: len(ids), cols: r.n}, nil
}

// Free releases the device buffer. Safe to call more than once.
func (r *ResidentB) Free() {
	if r.ptr != nil {
		C.cu_free_f32(r.ptr)
		r.ptr = nil
	}
}

// ResidentVec is a 1-D f32 weight resident on the GPU (e.g. an RMSNorm gamma),
// uploaded once and reused. Free when done.
type ResidentVec struct {
	ptr unsafe.Pointer
	n   int
}

// NewResidentVec uploads a 1-D f32 tensor to the GPU.
func NewResidentVec(v *tensor.Tensor) (*ResidentVec, error) {
	if v.Ndim() != 1 || v.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("cuda: ResidentVec needs rank-1 f32, got %dD %v", v.Ndim(), v.Dtype())
	}
	vc := v.Contiguous()
	s := vc.Storage().F32()
	ptr := C.cu_upload_f32((*C.float)(&s[0]), C.int(len(s)))
	if ptr == nil {
		return nil, fmt.Errorf("cuda: ResidentVec upload failed (%d)", len(s))
	}
	return &ResidentVec{ptr: ptr, n: vc.Shape()[0]}, nil
}

// Free releases the device buffer. Safe to call more than once.
func (r *ResidentVec) Free() {
	if r.ptr != nil {
		C.cu_free_f32(r.ptr)
		r.ptr = nil
	}
}

// DeviceF32 is a rank-2 f32 activation resident on the GPU (§V14 Phase-2). It
// lets a chain of matmuls keep its intermediates on-device: only the first
// UploadF32 and the final ToHost touch host memory, so an MLP x·W1·W2 pays one
// H2D + one D2H instead of a round-trip per matmul. Combine with ResidentB
// (resident weights) for a fully-on-GPU linear stack. Free when done.
type DeviceF32 struct {
	ptr        unsafe.Pointer
	rows, cols int
}

// UploadF32 copies a rank-2 f32 tensor to the GPU as the start of an on-device
// chain. Free the result (directly or via the last chain output) when done.
func UploadF32(a *tensor.Tensor) (*DeviceF32, error) {
	if a.Ndim() != 2 || a.Dtype() != tensor.F32 {
		return nil, fmt.Errorf("cuda: UploadF32 needs rank-2 f32, got %dD %v", a.Ndim(), a.Dtype())
	}
	ac := a.Contiguous()
	s := ac.Storage().F32()
	ptr := C.cu_upload_f32((*C.float)(&s[0]), C.int(len(s)))
	if ptr == nil {
		return nil, fmt.Errorf("cuda: UploadF32 failed (%v)", a.Shape())
	}
	return &DeviceF32{ptr: ptr, rows: ac.Shape()[0], cols: ac.Shape()[1]}, nil
}

// MatMulDevice computes a·B with the activation a already resident and the weight
// B resident, leaving the result on the GPU — no host round-trip. Chain it:
// h := W1.MatMulDevice(dx); y := W2.MatMulDevice(h). Free intermediates you no
// longer need.
func (r *ResidentB) MatMulDevice(a *DeviceF32) (*DeviceF32, error) {
	if r.ptr == nil || a.ptr == nil {
		return nil, fmt.Errorf("cuda: MatMulDevice on a freed handle")
	}
	if a.cols != r.k {
		return nil, fmt.Errorf("cuda: MatMulDevice inner dim mismatch a[%d,%d]·B[%d,%d]", a.rows, a.cols, r.k, r.n)
	}
	out := C.cu_alloc_f32(C.int(a.rows * r.n))
	if out == nil {
		return nil, fmt.Errorf("cuda: MatMulDevice output alloc failed")
	}
	rc := C.cu_matmul_f32_ddd(a.ptr, r.ptr, out, C.int(a.rows), C.int(r.k), C.int(r.n))
	if rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: device matmul failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: a.rows, cols: r.n}, nil
}

// MatMulAccInto computes c += a·B in place (cuBLAS beta=1), fusing a transformer
// residual add into the projection matmul: c must already hold the residual and
// have shape [a.rows, B.n]. Saves a separate Add kernel launch and an output
// allocation per residual on the decode hot path (§PERF: decode is launch-bound).
func (r *ResidentB) MatMulAccInto(a, c *DeviceF32) error {
	if r.ptr == nil || a.ptr == nil || c.ptr == nil {
		return fmt.Errorf("cuda: MatMulAccInto on a freed handle")
	}
	if a.cols != r.k {
		return fmt.Errorf("cuda: MatMulAccInto inner dim mismatch a[%d,%d]·B[%d,%d]", a.rows, a.cols, r.k, r.n)
	}
	if c.rows != a.rows || c.cols != r.n {
		return fmt.Errorf("cuda: MatMulAccInto accumulator must be [%d,%d], got [%d,%d]", a.rows, r.n, c.rows, c.cols)
	}
	if rc := C.cu_matmul_f32_ddd_acc(a.ptr, r.ptr, c.ptr, C.int(a.rows), C.int(r.k), C.int(r.n)); rc != 0 {
		return fmt.Errorf("cuda: device matmul-acc failed (code %d)", int(rc))
	}
	return nil
}

// ToHost downloads the resident activation to a new host tensor.
func (d *DeviceF32) ToHost() (*tensor.Tensor, error) {
	if d.ptr == nil {
		return nil, fmt.Errorf("cuda: ToHost on a freed handle")
	}
	out := tensor.New(tensor.F32, tensor.Shape{d.rows, d.cols})
	rc := C.cu_download_f32(d.ptr, (*C.float)(&out.Storage().F32()[0]), C.int(d.rows*d.cols))
	if rc != 0 {
		return nil, fmt.Errorf("cuda: download failed (code %d)", int(rc))
	}
	return out, nil
}

// GELU applies the exact Gaussian Error Linear Unit (0.5·x·(1+erf(x/√2)))
// in-place on the resident activation, on the GPU. The kernel is compiled once
// from CUDA-C via nvrtc (no nvcc) and launched on the matmul stream, so a
// transformer-style matmul→GELU→matmul block stays fully device-resident with no
// host round-trip for the activation. Result matches the Pure-Go GELU within the
// f32 tolerance.
func (d *DeviceF32) GELU() error {
	if d.ptr == nil {
		return fmt.Errorf("cuda: GELU on a freed handle")
	}
	if rc := C.cu_gelu_f32(d.ptr, C.int(d.rows*d.cols)); rc != 0 {
		return fmt.Errorf("cuda: gelu failed (code %d)", int(rc))
	}
	return nil
}

// SiLU applies the SiLU / swish activation (x·sigmoid(x)) in-place on the GPU
// (SwiGLU feed-forward). Same on-device, no-round-trip contract as GELU; matches
// the Pure-Go SiLU within the f32 tolerance.
func (d *DeviceF32) SiLU() error {
	if d.ptr == nil {
		return fmt.Errorf("cuda: SiLU on a freed handle")
	}
	if rc := C.cu_silu_f32(d.ptr, C.int(d.rows*d.cols)); rc != 0 {
		return fmt.Errorf("cuda: silu failed (code %d)", int(rc))
	}
	return nil
}

// Add adds another resident activation of the same shape into this one in-place
// (d += other), on the GPU — the residual connection of a transformer block,
// kept device-resident. other is unchanged.
func (d *DeviceF32) Add(other *DeviceF32) error {
	if d.ptr == nil || other.ptr == nil {
		return fmt.Errorf("cuda: Add on a freed handle")
	}
	if d.rows != other.rows || d.cols != other.cols {
		return fmt.Errorf("cuda: Add shape mismatch [%d,%d] += [%d,%d]", d.rows, d.cols, other.rows, other.cols)
	}
	if rc := C.cu_add_f32(d.ptr, other.ptr, C.int(d.rows*d.cols)); rc != 0 {
		return fmt.Errorf("cuda: add failed (code %d)", int(rc))
	}
	return nil
}

// Mul multiplies another resident activation of the same shape into this one
// in-place (d *= other), on the GPU — e.g. the SwiGLU gate·up product. other is
// unchanged.
func (d *DeviceF32) Mul(other *DeviceF32) error {
	if d.ptr == nil || other.ptr == nil {
		return fmt.Errorf("cuda: Mul on a freed handle")
	}
	if d.rows != other.rows || d.cols != other.cols {
		return fmt.Errorf("cuda: Mul shape mismatch [%d,%d] *= [%d,%d]", d.rows, d.cols, other.rows, other.cols)
	}
	if rc := C.cu_mul_f32(d.ptr, other.ptr, C.int(d.rows*d.cols)); rc != 0 {
		return fmt.Errorf("cuda: mul failed (code %d)", int(rc))
	}
	return nil
}

// SwiGLU fuses the gated-FFN nonlinearity in-place: d = SiLU(d) ⊙ up (this
// activation is the gate). One kernel/one memory pass instead of SiLU then Mul.
// up must have the same shape and is unchanged. Matches SiLU-then-Mul to f32.
func (d *DeviceF32) SwiGLU(up *DeviceF32) error {
	if d.ptr == nil || up.ptr == nil {
		return fmt.Errorf("cuda: SwiGLU on a freed handle")
	}
	if d.rows != up.rows || d.cols != up.cols {
		return fmt.Errorf("cuda: SwiGLU shape mismatch [%d,%d] vs [%d,%d]", d.rows, d.cols, up.rows, up.cols)
	}
	if rc := C.cu_swiglu_f32(d.ptr, up.ptr, C.int(d.rows*d.cols)); rc != 0 {
		return fmt.Errorf("cuda: swiglu failed (code %d)", int(rc))
	}
	return nil
}

// RMSNorm applies RMSNorm (y = x/√(mean(x²)+eps)·gamma) in-place, row-wise over
// the last axis, on the GPU — the pre-block norm of a Llama-style transformer,
// kept device-resident. gamma is a resident [cols] weight; eps ≤ 0 uses 1e-5
// (the reference default). Matches the Pure-Go RMSNorm within the f32 tolerance
// (sum-of-squares accumulated in double, mirroring §V10).
func (d *DeviceF32) RMSNorm(gamma *ResidentVec, eps float32) error {
	if d.ptr == nil || gamma.ptr == nil {
		return fmt.Errorf("cuda: RMSNorm on a freed handle")
	}
	if gamma.n != d.cols {
		return fmt.Errorf("cuda: RMSNorm gamma [%d] != cols %d", gamma.n, d.cols)
	}
	if eps <= 0 {
		eps = 1e-5
	}
	if rc := C.cu_rmsnorm_f32(d.ptr, d.ptr, gamma.ptr, C.int(d.rows), C.int(d.cols), C.float(eps)); rc != 0 {
		return fmt.Errorf("cuda: rmsnorm failed (code %d)", int(rc))
	}
	return nil
}

// RMSNormTo returns a NEW resident buffer holding RMSNorm(d)·gamma, leaving d
// unchanged — the out-of-place form used on the transformer residual path so the
// residual need not be cloned before normalizing (one fewer launch + alloc per
// norm than Clone+RMSNorm). Free the result when done.
func (d *DeviceF32) RMSNormTo(gamma *ResidentVec, eps float32) (*DeviceF32, error) {
	if d.ptr == nil || gamma.ptr == nil {
		return nil, fmt.Errorf("cuda: RMSNormTo on a freed handle")
	}
	if gamma.n != d.cols {
		return nil, fmt.Errorf("cuda: RMSNormTo gamma [%d] != cols %d", gamma.n, d.cols)
	}
	if eps <= 0 {
		eps = 1e-5
	}
	out := C.cu_alloc_f32(C.int(d.rows * d.cols))
	if out == nil {
		return nil, fmt.Errorf("cuda: RMSNormTo alloc failed")
	}
	if rc := C.cu_rmsnorm_f32(d.ptr, out, gamma.ptr, C.int(d.rows), C.int(d.cols), C.float(eps)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: rmsnorm-to failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: d.rows, cols: d.cols}, nil
}

// Softmax applies a numerically-stable softmax (exp(x−max)/Σexp) in-place along
// the last axis (each row), on the GPU — the attention-weight normalization,
// kept device-resident. Matches the Pure-Go softmax within the f32 tolerance.
func (d *DeviceF32) Softmax() error {
	if d.ptr == nil {
		return fmt.Errorf("cuda: Softmax on a freed handle")
	}
	if rc := C.cu_softmax_f32(d.ptr, C.int(d.rows), C.int(d.cols)); rc != 0 {
		return fmt.Errorf("cuda: softmax failed (code %d)", int(rc))
	}
	return nil
}

// CausalScaleMask scales these attention scores[qRows, kCols] by scale and masks
// out future keys (score(i,j) → −inf when j > i + offset) in-place, on the GPU —
// the pre-softmax step of causal attention. offset = kCols − qRows aligns a short
// query window to the tail of the key sequence (0 for a full square prefill).
func (d *DeviceF32) CausalScaleMask(scale float32, offset int) error {
	if d.ptr == nil {
		return fmt.Errorf("cuda: CausalScaleMask on a freed handle")
	}
	if rc := C.cu_causal_scale_f32(d.ptr, C.int(d.rows), C.int(d.cols), C.float(scale), C.int(offset)); rc != 0 {
		return fmt.Errorf("cuda: causal-scale failed (code %d)", int(rc))
	}
	return nil
}

// RoPE applies rotary position embeddings in-place to this activation, treated
// as q[seq, heads·headDim] (rows = seq, cols = heads·headDim), on the GPU — the
// query/key rotation of a Llama-style attention, kept device-resident. The
// rotary frequencies (incl. PI/YaRN scaling) come from backend.RoPEFreqs, so the
// result matches the Pure-Go RoPE within the f32 tolerance. XPos is not yet
// supported on-device.
func (d *DeviceF32) RoPE(attrs backend.RoPEAttrs) error {
	if d.ptr == nil {
		return fmt.Errorf("cuda: RoPE on a freed handle")
	}
	if attrs.XPos {
		return fmt.Errorf("cuda: RoPE XPos not supported on device yet")
	}
	heads := attrs.Heads
	if heads <= 0 {
		heads = 1
	}
	if d.cols%heads != 0 {
		return fmt.Errorf("cuda: RoPE width %d not divisible by heads %d", d.cols, heads)
	}
	hd := d.cols / heads
	if hd%2 != 0 {
		return fmt.Errorf("cuda: RoPE head dim %d must be even", hd)
	}
	inv, posDiv := backend.RoPEFreqs(hd, attrs)
	inv32 := make([]float32, len(inv))
	for i, v := range inv {
		inv32[i] = float32(v)
	}
	invPtr := C.cu_upload_f32((*C.float)(&inv32[0]), C.int(len(inv32)))
	if invPtr == nil {
		return fmt.Errorf("cuda: RoPE frequency upload failed")
	}
	defer C.cu_free_f32(invPtr)
	if rc := C.cu_rope_f32(d.ptr, invPtr, C.int(d.rows), C.int(heads), C.int(hd), C.int(attrs.PosOffset), C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: rope failed (code %d)", int(rc))
	}
	return nil
}

// MatMul computes d·b for two resident activations (d[M,K]·b[K,N] → [M,N]),
// leaving the result on the GPU — e.g. attention scores·V. Both operands are
// device-resident; the result is a new DeviceF32 the caller frees.
func (d *DeviceF32) MatMul(b *DeviceF32) (*DeviceF32, error) {
	if d.ptr == nil || b.ptr == nil {
		return nil, fmt.Errorf("cuda: MatMul on a freed handle")
	}
	if d.cols != b.rows {
		return nil, fmt.Errorf("cuda: MatMul inner dim mismatch [%d,%d]·[%d,%d]", d.rows, d.cols, b.rows, b.cols)
	}
	out := C.cu_alloc_f32(C.int(d.rows * b.cols))
	if out == nil {
		return nil, fmt.Errorf("cuda: MatMul output alloc failed")
	}
	if rc := C.cu_matmul_f32_ddd(d.ptr, b.ptr, out, C.int(d.rows), C.int(d.cols), C.int(b.cols)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: device matmul failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: d.rows, cols: b.cols}, nil
}

// MatMulBT computes d·bᵀ for two resident activations (d[M,K]·b[N,K]ᵀ → [M,N]),
// on the GPU — attention scores = Q·Kᵀ. Requires d.cols == b.cols; the result is
// [d.rows, b.rows]. Both operands device-resident; caller frees the result.
func (d *DeviceF32) MatMulBT(b *DeviceF32) (*DeviceF32, error) {
	if d.ptr == nil || b.ptr == nil {
		return nil, fmt.Errorf("cuda: MatMulBT on a freed handle")
	}
	if d.cols != b.cols {
		return nil, fmt.Errorf("cuda: MatMulBT contraction mismatch [%d,%d]·[%d,%d]ᵀ", d.rows, d.cols, b.rows, b.cols)
	}
	out := C.cu_alloc_f32(C.int(d.rows * b.rows))
	if out == nil {
		return nil, fmt.Errorf("cuda: MatMulBT output alloc failed")
	}
	if rc := C.cu_matmul_f32_ddd_bt(d.ptr, b.ptr, out, C.int(d.rows), C.int(d.cols), C.int(b.rows)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: device matmul-bt failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: d.rows, cols: b.rows}, nil
}

// MultiHeadAttention runs scaled dot-product attention over q, k, v — each a
// resident [seq, heads·headDim] activation with RoPE already applied — entirely
// on the GPU, and returns the resident [seq, heads·headDim] output. Every head's
// QKᵀ and scores·V run as one batched strided cuBLAS Sgemm; the scale (1/√headDim)
// + optional causal mask + per-(head,query) softmax run on-device. Free the
// result when done.
func MultiHeadAttention(q, k, v *DeviceF32, heads int, causal bool) (*DeviceF32, error) {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil {
		return nil, fmt.Errorf("cuda: MultiHeadAttention on a freed handle")
	}
	seq, w := q.rows, q.cols
	if heads <= 0 || w%heads != 0 {
		return nil, fmt.Errorf("cuda: MHA width %d not divisible by heads %d", w, heads)
	}
	if k.rows != seq || k.cols != w || v.rows != seq || v.cols != w {
		return nil, fmt.Errorf("cuda: MHA q/k/v shape mismatch")
	}
	hd := w / heads
	scores := C.cu_alloc_f32(C.int(heads * seq * seq))
	if scores == nil {
		return nil, fmt.Errorf("cuda: MHA scores alloc failed")
	}
	defer C.cu_free_f32(scores)
	if rc := C.cu_mha_scores(q.ptr, k.ptr, scores, C.int(seq), C.int(heads), C.int(hd)); rc != 0 {
		return nil, fmt.Errorf("cuda: MHA scores failed (code %d)", int(rc))
	}
	// offset seq disables masking (no key is ever in the future) → scale only.
	offset := seq
	if causal {
		offset = 0
	}
	scale := C.float(1 / math.Sqrt(float64(hd)))
	if rc := C.cu_causal_scale_mh(scores, C.int(heads), C.int(seq), C.int(seq), scale, C.int(offset)); rc != 0 {
		return nil, fmt.Errorf("cuda: MHA scale/mask failed (code %d)", int(rc))
	}
	if rc := C.cu_softmax_f32(scores, C.int(heads*seq), C.int(seq)); rc != 0 {
		return nil, fmt.Errorf("cuda: MHA softmax failed (code %d)", int(rc))
	}
	out := C.cu_alloc_f32(C.int(seq * w))
	if out == nil {
		return nil, fmt.Errorf("cuda: MHA out alloc failed")
	}
	if rc := C.cu_mha_out(scores, v.ptr, out, C.int(seq), C.int(heads), C.int(hd)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: MHA out failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: seq, cols: w}, nil
}

// GroupedQueryAttention is MultiHeadAttention with fewer key/value heads than
// query heads (Llama-3 / Mistral GQA): q is [seq, qHeads·hd], k and v are
// [seq, kvHeads·hd] with qHeads a multiple of kvHeads, and query head h attends
// to kv head h/(qHeads/kvHeads). RoPE is applied by the caller. Returns the
// resident [seq, qHeads·hd] output; free when done. With kvHeads == qHeads this
// is ordinary multi-head attention (MultiHeadAttention is the faster strided
// path for that case).
func GroupedQueryAttention(q, k, v *DeviceF32, qHeads, kvHeads int, causal bool) (*DeviceF32, error) {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil {
		return nil, fmt.Errorf("cuda: GQA on a freed handle")
	}
	seq, wq := q.rows, q.cols
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || wq%qHeads != 0 {
		return nil, fmt.Errorf("cuda: GQA bad head config q=%d kv=%d width=%d", qHeads, kvHeads, wq)
	}
	hd := wq / qHeads
	wkv := kvHeads * hd
	if k.rows != seq || k.cols != wkv || v.rows != seq || v.cols != wkv {
		return nil, fmt.Errorf("cuda: GQA k/v must be [%d,%d], got k%v v%v", seq, wkv, k.shape(), v.shape())
	}
	// offset seq disables masking (full prefill, seqQ==seqKV); offset 0 = causal.
	offset := seq
	if causal {
		offset = 0
	}
	return gqaCore(q, k, v, qHeads, kvHeads, hd, offset)
}

// GroupedQueryAttentionKV is the KV-cache decode form of GroupedQueryAttention:
// q holds the seqQ NEW query rows, k and v hold the FULL cached context of seqKV
// rows (seqQ ≤ seqKV), and the new rows are the tail of that context. Attention
// is causal — new query row i is at absolute position (seqKV−seqQ)+i and attends
// cached keys 0..that. With seqQ==1 this is a single decode step attending the
// whole cache. Returns the resident [seqQ, qHeads·hd] output; free when done.
func GroupedQueryAttentionKV(q, k, v *DeviceF32, qHeads, kvHeads int) (*DeviceF32, error) {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil {
		return nil, fmt.Errorf("cuda: GQA-KV on a freed handle")
	}
	seqQ, wq := q.rows, q.cols
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || wq%qHeads != 0 {
		return nil, fmt.Errorf("cuda: GQA-KV bad head config q=%d kv=%d width=%d", qHeads, kvHeads, wq)
	}
	hd := wq / qHeads
	seqKV, wkv := k.rows, kvHeads*hd
	if seqQ > seqKV {
		return nil, fmt.Errorf("cuda: GQA-KV seqQ %d > seqKV %d", seqQ, seqKV)
	}
	if k.cols != wkv || v.rows != seqKV || v.cols != wkv {
		return nil, fmt.Errorf("cuda: GQA-KV k/v must be [%d,%d], got k%v v%v", seqKV, wkv, k.shape(), v.shape())
	}
	return gqaCore(q, k, v, qHeads, kvHeads, hd, seqKV-seqQ)
}

// gqaCore runs the batched QKᵀ → fused scale/causal-mask/softmax → ·V pipeline
// for q [seqQ,·] against k/v [seqKV,·]. offset is passed to the mask: key j is
// kept for query row i when j ≤ i+offset (offset≥seqKV disables masking).
func gqaCore(q, k, v *DeviceF32, qHeads, kvHeads, hd, offset int) (*DeviceF32, error) {
	seqQ, seqKV, wq := q.rows, k.rows, q.cols
	scores := C.cu_alloc_f32(C.int(qHeads * seqQ * seqKV))
	if scores == nil {
		return nil, fmt.Errorf("cuda: GQA scores alloc failed")
	}
	defer C.cu_free_f32(scores)
	if rc := C.cu_gqa_scores(q.ptr, k.ptr, scores, C.int(seqQ), C.int(seqKV), C.int(qHeads), C.int(kvHeads), C.int(hd)); rc != 0 {
		return nil, fmt.Errorf("cuda: GQA scores failed (code %d)", int(rc))
	}
	// One fused kernel: scale (1/√hd) + causal mask (offset) + stable softmax.
	if rc := C.cu_attn_softmax(scores, C.int(qHeads*seqQ), C.int(seqKV), C.float(1/math.Sqrt(float64(hd))), C.int(offset), C.int(seqQ)); rc != 0 {
		return nil, fmt.Errorf("cuda: GQA fused softmax failed (code %d)", int(rc))
	}
	out := C.cu_alloc_f32(C.int(seqQ * wq))
	if out == nil {
		return nil, fmt.Errorf("cuda: GQA out alloc failed")
	}
	if rc := C.cu_gqa_out(scores, v.ptr, out, C.int(seqQ), C.int(seqKV), C.int(qHeads), C.int(kvHeads), C.int(hd)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: GQA out failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: seqQ, cols: wq}, nil
}

func (d *DeviceF32) shape() tensor.Shape { return tensor.Shape{d.rows, d.cols} }

// Clone returns a new resident activation with a device→device copy of this
// one's data — for a residual branch, keeping x while an in-place op (RMSNorm,
// activation) runs on the copy. Free the clone when done.
func (d *DeviceF32) Clone() (*DeviceF32, error) {
	if d.ptr == nil {
		return nil, fmt.Errorf("cuda: Clone on a freed handle")
	}
	p := C.cu_clone_f32(d.ptr, C.int(d.rows*d.cols))
	if p == nil {
		return nil, fmt.Errorf("cuda: Clone failed")
	}
	return &DeviceF32{ptr: p, rows: d.rows, cols: d.cols}, nil
}

// Free releases the device buffer. Safe to call more than once.
func (d *DeviceF32) Free() {
	if d.ptr != nil {
		C.cu_free_f32(d.ptr)
		d.ptr = nil
	}
}

func init() {
	if Available() {
		backend.Register(Backend{})
	}
	// no CUDA GPU → not registered → feature detection via backend.Available() (§I4/§V4)
}
