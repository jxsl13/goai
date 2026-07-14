//go:build cuda && cgo && (linux || windows)

package cuda

/*
#cgo LDFLAGS: -lcudart -lcublas -lnvrtc -lcuda
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
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

// Free releases the device buffer. Safe to call more than once.
func (r *ResidentB) Free() {
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
