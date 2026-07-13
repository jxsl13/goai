//go:build cuda && cgo && (linux || windows)

package cuda

/*
#cgo LDFLAGS: -lcudart -lcublas
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"

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

func init() {
	if Available() {
		backend.Register(Backend{})
	}
	// no CUDA GPU → not registered → feature detection via backend.Available() (§I4/§V4)
}
