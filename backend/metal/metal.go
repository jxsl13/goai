//go:build metal && darwin && cgo

package metal

/*
#cgo CFLAGS: -fobjc-arc -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
#include "metal_bridge.h"
*/
import "C"

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// device is the metal tensor.Device. Tensor memory stays host-side (Apple
// Silicon UMA); the kernel copies through shared MTLBuffers per call — honest
// about the current transfer cost; zero-copy is a later optimization.
type device struct{}

func (device) Kind() tensor.DeviceKind    { return tensor.KindMetal }
func (device) String() string             { return "metal" }
func (device) Allocator() tensor.Allocator { return tensor.Heap() }

// Backend implements backend.Backend over Metal/MPS. Synchronous: every kernel
// commits and waits before returning, so Synchronize is a no-op (§V14 permits
// async later without an API break).
type Backend struct{}

func (Backend) Name() string          { return "metal" }
func (Backend) Device() tensor.Device { return device{} }
func (Backend) Synchronize() error    { return nil }

// Available reports whether a Metal device with MPS support exists.
func Available() bool { return C.mtl_available() == 1 }

func (Backend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpMatMul && dtype == tensor.F32 {
		return matmulF32, true
	}
	return nil, false // everything else: fallback to Pure-Go (§I4)
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
		return backend.Execute(ctx.WithBackend(backend.Reference()), backend.OpMatMul, in, nil)
	}
	ac, bc := a.Contiguous(), b.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	rc := C.mtl_matmul_f32(
		(*C.float)(&ac.Storage().F32()[0]),
		(*C.float)(&bc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal: matmul failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	if Available() {
		backend.Register(Backend{})
	}
	// no device → not registered → feature detection via backend.Available() (§I4/§V4)
}
