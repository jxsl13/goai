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

// device is the Vulkan tensor.Device. Tensor memory stays host-side; the kernel
// copies through host-visible buffers per call — honest about transfer cost.
type device struct{}

func (device) Kind() tensor.DeviceKind     { return tensor.KindCUDA } // no KindVulkan yet; generic accel
func (device) String() string              { return "vulkan" }
func (device) Allocator() tensor.Allocator { return tensor.Heap() }

// Backend implements backend.Backend over Vulkan compute. Synchronous: every
// kernel submits and vkQueueWaitIdle before returning, so Synchronize is a no-op
// (§V14 permits async later without an API break).
type Backend struct{}

func (Backend) Name() string          { return "vulkan" }
func (Backend) Device() tensor.Device { return device{} }
func (Backend) Synchronize() error    { return nil }

// Available reports whether a Vulkan compute-capable device is present.
func Available() bool { return C.vk_available() == 1 }

func (Backend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpMatMul && dtype == tensor.F32 {
		return matmulF32, true
	}
	return nil, false // everything else: fallback to Pure-Go (§I4)
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
		return backend.Execute(ctx.WithBackend(backend.Reference()), backend.OpMatMul, in, nil)
	}
	if len(spirv) == 0 {
		return nil, fmt.Errorf("vulkan: matmul.spv not embedded — run `make vulkan-spv` (glslc)")
	}
	ac, bc := a.Contiguous(), b.Contiguous()
	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	rc := C.vk_matmul_f32(
		(*C.uint32_t)(unsafe.Pointer(&spirv[0])), C.int(len(spirv)),
		(*C.float)(&ac.Storage().F32()[0]),
		(*C.float)(&bc.Storage().F32()[0]),
		(*C.float)(&out.Storage().F32()[0]),
		C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("vulkan: matmul failed (code %d)", int(rc))
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	if Available() {
		backend.Register(Backend{})
	}
	// no Vulkan device → not registered → feature detection via backend.Available() (§I4/§V4)
}
