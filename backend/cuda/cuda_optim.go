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
)

// DeviceAdam is a GPU-resident AdamW optimizer: the parameters, gradients, and the first/second moment
// state all live on the device, so no optimizer state crosses the PCIe bus between steps — the primitive
// a GPU training path needs (GoAI's optimizers were CPU-only). Moments are kept in f64 (the master-state
// analog of nn.Adam), so the update matches the CPU nn.Adam f32-parameter fast path bit-for-bit. Build it
// with the per-parameter element counts; Step consumes device parameter + gradient buffers in that order.
type DeviceAdam struct {
	LR, Beta1, Beta2, Eps, WeightDecay float64

	m, v  []unsafe.Pointer // per-param device f64 moment buffers (allocated as 2n f32 = n f64)
	sizes []int
	t     int
}

// NewDeviceAdam allocates zeroed f64 moment buffers for parameters of the given element counts.
func NewDeviceAdam(sizes []int, lr, beta1, beta2, eps, weightDecay float64) (*DeviceAdam, error) {
	o := &DeviceAdam{
		LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps, WeightDecay: weightDecay,
		sizes: append([]int(nil), sizes...),
		m:     make([]unsafe.Pointer, len(sizes)),
		v:     make([]unsafe.Pointer, len(sizes)),
	}
	for i, n := range sizes {
		if n <= 0 {
			o.Free()
			return nil, fmt.Errorf("cuda: DeviceAdam param %d size %d", i, n)
		}
		o.m[i] = unsafe.Pointer(C.cu_alloc_f32(C.int(2 * n))) // 8n bytes = n f64
		o.v[i] = unsafe.Pointer(C.cu_alloc_f32(C.int(2 * n)))
		if o.m[i] == nil || o.v[i] == nil {
			o.Free()
			return nil, fmt.Errorf("cuda: DeviceAdam moment alloc failed (param %d)", i)
		}
		C.cu_zero_f32(o.m[i], C.int(2*n))
		C.cu_zero_f32(o.v[i], C.int(2*n))
	}
	return o, nil
}

// Step performs one AdamW update: params[i] is updated in place from grads[i] (both device f32 buffers
// whose element count matches the size given at construction, order-aligned). A nil grad skips that
// parameter (gradient accumulation / frozen params). Bias-correction reciprocals and the decoupled decay
// factor are hoisted per step exactly as nn.Adam does.
func (o *DeviceAdam) Step(params, grads []*DeviceF32) error {
	if len(params) != len(o.sizes) || len(grads) != len(o.sizes) {
		return fmt.Errorf("cuda: DeviceAdam.Step got %d params / %d grads, want %d", len(params), len(grads), len(o.sizes))
	}
	o.t++
	ic1 := 1.0 / (1.0 - math.Pow(o.Beta1, float64(o.t)))
	ic2 := 1.0 / (1.0 - math.Pow(o.Beta2, float64(o.t)))
	decay := 1.0 - o.LR*o.WeightDecay
	for i := range params {
		if grads[i] == nil {
			continue
		}
		n := o.sizes[i]
		if params[i].rows*params[i].cols != n || grads[i].rows*grads[i].cols != n {
			return fmt.Errorf("cuda: DeviceAdam.Step param %d numel mismatch (want %d)", i, n)
		}
		rc := C.cu_adam_step_f32(params[i].ptr, grads[i].ptr, o.m[i], o.v[i], C.int(n),
			C.double(o.LR), C.double(o.Beta1), C.double(o.Beta2), C.double(o.Eps),
			C.double(decay), C.double(ic1), C.double(ic2))
		if rc != 0 {
			return fmt.Errorf("cuda: DeviceAdam.Step param %d rc=%d", i, int(rc))
		}
	}
	return nil
}

// Free releases the moment buffers.
func (o *DeviceAdam) Free() {
	for _, p := range o.m {
		if p != nil {
			C.cu_free_f32(p)
		}
	}
	for _, p := range o.v {
		if p != nil {
			C.cu_free_f32(p)
		}
	}
	o.m, o.v = nil, nil
}
