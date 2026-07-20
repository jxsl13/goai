//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/jxsl13/goai/backend"
)

// DeviceF16 is a device-resident f16 (u16) activation buffer — the A1 fp16-activation path's clean
// type. Threading DeviceF16 through the decode forward (vs f32 DeviceF32) drops the per-GEMM
// f32<->f16 conversions and halves activation traffic: ~1.47x on the batched decode step, at
// vLLM's f16-activation precision. Weights (ResidentBF16) and gammas (ResidentVec) stay resident.
type DeviceF16 struct {
	ptr        unsafe.Pointer
	rows, cols int
}

// NewDeviceF16 allocates an uninitialized [rows,cols] f16 buffer.
func NewDeviceF16(rows, cols int) (*DeviceF16, error) {
	p := C.cu_alloc_u16(C.int(rows * cols))
	if p == nil {
		return nil, fmt.Errorf("cuda: NewDeviceF16 alloc [%d,%d] failed", rows, cols)
	}
	return &DeviceF16{ptr: unsafe.Pointer(p), rows: rows, cols: cols}, nil
}

func (d *DeviceF16) Rows() int { return d.rows }
func (d *DeviceF16) Cols() int { return d.cols }

// Free releases the buffer.
func (d *DeviceF16) Free() {
	if d.ptr != nil {
		C.cu_free_f32(d.ptr)
		d.ptr = nil
	}
}

// F16FromF32 converts an f32 device buffer to a fresh f16 buffer.
func F16FromF32(src *DeviceF32) (*DeviceF16, error) {
	d, err := NewDeviceF16(src.rows, src.cols)
	if err != nil {
		return nil, err
	}
	if rc := C.cu_cvt_f32_to_f16(d.ptr, src.ptr, C.long(src.rows*src.cols)); rc != 0 {
		d.Free()
		return nil, fmt.Errorf("cuda: F16FromF32 rc=%d", int(rc))
	}
	return d, nil
}

// ToF32 converts back to a fresh f32 device buffer.
func (d *DeviceF16) ToF32() (*DeviceF32, error) {
	out, err := NewDeviceF32(d.rows, d.cols)
	if err != nil {
		return nil, err
	}
	if rc := C.cu_cvt_f16_to_f32(out.ptr, d.ptr, C.long(d.rows*d.cols)); rc != 0 {
		out.Free()
		return nil, fmt.Errorf("cuda: DeviceF16.ToF32 rc=%d", int(rc))
	}
	return out, nil
}

// RMSNormInto writes RMSNorm(d)·gamma into out (both [rows,cols] f16); gamma is resident f32.
func (d *DeviceF16) RMSNormInto(gamma *ResidentVec, eps float32, out *DeviceF16) error {
	if out.rows != d.rows || out.cols != d.cols {
		return fmt.Errorf("cuda: RMSNormInto shape mismatch")
	}
	if rc := C.cu_rmsnorm_f16(d.ptr, out.ptr, gamma.ptr, C.int(d.rows), C.int(d.cols), C.float(eps)); rc != 0 {
		return fmt.Errorf("cuda: RMSNormInto rc=%d", int(rc))
	}
	return nil
}

// MatMulF16 computes C[rows,N] = a[rows,K]·W[K,N] with the pure-f16 GEMM (no conversion).
func (r *ResidentBF16) MatMulF16(a *DeviceF16) (*DeviceF16, error) {
	if a.cols != r.k {
		return nil, fmt.Errorf("cuda: MatMulF16 inner dim a[%d,%d]·W[%d,%d]", a.rows, a.cols, r.k, r.n)
	}
	out, err := NewDeviceF16(a.rows, r.n)
	if err != nil {
		return nil, err
	}
	if rc := C.cu_gemm_f16_pure(a.ptr, r.ptr, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n)); rc != 0 {
		out.Free()
		return nil, fmt.Errorf("cuda: MatMulF16 rc=%d", int(rc))
	}
	return out, nil
}

// RoPE applies rotary position embedding in place (inv is a resident f32 [hd/2] table).
func (d *DeviceF16) RoPE(inv *DeviceF32, attrs backend.RoPEAttrs) error {
	heads := attrs.Heads
	if heads <= 0 {
		heads = 1
	}
	hd := d.cols / heads
	_, posDiv := backend.RoPEFreqs(hd, attrs)
	if rc := C.cu_rope_f16(d.ptr, inv.ptr, C.int(d.rows), C.int(heads), C.int(hd), C.int(attrs.PosOffset), C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: DeviceF16.RoPE rc=%d", int(rc))
	}
	return nil
}

// SwiGLU applies gate = silu(gate)·up in place.
func (d *DeviceF16) SwiGLU(up *DeviceF16) error {
	if up.rows != d.rows || up.cols != d.cols {
		return fmt.Errorf("cuda: DeviceF16.SwiGLU shape mismatch")
	}
	if rc := C.cu_swiglu_f16(d.ptr, up.ptr, C.int(d.rows*d.cols)); rc != 0 {
		return fmt.Errorf("cuda: DeviceF16.SwiGLU rc=%d", int(rc))
	}
	return nil
}

// Add accumulates src into d in place (the residual add).
func (d *DeviceF16) Add(src *DeviceF16) error {
	if src.rows != d.rows || src.cols != d.cols {
		return fmt.Errorf("cuda: DeviceF16.Add shape mismatch")
	}
	if rc := C.cu_add_f16(d.ptr, src.ptr, C.int(d.rows*d.cols)); rc != 0 {
		return fmt.Errorf("cuda: DeviceF16.Add rc=%d", int(rc))
	}
	return nil
}

// BatchArgmax returns the greedy-sampled token index (argmax) of each row of the f16 logits
// [rows,cols] — the serving loop's sampling step. On-device reduction, only rows ints come back.
func (d *DeviceF16) BatchArgmax() ([]int32, error) {
	out := make([]int32, d.rows)
	if d.rows == 0 {
		return out, nil
	}
	if rc := C.cu_argmax_batched_f16(d.ptr, (*C.int)(&out[0]), C.int(d.rows), C.int(d.cols)); rc != 0 {
		return nil, fmt.Errorf("cuda: BatchArgmax rc=%d", int(rc))
	}
	return out, nil
}

// RoPEDpos applies RoPE in place using a device-resident position (for graph-capturable decode:
// the captured graph reads *pos, which is advanced via DevicePos.Set between launches).
func (d *DeviceF16) RoPEDpos(inv *DeviceF32, attrs backend.RoPEAttrs, pos *DevicePos) error {
	heads := attrs.Heads
	if heads <= 0 {
		heads = 1
	}
	hd := d.cols / heads
	_, posDiv := backend.RoPEFreqs(hd, attrs)
	if rc := C.cu_rope_f16_dpos(d.ptr, inv.ptr, C.int(d.rows), C.int(heads), C.int(hd), pos.ptr, C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: DeviceF16.RoPEDpos rc=%d", int(rc))
	}
	return nil
}
