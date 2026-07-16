//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
)

// RoPEPartial applies PARTIAL rotary position embeddings in place to d[rows, heads*hd]:
// only the first rotaryDim channels of each head are rotated (HF rotate_half within the
// rotaryDim block — pair (i, i+rotaryDim/2)), the remaining hd-rotaryDim channels pass
// through unchanged. This is the "partial rotary" of GPT-NeoX / Phi / StableLM
// (partial_rotary_factor < 1). When rotaryDim == hd it is exactly [DeviceF32.RoPE].
//
// The frequency table is built over rotaryDim (not hd), so PI/YaRN scaling matches the
// backend/ref via [backend.RoPEFreqs]. rotaryDim must be even, > 0, and <= hd.
func (d *DeviceF32) RoPEPartial(attrs backend.RoPEAttrs, rotaryDim int) error {
	if d.ptr == nil {
		return fmt.Errorf("cuda: RoPEPartial on a freed handle")
	}
	if attrs.XPos {
		return fmt.Errorf("cuda: RoPEPartial XPos not supported on device yet")
	}
	heads := attrs.Heads
	if heads <= 0 {
		heads = 1
	}
	if d.cols%heads != 0 {
		return fmt.Errorf("cuda: RoPEPartial width %d not divisible by heads %d", d.cols, heads)
	}
	hd := d.cols / heads
	if rotaryDim <= 0 || rotaryDim > hd {
		return fmt.Errorf("cuda: RoPEPartial rotaryDim %d must be in (0,%d]", rotaryDim, hd)
	}
	if rotaryDim%2 != 0 {
		return fmt.Errorf("cuda: RoPEPartial rotaryDim %d must be even", rotaryDim)
	}
	// Frequencies over the rotary sub-dim (a full RoPE on a rotaryDim-wide head).
	inv, posDiv := backend.RoPEFreqs(rotaryDim, attrs)
	inv32 := make([]float32, len(inv))
	for i, v := range inv {
		inv32[i] = float32(v)
	}
	invPtr := C.cu_upload_f32((*C.float)(&inv32[0]), C.int(len(inv32)))
	if invPtr == nil {
		return fmt.Errorf("cuda: RoPEPartial frequency upload failed")
	}
	defer C.cu_free_f32(invPtr)
	if rc := C.cu_rope_partial(d.ptr, invPtr, C.int(d.rows), C.int(heads), C.int(hd), C.int(rotaryDim), C.int(attrs.PosOffset), C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: rope_partial failed (code %d)", int(rc))
	}
	return nil
}

// RoPEPartialDpos is the device-position twin of [DeviceF32.RoPEPartial]: it reads the
// absolute position from the device int pos (updated between captured graph replays via
// [DevicePos.Set]) instead of a host PosOffset, so a graph-captured decode of the
// partial-rotary architectures rotates at the token's true position without re-capture.
// Uses the same backend.RoPEFreqs over rotaryDim, so at a matched position it is identical
// to RoPEPartial.
func (d *DeviceF32) RoPEPartialDpos(attrs backend.RoPEAttrs, rotaryDim int, pos *DevicePos) error {
	if d.ptr == nil {
		return fmt.Errorf("cuda: RoPEPartialDpos on a freed handle")
	}
	if attrs.XPos {
		return fmt.Errorf("cuda: RoPEPartialDpos XPos not supported on device yet")
	}
	heads := attrs.Heads
	if heads <= 0 {
		heads = 1
	}
	if d.cols%heads != 0 {
		return fmt.Errorf("cuda: RoPEPartialDpos width %d not divisible by heads %d", d.cols, heads)
	}
	hd := d.cols / heads
	if rotaryDim <= 0 || rotaryDim > hd {
		return fmt.Errorf("cuda: RoPEPartialDpos rotaryDim %d must be in (0,%d]", rotaryDim, hd)
	}
	if rotaryDim%2 != 0 {
		return fmt.Errorf("cuda: RoPEPartialDpos rotaryDim %d must be even", rotaryDim)
	}
	inv, posDiv := backend.RoPEFreqs(rotaryDim, attrs)
	inv32 := make([]float32, len(inv))
	for i, v := range inv {
		inv32[i] = float32(v)
	}
	invPtr := C.cu_upload_f32((*C.float)(&inv32[0]), C.int(len(inv32)))
	if invPtr == nil {
		return fmt.Errorf("cuda: RoPEPartialDpos frequency upload failed")
	}
	defer C.cu_free_f32(invPtr)
	if rc := C.cu_rope_partial_dpos(d.ptr, invPtr, C.int(d.rows), C.int(heads), C.int(hd), C.int(rotaryDim), pos.ptr, C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: rope_partial_dpos failed (code %d)", int(rc))
	}
	return nil
}

// RoPEPartialBand applies partial rotary in place to a strided BAND of this [rows, stride]
// buffer: `heads` heads (hd wide each) starting at float-element column off, with only the
// first rotaryDim channels of each head rotated. It is [DeviceF32.RoPEPartial] generalised
// with a row stride + column offset — the fused-QKV band path (§T613) for the partial-rotary
// architectures, rotating the q or k band of a single [rows, stride] QKV buffer without a
// copy-out. stride is this buffer's cols; off+heads*hd must fit within it.
func (d *DeviceF32) RoPEPartialBand(attrs backend.RoPEAttrs, rotaryDim, off, heads, hd int) error {
	if d.ptr == nil {
		return fmt.Errorf("cuda: RoPEPartialBand on a freed handle")
	}
	if heads <= 0 || hd <= 0 {
		return fmt.Errorf("cuda: RoPEPartialBand needs heads>0, hd>0, got %d,%d", heads, hd)
	}
	if off < 0 || off+heads*hd > d.cols {
		return fmt.Errorf("cuda: RoPEPartialBand band [%d,%d) exceeds stride %d", off, off+heads*hd, d.cols)
	}
	if rotaryDim <= 0 || rotaryDim > hd || rotaryDim%2 != 0 {
		return fmt.Errorf("cuda: RoPEPartialBand rotaryDim %d must be even, in (0,%d]", rotaryDim, hd)
	}
	inv, posDiv := backend.RoPEFreqs(rotaryDim, attrs)
	inv32 := make([]float32, len(inv))
	for i, v := range inv {
		inv32[i] = float32(v)
	}
	invPtr := C.cu_upload_f32((*C.float)(&inv32[0]), C.int(len(inv32)))
	if invPtr == nil {
		return fmt.Errorf("cuda: RoPEPartialBand frequency upload failed")
	}
	defer C.cu_free_f32(invPtr)
	if rc := C.cu_rope_partial_band(d.ptr, invPtr, C.int(d.rows), C.int(d.cols), C.int(off), C.int(heads), C.int(hd), C.int(rotaryDim), C.int(attrs.PosOffset), C.double(posDiv)); rc != 0 {
		return fmt.Errorf("cuda: rope_partial_band failed (code %d)", int(rc))
	}
	return nil
}
