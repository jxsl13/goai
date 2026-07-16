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
