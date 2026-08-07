//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "fmt"

// CopyCols copies a block of `width` columns from src (starting at column srcOff) into dst (starting at
// column dstOff), row by row: dst[i, dstOff+j] = src[i, srcOff+j]. Used to gather/scatter per-head slices.
func CopyCols(dst, src *DeviceF32, dstOff, srcOff, width int) error {
	if src.rows != dst.rows {
		return fmt.Errorf("cuda: CopyCols row mismatch %d != %d", src.rows, dst.rows)
	}
	if rc := C.cu_copy_cols_f32(dst.ptr, src.ptr, C.int(dst.rows), C.int(dst.cols), C.int(src.cols),
		C.int(dstOff), C.int(srcOff), C.int(width)); rc != 0 {
		return fmt.Errorf("cuda: CopyCols rc=%d", int(rc))
	}
	return nil
}

// MultiHeadAttentionBackward computes the VJP of multi-head scaled-dot-product attention. Q, K, V, dO and
// the outputs dQ, dK, dV are [L, nHeads*hd] (heads laid out as contiguous column blocks). Each head is
// independent scaled-dot-product attention over its hd columns with scale = 1/√hd; this gathers each
// head's slice, runs the single-head AttentionBackward, and scatters the result back. All device-resident.
func MultiHeadAttentionBackward(dQ, dK, dV, Q, K, V, dO *DeviceF32, nHeads int, scale float32) error {
	L, D := Q.rows, Q.cols
	if nHeads <= 0 || D%nHeads != 0 {
		return fmt.Errorf("cuda: MultiHeadAttentionBackward D=%d not divisible by nHeads=%d", D, nHeads)
	}
	hd := D / nHeads
	// Per-head contiguous scratch.
	mk := func() (*DeviceF32, error) { return NewDeviceF32(L, hd) }
	Qh, err := mk()
	if err != nil {
		return err
	}
	Kh, _ := mk()
	Vh, _ := mk()
	dOh, _ := mk()
	dQh, _ := mk()
	dKh, _ := mk()
	dVh, _ := mk()
	defer Qh.Free()
	defer Kh.Free()
	defer Vh.Free()
	defer dOh.Free()
	defer dQh.Free()
	defer dKh.Free()
	defer dVh.Free()

	for h := 0; h < nHeads; h++ {
		off := h * hd
		if err := CopyCols(Qh, Q, 0, off, hd); err != nil {
			return err
		}
		if err := CopyCols(Kh, K, 0, off, hd); err != nil {
			return err
		}
		if err := CopyCols(Vh, V, 0, off, hd); err != nil {
			return err
		}
		if err := CopyCols(dOh, dO, 0, off, hd); err != nil {
			return err
		}
		if err := AttentionBackward(dQh, dKh, dVh, Qh, Kh, Vh, dOh, scale); err != nil {
			return err
		}
		if err := CopyCols(dQ, dQh, off, 0, hd); err != nil {
			return err
		}
		if err := CopyCols(dK, dKh, off, 0, hd); err != nil {
			return err
		}
		if err := CopyCols(dV, dVh, off, 0, hd); err != nil {
			return err
		}
	}
	return nil
}
