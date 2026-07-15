//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Blit + Copy2D (the fused-QKV band-move primitives the llamagpu CUDA adapter needs)
// must match a host reference.
func TestCUDABlitCopy2D(t *testing.T) {
	skipNoGPU(t)

	// Blit: copy src[3:3+8] → dst[5:5+8]
	{
		src := bench.RandF32(tensor.Shape{1, 32}, 1)
		dst := bench.RandF32(tensor.Shape{1, 32}, 2)
		dsrc, _ := cuda.UploadF32(src)
		ddst, _ := cuda.UploadF32(dst)
		must(t, ddst.Blit(5, dsrc, 3, 8))
		got, _ := ddst.ToHost()
		want := dst.Contiguous().Storage().F32()
		copy(want[5:13], src.Contiguous().Storage().F32()[3:11])
		for i := 0; i < 32; i++ {
			if got.AtF64(0, i) != float64(want[i]) {
				t.Fatalf("Blit[%d]: %v != %v", i, got.AtF64(0, i), want[i])
			}
		}
		dsrc.Free()
		ddst.Free()
		t.Log("Blit == host reference")
	}

	// Copy2D: extract a [4, 6] band from src (row stride 20, off 2) into dst (row stride 6, off 0).
	{
		const rows, rf, srcStride, srcOff, dstStride, dstOff = 4, 6, 20, 2, 6, 0
		src := bench.RandF32(tensor.Shape{1, rows * srcStride}, 3)
		dst := bench.RandF32(tensor.Shape{1, rows * dstStride}, 4)
		dsrc, _ := cuda.UploadF32(src)
		ddst, _ := cuda.UploadF32(dst)
		must(t, ddst.Copy2D(dstOff, dstStride, dsrc, srcOff, srcStride, rows, rf))
		got, _ := ddst.ToHost()
		sf := src.Contiguous().Storage().F32()
		want := append([]float32{}, dst.Contiguous().Storage().F32()...)
		for r := 0; r < rows; r++ {
			for c := 0; c < rf; c++ {
				want[dstOff+r*dstStride+c] = sf[srcOff+r*srcStride+c]
			}
		}
		for i := range want {
			if got.AtF64(0, i) != float64(want[i]) {
				t.Fatalf("Copy2D[%d]: %v != %v", i, got.AtF64(0, i), want[i])
			}
		}
		dsrc.Free()
		ddst.Free()
		t.Log("Copy2D == host reference")
	}
}
