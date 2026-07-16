//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// quantMXFP4 encodes a [K,N] f32 weight into GPU-resident MXFP4 (transpose to
// output-major [N,K], gguf encoder, upload). Unlike IQ4, MXFP4 HAS a gguf encoder.
func quantMXFP4(w *tensor.Tensor) (qProj, error) {
	k, n := w.Shape()[0], w.Shape()[1]
	blocks, err := gguf.Quantize(transpose2D(w), gguf.MXFP4)
	if err != nil {
		return nil, err
	}
	return cuda.NewResidentBMXFP4FromBlocks(blocks, k, n)
}

// The MXFP4 GEMV must reproduce the EXACT dequantization of the format — reference is
// quantize→dequantize (gguf, §T555) → f64 dot; only f32 summation order may differ.
// 17-byte, 32-element blocks (E8M0 scale byte + FP4 codebook). beta=1 on a non-zero buffer.
func TestCUDAMXFP4MatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48
	rng := rand.New(rand.NewSource(37))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rq, err := quantMXFP4(w)
	must(t, err)
	defer rq.Free()
	da, err := cuda.NewDeviceF32(1, K)
	must(t, err)
	defer da.Free()
	must(t, da.UploadF32(a))
	dout, err := cuda.NewDeviceF32(1, N)
	must(t, err)
	defer dout.Free()
	must(t, rq.QMatMulInto(da, dout))
	got, err := dout.ToHost()
	must(t, err)

	blocks, err := gguf.Quantize(transpose2D(w), gguf.MXFP4)
	must(t, err)
	deq, err := gguf.Dequantize(blocks, gguf.MXFP4, N*K)
	must(t, err)
	df := deq.Storage().F32()
	ref := make([]float64, N)
	var maxAbs float64
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			ref[n] += float64(a[k]) * float64(df[n*K+k])
		}
		if abs := math.Abs(got.AtF64(0, n) - ref[n]); abs > maxAbs {
			maxAbs = abs
		}
	}
	t.Logf("MXFP4 GEMV maxAbs %.3e", maxAbs)
	if maxAbs > 1e-4 {
		t.Fatalf("MXFP4 GEMV deviates from dequant reference: maxAbs %.3e", maxAbs)
	}

	// beta=1
	init := make([]float32, N)
	for i := range init {
		init[i] = float32(rng.NormFloat64())
	}
	must(t, dout.UploadF32(init))
	must(t, rq.QMatMulAccInto(da, dout))
	got2, err := dout.ToHost()
	must(t, err)
	maxAbs = 0
	for n := 0; n < N; n++ {
		if abs := math.Abs(got2.AtF64(0, n) - (float64(init[n]) + ref[n])); abs > maxAbs {
			maxAbs = abs
		}
	}
	t.Logf("MXFP4 GEMV acc maxAbs %.3e", maxAbs)
	if maxAbs > 1e-4 {
		t.Fatalf("MXFP4 GEMV beta=1 deviates: maxAbs %.3e", maxAbs)
	}
}
