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

// quantQ40 encodes a [K,N] f32 weight into GPU-resident Q4_0 (transpose to
// output-major [N,K], gguf encoder, upload).
func quantQ40(w *tensor.Tensor) (qProj, error) {
	k, n := w.Shape()[0], w.Shape()[1]
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q4_0)
	if err != nil {
		return nil, err
	}
	return cuda.NewResidentBQ40FromBlocks(blocks, k, n)
}

// The Q4_0 GEMV must reproduce the EXACT dequantization of the format — the reference
// is quantize→dequantize (gguf) → f64 dot, so the only allowed deviation is f32
// summation order. Q4_0 has 32-element blocks (K=512 = 16 blocks). beta=1 on a
// non-zero buffer. Asserts absolute error (near-zero outputs can spike the relative).
func TestCUDAQ40MatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48
	rng := rand.New(rand.NewSource(31))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rq, err := quantQ40(w)
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

	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q4_0)
	must(t, err)
	deq, err := gguf.QuantTensor{Data: blocks, GGType: 2 /* Q4_0 */, Shape: tensor.Shape{N, K}}.Dequantize()
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
	t.Logf("Q4_0 GEMV maxAbs %.3e", maxAbs)
	if maxAbs > 1e-4 {
		t.Fatalf("Q4_0 GEMV deviates from dequant reference: maxAbs %.3e", maxAbs)
	}

	// beta=1: out += a·W on a non-zero buffer.
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
		want := float64(init[n]) + ref[n]
		if abs := math.Abs(got2.AtF64(0, n) - want); abs > maxAbs {
			maxAbs = abs
		}
	}
	t.Logf("Q4_0 GEMV acc maxAbs %.3e", maxAbs)
	if maxAbs > 1e-4 {
		t.Fatalf("Q4_0 GEMV beta=1 deviates: maxAbs %.3e", maxAbs)
	}
}
