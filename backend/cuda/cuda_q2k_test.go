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

// quantQ2K encodes a [K,N] f32 weight into GPU-resident Q2_K (transpose to
// output-major [N,K], gguf encoder, upload — same layering as the other K-quants).
func quantQ2K(w *tensor.Tensor) (qProj, error) {
	k, n := w.Shape()[0], w.Shape()[1]
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q2_K)
	if err != nil {
		return nil, err
	}
	return cuda.NewResidentBQ2KFromBlocks(blocks, k, n)
}

// The Q2_K GEMV must reproduce the EXACT dequantization semantics of the format —
// the reference is quantize→dequantize (gguf, validated against gguf-py) → f64 dot,
// so the only allowed deviation is f32 summation order (~1e-5), NOT the quant error.
// Q2_K is asymmetric affine with 16 sub-blocks of 16 (4-bit scale+min nibbles, 2-bit
// quants); K=512 = two super-blocks covers both nb halves. beta=1 on a non-zero buffer.
func TestCUDAQ2KMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48
	rng := rand.New(rand.NewSource(29))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rq, err := quantQ2K(w)
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

	// host reference: same blocks → dequant → f64 dot
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q2_K)
	must(t, err)
	deq, err := gguf.QuantTensor{Data: blocks, GGType: 10 /* Q2_K */, Shape: tensor.Shape{N, K}}.Dequantize()
	must(t, err)
	df := deq.Storage().F32()
	ref := make([]float64, N)
	var maxRel, maxAbs, worstRefMag float64
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			ref[n] += float64(a[k]) * float64(df[n*K+k])
		}
		abs := math.Abs(got.AtF64(0, n) - ref[n])
		if abs > maxAbs {
			maxAbs = abs
		}
		rel := abs / math.Max(math.Abs(ref[n]), 1e-6)
		if rel > maxRel {
			maxRel, worstRefMag = rel, math.Abs(ref[n])
		}
	}
	t.Logf("Q2_K GEMV maxRel %.3e (worst |ref|=%.3e), maxAbs %.3e", maxRel, worstRefMag, maxAbs)
	if maxAbs > 1e-4 {
		t.Fatalf("Q2_K GEMV deviates from dequant reference: maxAbs %.3e", maxAbs)
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
	maxRel = 0
	for n := 0; n < N; n++ {
		want := float64(init[n]) + ref[n]
		rel := math.Abs(got2.AtF64(0, n)-want) / math.Max(math.Abs(want), 1e-6)
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("Q2_K GEMV acc maxRel %.3e", maxRel)
	if maxRel > 1e-4 {
		t.Fatalf("Q2_K GEMV beta=1 deviates: maxRel %.3e", maxRel)
	}
}
