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

// quantQ4K encodes a [K,N] f32 weight into GPU-resident Q4_K: transpose to the
// output-major [N,K] orientation, encode with the gguf Q4_K encoder (layering: the
// cuda package only accepts pre-encoded blocks), upload.
func quantQ4K(w *tensor.Tensor) (qProj, error) {
	k, n := w.Shape()[0], w.Shape()[1]
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q4_K)
	if err != nil {
		return nil, err
	}
	return cuda.NewResidentBQ4KFromBlocks(blocks, k, n)
}

// The Q4_K GEMV must reproduce the EXACT dequantization semantics of the format —
// the reference is quantize→dequantize (gguf, validated against gguf-py) →f32 matmul,
// so the only allowed deviation is f32 summation order (~1e-5), NOT the quant error.
func TestCUDAQ4KMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48
	rng := rand.New(rand.NewSource(7))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	// device
	rq, err := quantQ4K(w)
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
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q4_K)
	must(t, err)
	deq, err := gguf.QuantTensor{Data: blocks, GGType: 12 /* Q4_K */, Shape: tensor.Shape{N, K}}.Dequantize()
	must(t, err)
	df := deq.Storage().F32()
	var maxRel float64
	for n := 0; n < N; n++ {
		var ref float64
		for k := 0; k < K; k++ {
			ref += float64(a[k]) * float64(df[n*K+k])
		}
		g := got.AtF64(0, n)
		rel := math.Abs(g-ref) / math.Max(math.Abs(ref), 1e-6)
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("Q4_K GEMV vs exact-dequant reference: max rel err %.3g", maxRel)
	if maxRel > 1e-4 { // f32 accumulation order only — NOT the quantization budget
		t.Fatalf("Q4_K kernel deviates from the format's dequant semantics: max rel %.3g", maxRel)
	}

	// beta=1 residual fuse
	must(t, rq.QMatMulAccInto(da, dout))
	got2, err := dout.ToHost()
	must(t, err)
	for n := 0; n < N; n++ {
		want := 2 * got.AtF64(0, n)
		if math.Abs(got2.AtF64(0, n)-want) > 1e-3*math.Max(math.Abs(want), 1) {
			t.Fatalf("QMatMulAccInto beta=1 wrong at %d: %g want %g", n, got2.AtF64(0, n), want)
		}
	}
}
