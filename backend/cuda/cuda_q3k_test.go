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

// quantQ3K encodes a [K,N] f32 weight into GPU-resident Q3_K (transpose to
// output-major [N,K], gguf encoder, upload — same layering as quantQ5K/quantQ6K).
func quantQ3K(w *tensor.Tensor) (qProj, error) {
	k, n := w.Shape()[0], w.Shape()[1]
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q3_K)
	if err != nil {
		return nil, err
	}
	return cuda.NewResidentBQ3KFromBlocks(blocks, k, n)
}

// The Q3_K GEMV must reproduce the EXACT dequantization semantics of the format —
// the reference is quantize→dequantize (gguf, validated against gguf-py) → f64 dot,
// so the only allowed deviation is f32 summation order (~1e-5), NOT the quant error.
// Q3_K is symmetric with 16 sub-blocks of 16 and the interleaved qs/hmask/scale
// indexing (the hardest K-quant layout); K=512 = two super-blocks exercises the
// per-super-block scale splice and both nb halves. beta=1 checked on a non-zero buffer.
func TestCUDAQ3KMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48
	rng := rand.New(rand.NewSource(23))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rq, err := quantQ3K(w)
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
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q3_K)
	must(t, err)
	deq, err := gguf.QuantTensor{Data: blocks, GGType: 11 /* Q3_K */, Shape: tensor.Shape{N, K}}.Dequantize()
	must(t, err)
	df := deq.Storage().F32()
	ref := make([]float64, N)
	var maxRel float64
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			ref[n] += float64(a[k]) * float64(df[n*K+k])
		}
		rel := math.Abs(got.AtF64(0, n)-ref[n]) / math.Max(math.Abs(ref[n]), 1e-6)
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("Q3_K GEMV maxRel %.3e", maxRel)
	if maxRel > 1e-4 {
		t.Fatalf("Q3_K GEMV deviates from dequant reference: maxRel %.3e", maxRel)
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
	t.Logf("Q3_K GEMV acc maxRel %.3e", maxRel)
	if maxRel > 1e-4 {
		t.Fatalf("Q3_K GEMV beta=1 deviates: maxRel %.3e", maxRel)
	}
}

// TestCUDAQ3KWMMAParity validates the tensor-core Q3_K prefill GEMM (dequant→f16→WMMA)
// against the scalar acc GEMV path, within f16-accumulate tolerance.
func TestCUDAQ3KWMMAParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 64, 512, 64 // K%256==0, M,N%16==0
	rng := rand.New(rand.NewSource(91))
	w := tensor.New(tensor.F64, tensor.Shape{K, N}) // [in,out] (quantQ3K transposes)
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = rng.NormFloat64() * 0.1
	}
	qi, err := quantQ3K(w)
	must(t, err)
	rq := qi.(*cuda.ResidentBQ3K)
	af := make([]float32, M*K)
	for i := range af {
		af[i] = float32(rng.NormFloat64()) * 0.2
	}
	da, _ := cuda.NewDeviceF32(M, K)
	must(t, da.UploadF32(af))
	defer da.Free()
	ref, _ := cuda.NewDeviceF32(M, N)
	defer ref.Free()
	got, _ := cuda.NewDeviceF32(M, N)
	defer got.Free()
	must(t, rq.QMatMulInto(da, ref))     // scalar oracle
	must(t, rq.QMatMulWMMAInto(da, got)) // tensor-core path
	a := make([]float32, M*N)
	b := make([]float32, M*N)
	ref.DownloadF32(a)
	got.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	rel := math.Sqrt(num / den)
	t.Logf("Q3_K WMMA prefill vs scalar GEMV: rel-RMS %.3e", rel)
	if rel > 2e-2 {
		t.Fatalf("Q3_K WMMA diverges: rel-RMS %.3e", rel)
	}
}

func benchQ3KPrefill(b *testing.B, M, K, N int, wmma bool) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(3))
	w := tensor.New(tensor.F64, tensor.Shape{K, N})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = rng.NormFloat64() * 0.1
	}
	qi, err := quantQ3K(w)
	if err != nil {
		b.Fatal(err)
	}
	rq := qi.(*cuda.ResidentBQ3K)
	af := make([]float32, M*K)
	for i := range af {
		af[i] = float32(rng.NormFloat64()) * 0.2
	}
	da, _ := cuda.NewDeviceF32(M, K)
	da.UploadF32(af)
	defer da.Free()
	out, _ := cuda.NewDeviceF32(M, N)
	defer out.Free()
	run := func() error {
		if wmma {
			return rq.QMatMulWMMAInto(da, out)
		}
		return rq.QMatMulInto(da, out)
	}
	if err := run(); err != nil {
		b.Skipf("q3k: %v", err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		run()
	}
	cuda.GraphSync()
}

func BenchmarkQ3KPrefill_M512_K4096_N4096_scalar(b *testing.B) {
	benchQ3KPrefill(b, 512, 4096, 4096, false)
}
func BenchmarkQ3KPrefill_M512_K4096_N4096_wmma(b *testing.B) {
	benchQ3KPrefill(b, 512, 4096, 4096, true)
}
