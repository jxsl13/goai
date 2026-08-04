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

// quantQ8 encodes a [K,N] f32 weight into a GPU-resident Q8_0 projection — the Q8 twin of
// quantQ4K. NewResidentBQ8 takes the in-major [K,N] orientation directly (no pre-encoded
// block layer), so the wrapper is a straight upload.
func quantQ8(w *tensor.Tensor) (qProj, error) {
	return cuda.NewResidentBQ8(w)
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

// benchQ4K reports the Q4_K GEMV's effective weight bandwidth (device-only, single
// sync — the PERF-GEMV-CEILING method) so its distance to the RTX 3060's ≈360 GB/s
// peak is a standing number next to the Q8 (1.125 B/w) and asym-Q4 (0.75 B/w) probes.
func benchQ4K(b *testing.B, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(3))
	w := tensor.New(tensor.F32, tensor.Shape{k, n})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, k)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := quantQ4K(w)
	if err != nil {
		b.Fatal(err)
	}
	defer rq.Free()
	da, _ := cuda.NewDeviceF32(1, k)
	defer da.Free()
	da.UploadF32(a)
	out, _ := cuda.NewDeviceF32(1, n)
	defer out.Free()
	rq.QMatMulInto(da, out)
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		rq.QMatMulInto(da, out)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(k)*float64(n)*0.5625/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func benchQ4KBatch(b *testing.B, mrows, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(3))
	w := tensor.New(tensor.F32, tensor.Shape{k, n})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, mrows*k)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := quantQ4K(w)
	if err != nil {
		b.Fatal(err)
	}
	defer rq.Free()
	da, _ := cuda.NewDeviceF32(mrows, k)
	defer da.Free()
	da.UploadF32(a)
	out, _ := cuda.NewDeviceF32(mrows, n)
	defer out.Free()
	rq.QMatMulInto(da, out)
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		rq.QMatMulInto(da, out)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(k)*float64(n)*0.5625/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func BenchmarkQ4KBatch_m4_2048x5632(b *testing.B) { benchQ4KBatch(b, 4, 2048, 5632) }
func BenchmarkQ4KBatch_m2_2048x5632(b *testing.B) { benchQ4KBatch(b, 2, 2048, 5632) }
func BenchmarkQ4KBatch_m3_2048x5632(b *testing.B) { benchQ4KBatch(b, 3, 2048, 5632) }

func BenchmarkGemvQ4K_2048x256(b *testing.B)   { benchQ4K(b, 2048, 256) }   // GQA k/v proj (small-N)
func BenchmarkGemvQ4K_2048(b *testing.B)       { benchQ4K(b, 2048, 2048) }  // q/o proj
func BenchmarkGemvQ4K_2048x5632(b *testing.B)  { benchQ4K(b, 2048, 5632) }  // gate/up
func BenchmarkGemvQ4K_5632x2048(b *testing.B)  { benchQ4K(b, 5632, 2048) }  // down
func BenchmarkGemvQ4K_2048x32000(b *testing.B) { benchQ4K(b, 2048, 32000) } // vocab head

// TestCUDAQ4KWMMAParity validates the tensor-core Q4_K prefill GEMM (dequant→f16→WMMA)
// against the scalar acc[8] GEMV path, within f16-accumulate tolerance.
func TestCUDAQ4KWMMAParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 64, 512, 64 // K%256==0, M,N%16==0
	rng := rand.New(rand.NewSource(91))
	w := tensor.New(tensor.F64, tensor.Shape{K, N}) // [in,out] (quantQ4K transposes)
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = rng.NormFloat64() * 0.1
	}
	qi, err := quantQ4K(w)
	must(t, err)
	rq := qi.(*cuda.ResidentBQ4K)
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
	t.Logf("Q4_K WMMA prefill vs scalar GEMV: rel-RMS %.3e", rel)
	if rel > 2e-2 {
		t.Fatalf("Q4_K WMMA diverges: rel-RMS %.3e", rel)
	}
}

func benchQ4KPrefill(b *testing.B, M, K, N int, wmma bool) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(3))
	w := tensor.New(tensor.F64, tensor.Shape{K, N})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = rng.NormFloat64() * 0.1
	}
	qi, err := quantQ4K(w)
	if err != nil {
		b.Fatal(err)
	}
	rq := qi.(*cuda.ResidentBQ4K)
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
		b.Skipf("q4k: %v", err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		run()
	}
	cuda.GraphSync()
}

func BenchmarkQ4KPrefill_M512_K4096_N4096_scalar(b *testing.B) {
	benchQ4KPrefill(b, 512, 4096, 4096, false)
}
func BenchmarkQ4KPrefill_M512_K4096_N4096_wmma(b *testing.B) {
	benchQ4KPrefill(b, 512, 4096, 4096, true)
}
