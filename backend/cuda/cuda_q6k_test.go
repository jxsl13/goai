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

// quantQ6K encodes a [K,N] f32 weight into GPU-resident Q6_K (transpose to
// output-major [N,K], gguf encoder, upload — same layering as quantQ4K).
func quantQ6K(w *tensor.Tensor) (qProj, error) {
	k, n := w.Shape()[0], w.Shape()[1]
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q6_K)
	if err != nil {
		return nil, err
	}
	return cuda.NewResidentBQ6KFromBlocks(blocks, k, n)
}

// The Q6_K GEMV must reproduce the EXACT dequantization semantics of the format —
// the reference is quantize→dequantize (gguf, validated against gguf-py) → f64 dot,
// so the only allowed deviation is f32 summation order (~1e-5), NOT the quant error.
// K=512 covers two super-blocks per row (the group/sub-block indexing seams); the
// beta=1 accumulate variant is checked on top of a non-zero out buffer.
func TestCUDAQ6KMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48
	rng := rand.New(rand.NewSource(11))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rq, err := quantQ6K(w)
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
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q6_K)
	must(t, err)
	deq, err := gguf.QuantTensor{Data: blocks, GGType: 14 /* Q6_K */, Shape: tensor.Shape{N, K}}.Dequantize()
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
	t.Logf("Q6_K GEMV maxRel %.3e", maxRel)
	if maxRel > 1e-4 {
		t.Fatalf("Q6_K GEMV deviates from dequant reference: maxRel %.3e", maxRel)
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
	t.Logf("Q6_K GEMV acc maxRel %.3e", maxRel)
	if maxRel > 1e-4 {
		t.Fatalf("Q6_K GEMV beta=1 deviates: maxRel %.3e", maxRel)
	}
}

// TestCUDAQ6KWMMAParity validates the tensor-core Q6_K prefill GEMM (dequant→f16→WMMA)
// against the scalar acc GEMV path, within f16-accumulate tolerance.
func TestCUDAQ6KWMMAParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 64, 512, 64 // K%256==0, M,N%16==0
	rng := rand.New(rand.NewSource(91))
	w := tensor.New(tensor.F64, tensor.Shape{K, N}) // [in,out] (quantQ6K transposes)
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = rng.NormFloat64() * 0.1
	}
	qi, err := quantQ6K(w)
	must(t, err)
	rq := qi.(*cuda.ResidentBQ6K)
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
	t.Logf("Q6_K WMMA prefill vs scalar GEMV: rel-RMS %.3e", rel)
	if rel > 2e-2 {
		t.Fatalf("Q6_K WMMA diverges: rel-RMS %.3e", rel)
	}
}

func benchQ6KPrefill(b *testing.B, M, K, N int, wmma bool) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(3))
	w := tensor.New(tensor.F64, tensor.Shape{K, N})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = rng.NormFloat64() * 0.1
	}
	qi, err := quantQ6K(w)
	if err != nil {
		b.Fatal(err)
	}
	rq := qi.(*cuda.ResidentBQ6K)
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
		b.Skipf("q6k: %v", err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		run()
	}
	cuda.GraphSync()
}

func BenchmarkQ6KPrefill_M512_K4096_N4096_scalar(b *testing.B) {
	benchQ6KPrefill(b, 512, 4096, 4096, false)
}
func BenchmarkQ6KPrefill_M512_K4096_N4096_wmma(b *testing.B) {
	benchQ6KPrefill(b, 512, 4096, 4096, true)
}

// TestCUDAQ6KRecorderPrefillWMMAParity validates the production Q6_K prefill wiring: the public
// Recorder.QMatMulResidentQ6K at m>=q6kWMMAThreshold routes to the tensor-core WMMA path, and its
// result matches the trusted M-tiled GEMV (QMatMulInto) within f16-accum tolerance. m=130 (not %16)
// exercises the pad-to-144 + first-m-rows copy-out.
func TestCUDAQ6KRecorderPrefillWMMAParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 130, 512, 256 // M>=48 triggers WMMA, M not %16, K%256==0, N%16==0
	rng := rand.New(rand.NewSource(61))
	w := tensor.New(tensor.F64, tensor.Shape{K, N})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = rng.NormFloat64() * 0.1
	}
	qi, err := quantQ6K(w)
	must(t, err)
	rq := qi.(*cuda.ResidentBQ6K)
	defer rq.Free()
	rec, err := cuda.NewRecorder()
	must(t, err)
	defer rec.Free()
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
	must(t, rq.QMatMulInto(da, ref))                // MT GEMV oracle
	must(t, rec.QMatMulResidentQ6K(da, rq, got, M)) // production entry, M>=48 → WMMA
	must(t, rec.Wait())
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
	t.Logf("Q6_K recorder prefill WMMA vs MT oracle (M=%d→pad%d): rel-RMS %.3e", M, ((M+15)/16)*16, rel)
	if den == 0 || math.IsNaN(rel) {
		t.Fatalf("degenerate reference (den=%g rel=%g)", den, rel)
	}
	if rel > 2e-2 {
		t.Fatalf("Q6_K recorder WMMA diverges: rel-RMS %.3e", rel)
	}
}

// TestCUDAQ6KRecorderAccParity validates the fused Q6_K decode-residual epilogue
// (QMatMulResidentAccQ6K, beta=1) is BIT-EXACT to the un-fused fallback (QMatMulResidentQ6K
// beta=0 into scratch, then residual add). Both compute r + a·W as a single IEEE f32 add, so
// they must match exactly — this is what lets recordAdd fuse the Q6_K o/down projections at
// m=1 decode (Q4_K_M's attn_v/ffn_down) without changing the generic Decoder's output.
func TestCUDAQ6KRecorderAccParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const K, N = 512, 64 // m=1 decode path (scalar cu_qmatmul_q6k), K%256==0
	rng := rand.New(rand.NewSource(77))
	w := tensor.New(tensor.F64, tensor.Shape{K, N})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = rng.NormFloat64() * 0.1
	}
	qi, err := quantQ6K(w)
	must(t, err)
	rq := qi.(*cuda.ResidentBQ6K)
	defer rq.Free()
	rec, err := cuda.NewRecorder()
	must(t, err)
	defer rec.Free()
	af := make([]float32, K)
	for i := range af {
		af[i] = float32(rng.NormFloat64()) * 0.2
	}
	da, _ := cuda.NewDeviceF32(1, K)
	must(t, da.UploadF32(af))
	defer da.Free()
	resid := make([]float32, N)
	for i := range resid {
		resid[i] = float32(rng.NormFloat64())
	}

	// un-fused fallback: GEMV(beta=0) into scratch, then host residual add
	scratch, _ := cuda.NewDeviceF32(1, N)
	defer scratch.Free()
	must(t, scratch.UploadF32(make([]float32, N))) // beta=0 GEMV overwrites, but 0*stale-Inf=NaN — zero first
	must(t, rec.QMatMulResidentQ6K(da, rq, scratch, 1))
	must(t, rec.Wait())
	sc := make([]float32, N)
	scratch.DownloadF32(sc)
	want := make([]float32, N)
	for i := range want {
		want[i] = resid[i] + sc[i]
	}

	// fused: preload dst with residual, beta=1 accumulate in-kernel
	dst, _ := cuda.NewDeviceF32(1, N)
	defer dst.Free()
	must(t, dst.UploadF32(resid))
	must(t, rec.QMatMulResidentAccQ6K(da, rq, dst, 1))
	must(t, rec.Wait())
	got := make([]float32, N)
	dst.DownloadF32(got)

	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("Q6_K fused acc not bit-exact at %d: fused %g vs fallback %g", i, got[i], want[i])
		}
	}
	t.Logf("Q6_K fused decode-residual epilogue bit-exact vs fallback across N=%d", N)
}
