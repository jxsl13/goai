//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// The fused-QKV projection reproduces three separate ResidentBF16.MatMulDevice calls: the
// concatenation + column-split only reorders storage, so under EXACT (f32) accumulation the
// only difference is cuBLAS's N-dependent tile choice = f32 summation order (~1e-5, §V-tol).
// Under the GeForce default f16-accumulate, the difference is the larger accumulate-rounding
// envelope (the same class the f16-acc prefill already accepts, norm-rel-RMS ~2-5e-3). The
// test forces f32-acc to isolate and prove the concat+split arithmetic is correct. GQA shapes:
// K=2048, Nq=2048 (32 heads·64), Nk=Nv=256 (4 kv-heads·64) — TinyLlama's projection widths.
func TestCUDAF16QKVFusedParity(t *testing.T) {
	skipNoGPU(t)
	t.Setenv("GOAI_CUDA_F16ACC", "0") // exact f32 accumulate → isolates split correctness from f16 rounding
	const K, Nq, Nk, Nv, M = 2048, 2048, 256, 256, 37
	rng := rand.New(rand.NewSource(91))
	mkW := func(n int) *tensor.Tensor {
		w := tensor.New(tensor.F32, tensor.Shape{K, n})
		wf := w.Storage().F32()
		for i := range wf {
			wf[i] = float32(rng.NormFloat64()) * 0.05
		}
		return w
	}
	wq, wk, wv := mkW(Nq), mkW(Nk), mkW(Nv)
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	da, err := cuda.NewDeviceF32(M, K)
	must(t, err)
	defer da.Free()
	must(t, da.UploadF32(a))

	// Separate path: three ResidentBF16 GEMMs.
	rq, err := cuda.NewResidentBF16(wq)
	must(t, err)
	defer rq.Free()
	rk, err := cuda.NewResidentBF16(wk)
	must(t, err)
	defer rk.Free()
	rv, err := cuda.NewResidentBF16(wv)
	must(t, err)
	defer rv.Free()
	sq, err := rq.MatMulDevice(da)
	must(t, err)
	defer sq.Free()
	sk, err := rk.MatMulDevice(da)
	must(t, err)
	defer sk.Free()
	sv, err := rv.MatMulDevice(da)
	must(t, err)
	defer sv.Free()
	hsq, err := sq.ToHost()
	must(t, err)
	hsk, err := sk.ToHost()
	must(t, err)
	hsv, err := sv.ToHost()
	must(t, err)

	// Fused path: one wide GEMM + column split.
	rqkv, err := cuda.NewResidentBF16QKV(wq, wk, wv)
	must(t, err)
	defer rqkv.Free()
	fq, fk, fv, err := rqkv.MatMulQKV(da)
	must(t, err)
	defer fq.Free()
	defer fk.Free()
	defer fv.Free()
	hfq, err := fq.ToHost()
	must(t, err)
	hfk, err := fk.ToHost()
	must(t, err)
	hfv, err := fv.ToHost()
	must(t, err)

	cmp := func(name string, sep, fused *tensor.Tensor, rows, cols int) {
		var maxAbs, num, den float64
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				s, f := sep.AtF64(r, c), fused.AtF64(r, c)
				d := math.Abs(s - f)
				if d > maxAbs {
					maxAbs = d
				}
				num += d * d
				den += s * s
			}
		}
		relRMS := math.Sqrt(num / den)
		t.Logf("%s fused vs separate: max abs %.3g, norm-rel-RMS %.3g", name, maxAbs, relRMS)
		// f32-accumulate: only cuBLAS's N-dependent tiling differs → f32 summation-order noise.
		if maxAbs > 1e-3 {
			t.Fatalf("%s fused-QKV diverges from separate GEMM: max abs %.3g (>1e-3, not summation-order noise)", name, maxAbs)
		}
	}
	cmp("q", hsq, hfq, M, Nq)
	cmp("k", hsk, hfk, M, Nk)
	cmp("v", hsv, hfv, M, Nv)
}

// End-to-end wall-time A/B of the fused-QKV projection vs three separate ResidentBF16 GEMMs,
// including the column-split copies, at a real prefill batch (M=128). This is the shipped-path
// number the BenchmarkF16Proj_* GEMM-only micro-bench predicts (~1.39×).
func benchQKVProj(b *testing.B, fused bool) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const K, Nq, Nk, Nv, M = 2048, 2048, 256, 256, 128
	rng := rand.New(rand.NewSource(5))
	mkW := func(n int) *tensor.Tensor {
		w := tensor.New(tensor.F32, tensor.Shape{K, n})
		wf := w.Storage().F32()
		for i := range wf {
			wf[i] = float32(rng.NormFloat64()) * 0.05
		}
		return w
	}
	wq, wk, wv := mkW(Nq), mkW(Nk), mkW(Nv)
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	da, _ := cuda.NewDeviceF32(M, K)
	defer da.Free()
	da.UploadF32(a)

	var run func()
	if fused {
		rqkv, _ := cuda.NewResidentBF16QKV(wq, wk, wv)
		defer rqkv.Free()
		run = func() {
			q, k, v, err := rqkv.MatMulQKV(da)
			if err != nil {
				b.Fatal(err)
			}
			q.Free()
			k.Free()
			v.Free()
		}
	} else {
		rq, _ := cuda.NewResidentBF16(wq)
		rk, _ := cuda.NewResidentBF16(wk)
		rv, _ := cuda.NewResidentBF16(wv)
		defer rq.Free()
		defer rk.Free()
		defer rv.Free()
		run = func() {
			q, _ := rq.MatMulDevice(da)
			k, _ := rk.MatMulDevice(da)
			v, _ := rv.MatMulDevice(da)
			q.Free()
			k.Free()
			v.Free()
		}
	}
	run()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		run()
	}
	cuda.GraphSync()
	b.StopTimer()
}

func BenchmarkQKVProjSeparate(b *testing.B) { benchQKVProj(b, false) }
func BenchmarkQKVProjFused(b *testing.B)    { benchQKVProj(b, true) }
