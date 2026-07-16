//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDAResidentBQ8ServesDecodeAndMMQ proves a SINGLE resident Q8 weight serves both the decode
// GEMV (QMatMulDevice, f32 activations) and the prefill int8 MMQ GEMM (MatMulMMQDevice, int8
// activations) — from the exact same device buffers, no second weight copy. Both track a true f32
// GEMM within their quantization budgets; the MMQ path also matches the AccInto residual twin.
func TestCUDAResidentBQ8ServesDecodeAndMMQ(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const K, N, M = 256, 128, 64
	rng := rand.New(rand.NewSource(4242))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.5
	}
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	r, err := cuda.NewResidentBQ8(w)
	if err != nil {
		t.Fatalf("NewResidentBQ8: %v", err)
	}
	defer r.Free()
	da, err := cuda.NewDeviceF32(M, K)
	if err != nil {
		t.Fatalf("NewDeviceF32: %v", err)
	}
	defer da.Free()
	if err := da.UploadF32(a); err != nil {
		t.Fatalf("UploadF32: %v", err)
	}

	// Same resident weight, two paths.
	dgemv, err := r.QMatMulDevice(da)
	if err != nil {
		t.Fatalf("QMatMulDevice (GEMV): %v", err)
	}
	defer dgemv.Free()
	dmmq, err := r.MatMulMMQDevice(da)
	if err != nil {
		t.Fatalf("MatMulMMQDevice: %v", err)
	}
	defer dmmq.Free()
	hgemv, err := dgemv.ToHost()
	if err != nil {
		t.Fatalf("ToHost gemv: %v", err)
	}
	hmmq, err := dmmq.ToHost()
	if err != nil {
		t.Fatalf("ToHost mmq: %v", err)
	}

	var seGemv, seMMQ, sr float64
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var ref float64
			for k := 0; k < K; k++ {
				ref += float64(a[m*K+k]) * float64(wf[k*N+n])
			}
			dg := hgemv.AtF64(m, n) - ref
			dm := hmmq.AtF64(m, n) - ref
			seGemv += dg * dg
			seMMQ += dm * dm
			sr += ref * ref
		}
	}
	rmsGemv := math.Sqrt(seGemv / sr)
	rmsMMQ := math.Sqrt(seMMQ / sr)
	t.Logf("one resident Q8 weight: GEMV-vs-f32 %.4f | MMQ-vs-f32 %.4f (same q/scales buffers)", rmsGemv, rmsMMQ)
	if rmsGemv > 1e-2 {
		t.Errorf("Q8 GEMV vs f32 RMS %.4f too high", rmsGemv)
	}
	if rmsMMQ > 2e-2 {
		t.Errorf("Q8 MMQ vs f32 RMS %.4f too high (int8 act + Q8 weight budget)", rmsMMQ)
	}

	// AccInto residual twin: c += a·B on a seed.
	c0 := make([]float32, M*N)
	for i := range c0 {
		c0[i] = float32(rng.NormFloat64())
	}
	dc, err := cuda.NewDeviceF32(M, N)
	if err != nil {
		t.Fatalf("NewDeviceF32 c: %v", err)
	}
	defer dc.Free()
	if err := dc.UploadF32(c0); err != nil {
		t.Fatalf("UploadF32 c: %v", err)
	}
	if err := r.MatMulMMQAccInto(da, dc); err != nil {
		t.Fatalf("MatMulMMQAccInto: %v", err)
	}
	hc, err := dc.ToHost()
	if err != nil {
		t.Fatalf("ToHost c: %v", err)
	}
	var se2, sr2 float64
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			ref := float64(c0[m*N+n]) + hmmq.AtF64(m, n)
			d := hc.AtF64(m, n) - ref
			se2 += d * d
			sr2 += ref * ref
		}
	}
	if rms := math.Sqrt(se2 / sr2); rms > 1e-6 {
		t.Errorf("MatMulMMQAccInto disagrees with MatMulMMQDevice + residual: RMS %.3g", rms)
	}
	t.Log("SINGLE RESIDENT Q8 WEIGHT SERVES DECODE-GEMV + PREFILL-MMQ (no second copy) — the unification")
}
