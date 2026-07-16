//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDAResidentMMQParity validates ResidentMMQ (int8 MMQ prefill weight) end-to-end vs a true
// f32 GEMM: weights per-block int8 [N][K], activations device-quantized per-row, f32 accumulate.
// M=100 is deliberately NOT a multiple of 64 to exercise the internal M-padding. Tolerance covers
// the int8 quantization error (Q8_0-class, ~1% norm-rel-RMS), not just summation order.
func TestCUDAResidentMMQParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const K, N, M = 256, 128, 100
	rng := rand.New(rand.NewSource(101))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.5
	}
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rw, err := cuda.NewResidentMMQ(w)
	if err != nil {
		t.Fatalf("NewResidentMMQ: %v", err)
	}
	defer rw.Free()
	da, err := cuda.NewDeviceF32(M, K)
	if err != nil {
		t.Fatalf("NewDeviceF32: %v", err)
	}
	defer da.Free()
	if err := da.UploadF32(a); err != nil {
		t.Fatalf("UploadF32: %v", err)
	}
	dout, err := rw.MatMulDevice(da)
	if err != nil {
		t.Fatalf("MatMulDevice: %v", err)
	}
	defer dout.Free()
	got, err := dout.ToHost()
	if err != nil {
		t.Fatalf("ToHost: %v", err)
	}
	if got.Shape()[0] != M || got.Shape()[1] != N {
		t.Fatalf("output shape %v, want [%d %d]", got.Shape(), M, N)
	}

	var sqErr, sqRef float64
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var ref float64
			for k := 0; k < K; k++ {
				ref += float64(a[m*K+k]) * float64(wf[k*N+n])
			}
			d := got.AtF64(m, n) - ref
			sqErr += d * d
			sqRef += ref * ref
		}
	}
	rms := math.Sqrt(sqErr / sqRef)
	t.Logf("ResidentMMQ vs f32 GEMM: norm-rel-RMS %.4g (M=%d not %%64 -> padding exercised)", rms, M)
	if rms > 2e-2 {
		t.Fatalf("ResidentMMQ accuracy too poor: RMS %.3g > 2e-2", rms)
	}

	// MatMulAccInto: c += a·B on top of a seeded residual. Verify c_after ≈ c0 + (a·B).
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
	if err := rw.MatMulAccInto(da, dc); err != nil {
		t.Fatalf("MatMulAccInto: %v", err)
	}
	gotAcc, err := dc.ToHost()
	if err != nil {
		t.Fatalf("ToHost c: %v", err)
	}
	var sqE2, sqR2 float64
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			ref := float64(c0[m*N+n]) + got.AtF64(m, n) // seeded residual + the validated a·B
			d := gotAcc.AtF64(m, n) - ref
			sqE2 += d * d
			sqR2 += ref * ref
		}
	}
	rmsAcc := math.Sqrt(sqE2 / sqR2)
	t.Logf("ResidentMMQ MatMulAccInto (c += a·B) vs c0 + MatMulDevice: norm-rel-RMS %.4g", rmsAcc)
	if rmsAcc > 1e-6 {
		t.Fatalf("MatMulAccInto disagrees with MatMulDevice + residual: RMS %.3g", rmsAcc)
	}
	t.Log("RESIDENT MMQ PREFILL WEIGHT CORRECT (int8, M-padding, ~Q8_0 accuracy, residual-fused AccInto)")
}
