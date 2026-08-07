//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// TestResidentBQ8FromBlocksParity: the direct Q8_0-blocks→ResidentBQ8 path (no f32 round-trip) must
// produce a matmul result equivalent to the f32 dequant+requant path it replaces in cudaUploadQWeight.
// (Direct is actually HIGHER fidelity — it keeps the GGUF's own quants; the round-trip re-rounds — so
// both are valid Q8 representations of the same weight and agree to Q8 rounding.)
func TestResidentBQ8FromBlocksParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	k, n := 256, 96 // K multiple of 32; N arbitrary
	// A weight in GGUF [Out=N, In=K] orientation.
	wt := tensor.New(tensor.F32, tensor.Shape{n, k})
	wf := wt.Storage().F32()
	for i := range wf {
		wf[i] = float32((i*13+7)%29-14) * 0.07
	}
	q8, err := gguf.Quantize(wt, gguf.Q8_0)
	if err != nil {
		t.Fatalf("Quantize Q8_0: %v", err)
	}

	// Path A: direct from raw Q8_0 blocks.
	ra, err := cuda.NewResidentBQ8FromBlocks(q8, k, n)
	if err != nil {
		t.Fatalf("NewResidentBQ8FromBlocks: %v", err)
	}
	defer ra.Free()

	// Path B: the f32 round-trip it replaces — dequant [N,K]→f32, transpose to [K,N], NewResidentBQ8.
	f32, err := (gguf.QuantTensor{Data: q8, GGType: 8, Shape: tensor.Shape{n, k}}).Dequantize()
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}
	tin, err := f32.Transpose(0, 1)
	if err != nil {
		t.Fatalf("transpose: %v", err)
	}
	rb, err := cuda.NewResidentBQ8(tin)
	if err != nil {
		t.Fatalf("NewResidentBQ8: %v", err)
	}
	defer rb.Free()

	// Same activation through both.
	af := make([]float32, k)
	for i := range af {
		af[i] = float32((i*5+2)%11-5) * 0.15
	}
	da, err := cuda.NewDeviceF32(1, k)
	if err != nil {
		t.Fatal(err)
	}
	defer da.Free()
	if err := da.UploadF32(af); err != nil {
		t.Fatal(err)
	}
	oa, err := ra.QMatMulDevice(da)
	if err != nil {
		t.Fatalf("QMatMul A: %v", err)
	}
	defer oa.Free()
	ob, err := rb.QMatMulDevice(da)
	if err != nil {
		t.Fatalf("QMatMul B: %v", err)
	}
	defer ob.Free()
	ha, hb := make([]float32, n), make([]float32, n)
	if err := oa.DownloadF32(ha); err != nil {
		t.Fatal(err)
	}
	if err := ob.DownloadF32(hb); err != nil {
		t.Fatal(err)
	}
	var maxAbs, maxRef float64
	for i := range ha {
		if d := math.Abs(float64(ha[i] - hb[i])); d > maxAbs {
			maxAbs = d
		}
		if r := math.Abs(float64(hb[i])); r > maxRef {
			maxRef = r
		}
	}
	t.Logf("direct-from-blocks vs f32-roundtrip: maxAbs=%.6g (maxRef=%.4g)", maxAbs, maxRef)
	// Both are Q8 quantizations of the same f32 weight; agreement to Q8 rounding of an O(maxRef) dot.
	if maxAbs > 1e-3*maxRef+1e-4 {
		t.Fatalf("parity: maxAbs %g exceeds tol (maxRef=%g)", maxAbs, maxRef)
	}
}
