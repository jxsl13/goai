//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Q8_0-quantized matmul must match the f32 matmul within the quantization error
// budget (symmetric per-32-block int8 → ~1/127 relative per weight, averaging
// down over the K-sum). Validates the transposed-quant layout, per-block scales,
// and the GEMV kernel against the exact same weights run through MatMulDevice.
func TestCUDAQ8MatMulMatchesF32(t *testing.T) {
	skipNoGPU(t)
	for _, dims := range []struct{ m, k, n int }{
		{1, 2048, 2048}, // decode shape (GEMV)
		{4, 256, 512},   // small GQA-ish
		{8, 5632, 2048}, // FFN down shape
	} {
		a := bench.RandF32(tensor.Shape{dims.m, dims.k}, 1)
		w := bench.RandF32(tensor.Shape{dims.k, dims.n}, 2)

		da, _ := cuda.UploadF32(a)
		rf, _ := cuda.NewResidentB(w)
		rq, err := cuda.NewResidentBQ8(w)
		if err != nil {
			t.Fatal(err)
		}
		fRes, err := rf.MatMulDevice(da)
		if err != nil {
			t.Fatal(err)
		}
		qRes, err := rq.QMatMulDevice(da)
		if err != nil {
			t.Fatal(err)
		}
		fHost, _ := fRes.ToHost()
		qHost, _ := qRes.ToHost()

		var maxRel, sumAbs, sumRef float64
		for i := 0; i < dims.m; i++ {
			for j := 0; j < dims.n; j++ {
				f := fHost.AtF64(i, j)
				q := qHost.AtF64(i, j)
				sumAbs += math.Abs(q - f)
				sumRef += math.Abs(f)
				if r := math.Abs(q-f) / (math.Abs(f) + 1); r > maxRel {
					maxRel = r
				}
			}
		}
		relL1 := sumAbs / (sumRef + 1e-9)
		// Q8 over a K-sum: aggregate L1 relative error should sit well under 1%.
		if relL1 > 0.02 {
			t.Fatalf("[%dx%dx%d] Q8 vs f32 L1 rel err %.4f too high", dims.m, dims.k, dims.n, relL1)
		}
		t.Logf("[%dx%dx%d] Q8 matmul: L1 rel err %.4f, max elem rel %.4f", dims.m, dims.k, dims.n, relL1, maxRel)

		da.Free()
		rf.Free()
		rq.Free()
		fRes.Free()
		qRes.Free()
	}
}

// Decode-shape GEMV: Q8 (int8 weights, 4× less bandwidth) vs f32 cuBLAS.
func benchMatmulShape(b *testing.B, m, k, n int, q8 bool) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	a := bench.RandF32(tensor.Shape{m, k}, 1)
	w := bench.RandF32(tensor.Shape{k, n}, 2)
	da, _ := cuda.UploadF32(a)
	defer da.Free()
	if q8 {
		rq, _ := cuda.NewResidentBQ8(w)
		defer rq.Free()
		b.ResetTimer()
		for range b.N {
			o, err := rq.QMatMulDevice(da)
			if err != nil {
				b.Fatal(err)
			}
			o.ToHost()
			o.Free()
		}
	} else {
		rf, _ := cuda.NewResidentB(w)
		defer rf.Free()
		b.ResetTimer()
		for range b.N {
			o, err := rf.MatMulDevice(da)
			if err != nil {
				b.Fatal(err)
			}
			o.ToHost()
			o.Free()
		}
	}
}

func BenchmarkDecodeGemvF32(b *testing.B) { benchMatmulShape(b, 1, 2048, 2048, false) }
func BenchmarkDecodeGemvQ8(b *testing.B)  { benchMatmulShape(b, 1, 2048, 2048, true) }
