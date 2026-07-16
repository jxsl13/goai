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

// TestCUDANewResidentQuantDispatch: the public NewResidentQuant dispatches a GGUF quant tensor to the
// right native resident. For Q4_K it must produce the SAME GEMV as the direct FromBlocks constructor;
// a format without a native kernel (Q8_0) must return an error so the caller can fall back.
func TestCUDANewResidentQuantDispatch(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48
	rng := rand.New(rand.NewSource(83))
	w := tensor.New(tensor.F32, tensor.Shape{N, K}) // [out,in]
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	blocks, err := gguf.Quantize(w, gguf.Q4_K)
	must(t, err)

	viaDispatch, err := cuda.NewResidentQuant(gguf.QuantTensor{Data: blocks, GGType: 12, Shape: tensor.Shape{N, K}})
	must(t, err)
	defer viaDispatch.Free()
	direct, err := cuda.NewResidentBQ4KFromBlocks(blocks, K, N)
	must(t, err)
	defer direct.Free()

	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	da, err := cuda.NewDeviceF32(1, K)
	must(t, err)
	defer da.Free()
	must(t, da.UploadF32(a))
	o1, err := cuda.NewDeviceF32(1, N)
	must(t, err)
	defer o1.Free()
	o2, err := cuda.NewDeviceF32(1, N)
	must(t, err)
	defer o2.Free()
	must(t, viaDispatch.QMatMulInto(da, o1))
	must(t, direct.QMatMulInto(da, o2))
	h1, err := o1.ToHost()
	must(t, err)
	h2, err := o2.ToHost()
	must(t, err)
	for n := 0; n < N; n++ {
		if math.Abs(h1.AtF64(0, n)-h2.AtF64(0, n)) > 1e-6 {
			t.Fatalf("dispatch Q4_K result != direct at col %d: %v vs %v", n, h1.AtF64(0, n), h2.AtF64(0, n))
		}
	}
	t.Logf("NewResidentQuant Q4_K dispatch == direct constructor (%d cols identical)", N)

	// A format without a native kernel must error (caller dequantizes + uses Q8).
	if _, err := cuda.NewResidentQuant(gguf.QuantTensor{Data: blocks, GGType: 8, Shape: tensor.Shape{N, K}}); err == nil {
		t.Fatal("NewResidentQuant should error on ggml type 8 (Q8_0, no native kernel)")
	}
}
