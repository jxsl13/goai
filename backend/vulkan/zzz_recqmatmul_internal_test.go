//go:build vulkan && cgo

package vulkan

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// §T414 (parity with metal §T413): the record-mode quantized matmul — Recorder.QMatMulResident over
// a resident Q8_0 weight — matches the existing (ref-validated) ResidentQWeight.QMatMul, and chains
// with a record-mode SiLU in one command buffer.
func TestVulkanRecorderQMatMulResident(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const m, k, n = 3, 64, 24
	w := tensor.New(tensor.F32, tensor.Shape{n, k})
	ws := w.Storage().F32()
	for i := range ws {
		ws[i] = float32((i%17)-8) * 0.05
	}
	wq, err := gguf.Quantize(w, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	rwi, err := Backend{}.UploadQuant(wq, uint32(gguf.Q8_0), n, k)
	if err != nil {
		t.Fatal(err)
	}
	rw := rwi.(*ResidentQWeight)
	defer rw.Close()

	xs := make([]float32, m*k)
	for i := range xs {
		xs[i] = float32((i%13)-6) * 0.1
	}
	xt := tensor.New(tensor.F32, tensor.Shape{m, k})
	copy(xt.Storage().F32(), xs)
	refOut, err := rw.QMatMul(xt)
	if err != nil {
		t.Fatal(err)
	}

	x, err := NewDeviceBufferF32(xs)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Release()
	o, _ := NewDeviceBufferF32(make([]float32, m*n))
	defer o.Release()
	act, _ := NewDeviceBufferF32(make([]float32, m*n))
	defer act.Release()

	r, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.QMatMulResident(x, rw, o, m); err != nil {
		t.Fatal(err)
	}
	if err := r.Unary(o, act, unarySiLU); err != nil {
		t.Fatal(err)
	}
	if err := r.Finish(); err != nil {
		t.Fatal(err)
	}
	r.Free()

	got := make([]float32, m*n)
	if err := o.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	gotAct := make([]float32, m*n)
	if err := act.DownloadF32(gotAct); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		want := refOut.AtF64(i/n, i%n)
		if math.Abs(float64(got[i])-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Fatalf("qmatmul[%d]: recorder %v vs resident %v", i, got[i], want)
		}
		silu := want / (1.0 + math.Exp(-want))
		if math.Abs(float64(gotAct[i])-silu) > 1e-3*math.Max(1, math.Abs(silu)) {
			t.Fatalf("chained silu[%d]: %v vs want %v", i, gotAct[i], silu)
		}
	}
}
