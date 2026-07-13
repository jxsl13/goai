//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// §T413: the record-mode quantized matmul — Recorder.QMatMulResident over a resident Q8_0 weight —
// matches the existing (ref-validated) ResidentQWeight.QMatMul, and chains with other record-mode
// ops (SiLU) in one command buffer. This is the quantized batched-decode building block: a Q4/Q8
// model's decode step can run as one command buffer over resident quantized weights.
func TestRecorderQMatMulResident(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const m, k, n = 3, 64, 24 // k multiple of 32 (Q8_0 block)
	// weight [n,k] as an f32 tensor, quantized to Q8_0 bytes.
	w := tensor.New(tensor.F32, tensor.Shape{n, k})
	ws := w.Storage().F32()
	for i := range ws {
		ws[i] = float32((i%17)-8) * 0.05
	}
	wq, err := gguf.Quantize(w, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	rw, err := UploadQWeightQ8_0(wq, n, k)
	if err != nil {
		t.Fatal(err)
	}
	defer rw.Close()

	xs := make([]float32, m*k)
	for i := range xs {
		xs[i] = float32((i%13)-6) * 0.1
	}

	// reference: the existing per-op resident path (itself cross-validated against ref).
	xt := tensor.New(tensor.F32, tensor.Shape{m, k})
	copy(xt.Storage().F32(), xs)
	refOut, err := rw.QMatMul(xt)
	if err != nil {
		t.Fatal(err)
	}

	// recorder: qmatmul then SiLU chained in ONE command buffer over device buffers.
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
		// the chained SiLU consumed the qmatmul result inside the same command buffer.
		silu := want / (1.0 + math.Exp(-want))
		if math.Abs(float64(gotAct[i])-silu) > 1e-3*math.Max(1, math.Abs(silu)) {
			t.Fatalf("chained silu[%d]: %v vs want %v", i, gotAct[i], silu)
		}
	}
}
